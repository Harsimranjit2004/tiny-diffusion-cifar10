"""
src/tiny_diffusion/models/unet.py

Full U-Net assembly — split out from Phase 1's combined file.

WHY THE IMPORTS BELOW LOOK DIFFERENT FROM THE ORIGINAL FILE: Phase 1 was
one flat script where every class lived in the same namespace. Now that
ResBlock/SelfAttentionBlock/Downsample/Upsample live in blocks.py and the
embeddings live in embeddings.py, UNet needs to import them explicitly —
this is the real benefit of the package split: import errors here would
immediately reveal if a class was renamed or moved, rather than silently
referencing a name that happened to exist in the same script.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from tiny_diffusion.models.blocks import Downsample, ResBlock, SelfAttentionBlock, Upsample
from tiny_diffusion.models.config import ModelConfig
from tiny_diffusion.models.embeddings import ClassEmbedding, SinusoidalTimeEmbedding


class UNet(nn.Module):
    """
    Full U-Net for conditional diffusion on CIFAR-10.

    Predicts noise eps_theta(x_t, t, c) given:
      x_t: noisy image at timestep t  [B, 3, 32, 32]
      t:   timestep integer            [B]
      c:   class label (0-9, or 10 for null/unconditional)  [B]
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Shorthand
        C = config.base_channels  # 128
        M = config.channel_mult  # [1, 2, 4, 8]
        T_embed = config.time_embed_dim  # 512
        num_res = config.num_res_blocks  # 2
        attn_res = config.attention_resolutions  # [8, 4]
        num_groups = config.num_groups  # 32
        num_classes = config.num_classes  # 10

        # ── Conditioning modules ───────────────────────────────────────────
        self.time_embed = SinusoidalTimeEmbedding(T_embed)
        self.class_embed = ClassEmbedding(num_classes, T_embed, config.cfg_dropout)
        # Note: time and class embeddings are simply ADDED.
        # Both are T_embed=512 dimensional, so addition is shape-compatible.
        # No extra projection needed.

        # ── Stem: first conv to get from 3 channels to base_channels ──────
        # kernel_size=3, padding=1 preserves spatial dimensions.
        # This is just a channel-count change; no resolution change here.
        self.stem = nn.Conv2d(config.in_channels, C, kernel_size=3, padding=1)

        # ── Build down path ────────────────────────────────────────────────
        # Store in nn.ModuleList so PyTorch tracks all parameters.
        # Also store skip connection channel counts for building the up path.
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        skip_channels = []  # track channels at each skip connection

        current_channels = C  # start at base_channels=128
        current_res = config.image_size  # start at 32

        for level, mult in enumerate(M):
            level_channels = C * mult  # 128, 256, 512, 1024

            # Build num_res_blocks ResBlocks for this level
            for block_idx in range(num_res):
                use_attention = current_res in attn_res
                block = self._make_res_attn_block(
                    in_ch=current_channels,
                    out_ch=level_channels,
                    cond_dim=T_embed,
                    num_groups=num_groups,
                    use_attention=use_attention,
                    num_heads=config.num_heads,
                )
                self.down_blocks.append(block)
                skip_channels.append(level_channels)  # record output channels
                current_channels = level_channels

            # Add downsampling AFTER all ResBlocks at this level,
            # EXCEPT at the last level (bottleneck, no downsampling after)
            if level < len(M) - 1:
                self.down_samples.append(Downsample(current_channels))
                current_res = current_res // 2  # 32 → 16 → 8 → 4

        # ── Bottleneck ─────────────────────────────────────────────────────
        # Two ResBlocks with self-attention in between.
        # This is the lowest-resolution, highest-channel processing stage.
        # All 4x4=16 positions attend to each other directly.
        bottleneck_channels = current_channels  # 1024
        self.bottleneck_res1 = ResBlock(
            bottleneck_channels, bottleneck_channels, T_embed, num_groups
        )
        self.bottleneck_attn = SelfAttentionBlock(bottleneck_channels, config.num_heads, num_groups)
        self.bottleneck_res2 = ResBlock(
            bottleneck_channels, bottleneck_channels, T_embed, num_groups
        )

        # ── Build up path ──────────────────────────────────────────────────
        # Mirror of the down path but with skip connections.
        # Each ResBlock receives (own channels + skip channels) as input.
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        # Reverse the level order for the up path
        for level, mult in reversed(list(enumerate(M))):
            level_channels = C * mult

            # Build num_res_blocks ResBlocks for this level
            for block_idx in range(num_res):
                use_attention = current_res in attn_res
                # Each up-block receives (current_channels + skip_channels) input
                # because we concatenate the skip connection
                skip_ch = skip_channels.pop()
                block = self._make_res_attn_block(
                    in_ch=current_channels + skip_ch,  # concat skip connection
                    out_ch=level_channels,
                    cond_dim=T_embed,
                    num_groups=num_groups,
                    use_attention=use_attention,
                    num_heads=config.num_heads,
                )
                self.up_blocks.append(block)
                current_channels = level_channels

            # Add upsampling BEFORE moving to the next (coarser → finer) level,
            # EXCEPT at the first level (level=0, highest res, no upsample needed)
            if level > 0:
                self.up_samples.append(Upsample(current_channels))
                current_res = current_res * 2

        # ── Output head ────────────────────────────────────────────────────
        # GroupNorm → SiLU → Conv3x3 → out_channels (3 for RGB noise prediction)
        # WHY GroupNorm before the final conv:
        #   The accumulated features may have different scales.
        #   Normalizing before the final conv gives the head a cleaner signal.
        self.out_norm = nn.GroupNorm(num_groups, current_channels)
        self.out_conv = nn.Conv2d(current_channels, config.out_channels, kernel_size=3, padding=1)
        # Zero-init the final conv: at initialization, model predicts zero noise.
        # This is a safe starting point — zero noise prediction = x_t unchanged.
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def _make_res_attn_block(
        self,
        in_ch: int,
        out_ch: int,
        cond_dim: int,
        num_groups: int,
        use_attention: bool,
        num_heads: int,
    ) -> nn.Module:
        """
        Create a (ResBlock, optional SelfAttentionBlock) pair as a single module.
        Wraps them in an nn.Sequential-like container.
        """
        return ResAttnBlock(in_ch, out_ch, cond_dim, num_groups, use_attention, num_heads)

    def forward(
        self,
        x: torch.Tensor,  # [B, 3, H, W] noisy image
        t: torch.Tensor,  # [B] integer timesteps
        c: torch.Tensor,  # [B] class labels (0-9, or 10 for null)
        force_uncond: bool = False,  # for CFG inference
    ) -> torch.Tensor:
        """
        Predict noise eps_theta(x_t, t, c).

        Returns:
            predicted_noise: [B, 3, H, W] — same shape as input
        """
        # ── Step 1: Build conditioning vector ─────────────────────────────
        # time_emb: [B, T_embed]  — encodes "how noisy is this?"
        # class_emb: [B, T_embed] — encodes "what class is this?" (or null)
        # cond = their sum: [B, T_embed]
        # This single vector flows into every ResBlock via AdaGN.
        time_emb = self.time_embed(t)  # [B, 512]
        class_emb = self.class_embed(c, force_uncond=force_uncond)  # [B, 512]
        cond = time_emb + class_emb  # [B, 512]

        # ── Step 2: Stem ───────────────────────────────────────────────────
        h = self.stem(x)  # [B, 3, 32, 32] → [B, 128, 32, 32]

        # ── Step 3: Down path ──────────────────────────────────────────────
        skips = []  # store feature maps for skip connections
        down_block_idx = 0
        down_sample_idx = 0

        for level, mult in enumerate(self.config.channel_mult):
            for _ in range(self.config.num_res_blocks):
                h = self.down_blocks[down_block_idx](h, cond)
                skips.append(h)  # save for later
                down_block_idx += 1

            if level < len(self.config.channel_mult) - 1:
                h = self.down_samples[down_sample_idx](h)
                down_sample_idx += 1

        # ── Step 4: Bottleneck ─────────────────────────────────────────────
        h = self.bottleneck_res1(h, cond)
        h = self.bottleneck_attn(h)
        h = self.bottleneck_res2(h, cond)

        # ── Step 5: Up path ────────────────────────────────────────────────
        up_block_idx = 0
        up_sample_idx = 0

        for level, mult in reversed(list(enumerate(self.config.channel_mult))):
            for _ in range(self.config.num_res_blocks):
                skip = skips.pop()  # retrieve matching skip connection
                # Concatenate skip along channel dimension
                h = torch.cat([h, skip], dim=1)  # doubles channel count
                h = self.up_blocks[up_block_idx](h, cond)
                up_block_idx += 1

            if level > 0:
                h = self.up_samples[up_sample_idx](h)
                up_sample_idx += 1

        # ── Step 6: Output head ────────────────────────────────────────────
        h = self.out_norm(h)  # GroupNorm
        h = F.silu(h)  # SiLU activation
        h = self.out_conv(h)  # → [B, 3, 32, 32]

        return h  # predicted noise eps_theta


class ResAttnBlock(nn.Module):
    """Container for (ResBlock, optional SelfAttentionBlock) pair."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        cond_dim: int,
        num_groups: int,
        use_attention: bool,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, cond_dim, num_groups)
        self.attn = (
            SelfAttentionBlock(out_ch, num_heads, num_groups) if use_attention else nn.Identity()
        )
        self.use_attention = use_attention

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.res(x, cond)
        if self.use_attention:
            h = self.attn(h)
        return h
