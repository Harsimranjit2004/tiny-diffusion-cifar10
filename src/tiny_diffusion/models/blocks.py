"""
src/tiny_diffusion/models/blocks.py

ResBlock, SelfAttentionBlock, Downsample, Upsample — split out from
Phase 1's combined file. See Phase 0/Phase 1 for full design reasoning
on AdaGN conditioning, attention placement, and the strided-conv vs
bilinear-upsample decisions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ResBlock(nn.Module):
    """
    Residual block with AdaGN time+class conditioning.

    The conditioning vector (time + class embedding, shape [B, cond_dim])
    predicts per-channel scale and shift via AdaGN injection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,  # dimension of the conditioning vector (time_embed_dim)
        num_groups: int = 32,  # GroupNorm groups
        dropout: float = 0.1,  # dropout in the middle — regularization
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # ── First half: normalize → activate → conv ────────────────────────
        # GroupNorm before Conv is the standard order in modern diffusion nets.
        # WHY norm before conv (pre-norm) instead of after (post-norm):
        #   Pre-norm: gradients flow directly to x without going through norm.
        #   More stable training, especially deep into the network.
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # padding=1 on a 3x3 conv preserves spatial dimensions (H,W unchanged)

        # ── AdaGN conditioning projection ─────────────────────────────────
        # Projects cond_dim → 2 * out_channels (scale AND shift for each channel)
        # WHY 2*: we need one scale and one shift per output channel.
        # The * 2 then .chunk(2) pattern splits them cleanly.
        self.cond_proj = nn.Linear(cond_dim, out_channels * 2)
        # Initialize to zero: at the start of training, scale=0 and shift=0,
        # so the conditioning has no effect. The network learns to use it.
        # This "zero-init" trick stabilizes early training.
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

        # ── Second half: normalize → dropout → activate → conv ────────────
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # Zero-init the second conv's weights: this is the "ReZero" / "zero-output"
        # trick. At initialization, the ResBlock outputs exactly its input (identity).
        # This makes very deep networks trainable from the start.
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

        # ── Residual connection ────────────────────────────────────────────
        # If in_channels != out_channels, we need a projection to match dimensions.
        # 1x1 conv is the standard way to change channel count without spatial effect.
        if in_channels != out_channels:
            self.skip_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_proj = nn.Identity()  # no-op when channels already match

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    feature map, shape [B, in_channels, H, W]
            cond: conditioning vector (time + class), shape [B, cond_dim]
        Returns:
            out:  feature map, shape [B, out_channels, H, W]
        """
        # ── Save residual for skip connection ──────────────────────────────
        residual = x  # [B, in_channels, H, W]

        # ── First conv block ───────────────────────────────────────────────
        h = self.norm1(x)  # GroupNorm: normalize across channels
        h = F.silu(h)  # SiLU activation: smooth, no dead neurons
        h = self.conv1(h)  # [B, in_channels, H, W] → [B, out_channels, H, W]

        # ── AdaGN: inject time and class conditioning ──────────────────────
        # cond: [B, cond_dim] → project to [B, 2*out_channels]
        cond_out = self.cond_proj(cond)  # [B, 2*out_channels]

        # Split into scale and shift along channel dimension
        # scale: [B, out_channels],  shift: [B, out_channels]
        scale, shift = cond_out.chunk(2, dim=1)

        # Reshape for broadcasting against spatial feature maps:
        # [B, out_channels] → [B, out_channels, 1, 1]
        # The trailing 1s broadcast across H and W.
        scale = scale[:, :, None, None]  # [B, out_channels, 1, 1]
        shift = shift[:, :, None, None]  # [B, out_channels, 1, 1]

        # Apply AdaGN to normalized features:
        # h = h * (1 + scale) + shift
        # WHY (1 + scale) instead of just scale:
        #   At init, cond_proj weights=0, so scale=0.
        #   (1 + 0) = 1 → multiplication by 1 → identity at init.
        #   If we used just scale, scale=0 → multiply by 0 → zero output. Bad.
        h = h * (1 + scale) + shift  # [B, out_channels, H, W]

        # ── Second conv block ──────────────────────────────────────────────
        h = self.norm2(h)  # GroupNorm again
        h = F.silu(h)
        h = self.dropout(h)  # dropout for regularization (10%)
        h = self.conv2(h)  # [B, out_channels, H, W] — zero-init so identity at start

        # ── Residual connection ────────────────────────────────────────────
        # Project residual if needed (when in_channels != out_channels)
        return h + self.skip_proj(residual)


# =============================================================================
# STEP 6 — SELF-ATTENTION BLOCK
# =============================================================================
#
# PURPOSE: allow every spatial position to attend to every other position.
# This gives the model "global receptive field" — a pixel at position (0,0)
# can directly influence position (7,7) in a single layer.
#
# WHY WE NEED ATTENTION AT ALL:
#   Convolutions have a local receptive field (3x3 kernel = only sees neighbors).
#   To model global structure (e.g., the overall shape of an airplane), you need
#   many stacked conv layers for information to propagate across the image.
#   Attention does it in one step — every position directly sees every other.
#
# WHERE WE PLACE ATTENTION (from Phase 0):
#   ONLY at 8x8 and 4x4 resolutions. Reason:
#   - 32x32: 1024 tokens → attention cost 1024^2 = 1M per layer. Too slow.
#   - 16x16: 256 tokens → 65k. Borderline. We skip it for safety.
#   - 8x8:   64 tokens  → 4096. Affordable.
#   - 4x4:   16 tokens  → 256. Trivial.
#
# MULTI-HEAD ATTENTION:
#   Split channels into num_heads independent attention heads.
#   Each head learns to attend based on different "aspects" of the content.
#   head_dim = channels / num_heads
#   At 8x8 with 512 channels and 4 heads: head_dim = 128.
#   At 4x4 with 1024 channels and 4 heads: head_dim = 256.
#
# CONNECTION TO YOUR STABLE DIFFUSION IMPLEMENTATION:
#   Your SD implementation uses cross-attention (text as key/value, image as query).
#   Here we use SELF-attention (image attends to itself — Q, K, V all from image).
#   The mechanics are the same; the inputs are different.
#   If you had cross-attention in SD working, self-attention is simpler.


class SelfAttentionBlock(nn.Module):
    """
    Multi-head self-attention over spatial positions.
    Applied only at low-resolution feature maps (8x8, 4x4).

    Treats spatial positions as a sequence: [B, C, H, W] → [B, H*W, C]
    Runs attention over the H*W sequence dimension.
    Reshapes back: [B, H*W, C] → [B, C, H, W]
    """

    def __init__(self, channels: int, num_heads: int = 4, num_groups: int = 32):
        super().__init__()
        assert (
            channels % num_heads == 0
        ), f"channels ({channels}) must be divisible by num_heads ({num_heads})"

        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        # Scale factor for dot-product attention: 1/sqrt(head_dim)
        # WHY: the dot product Q·K can get large when head_dim is large,
        # pushing softmax into saturation (near-zero gradients).
        # Dividing by sqrt(head_dim) keeps the magnitude reasonable.
        self.scale = self.head_dim**-0.5

        # Pre-norm (GroupNorm before attention) — same reasoning as ResBlock.
        self.norm = nn.GroupNorm(num_groups, channels)

        # Single linear layer to produce Q, K, V all at once.
        # Output is 3*channels: first channels = Q, next = K, last = V.
        # WHY one fused projection instead of three separate:
        #   Fewer kernel launches = faster on GPU.
        #   The three projections are independent anyway, so fusing is safe.
        self.qkv_proj = nn.Linear(channels, channels * 3)

        # Output projection: after attention, project back to channels.
        # Zero-init: attention has no effect at initialization → stable training.
        self.out_proj = nn.Linear(channels, channels)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: feature map, shape [B, C, H, W]
        Returns:
            out: feature map, shape [B, C, H, W]  (same shape)
        """
        B, C, H, W = x.shape
        residual = x  # save for skip connection

        # ── Pre-norm ───────────────────────────────────────────────────────
        h = self.norm(x)  # [B, C, H, W]

        # ── Reshape: spatial map → sequence ───────────────────────────────
        # Flatten H*W into a sequence dimension for attention.
        # rearrange from einops makes this readable:
        #   'b c h w -> b (h w) c'
        #   means: for each batch, flatten h*w positions, put C last
        # The sequence length N = H*W (64 at 8x8, 16 at 4x4)
        h = rearrange(h, "b c h w -> b (h w) c")  # [B, N, C]

        # ── Q, K, V projections ────────────────────────────────────────────
        qkv = self.qkv_proj(h)  # [B, N, 3*C]
        # Split along last dim into three equal tensors
        q, k, v = qkv.chunk(3, dim=-1)  # each [B, N, C]

        # ── Reshape for multi-head attention ───────────────────────────────
        # Split C into (num_heads, head_dim)
        # 'b n (h d) -> b h n d' means: put heads before sequence length
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)  # [B, H, N, D]
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)  # [B, H, N, D]
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)  # [B, H, N, D]

        # ── Scaled dot-product attention ───────────────────────────────────
        # attn_weights = softmax(Q * K^T / sqrt(d)) * V
        #
        # Q: [B, heads, N, D]
        # K^T: [B, heads, D, N]
        # Q @ K^T: [B, heads, N, N] — each position attending to every other
        #
        # WHY use PyTorch's built-in scaled_dot_product_attention (SDPA):
        #   1. Flash attention: fuses the softmax+matmul into one kernel.
        #      Saves memory by never materializing the full [N,N] attention matrix.
        #   2. Significantly faster on modern GPUs (A100, H100).
        #   3. Falls back gracefully to standard attention on older hardware.
        #   Available in PyTorch 2.0+.
        #
        # At 8x8: N=64, attention matrix = 64x64 = tiny. Memory is not the concern.
        # We use SDPA anyway because it's a free speedup with no downside.
        attn_out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        # Output: [B, heads, N, D]

        # ── Merge heads back ───────────────────────────────────────────────
        # 'b h n d -> b n (h d)': put heads and head_dim back together
        attn_out = rearrange(attn_out, "b h n d -> b n (h d)")  # [B, N, C]

        # ── Output projection ──────────────────────────────────────────────
        attn_out = self.out_proj(attn_out)  # [B, N, C]

        # ── Reshape back to spatial ────────────────────────────────────────
        # 'b (h w) c -> b c h w': restore spatial structure
        attn_out = rearrange(attn_out, "b (h w) c -> b c h w", h=H, w=W)  # [B, C, H, W]

        # ── Residual connection ────────────────────────────────────────────
        return attn_out + residual


# =============================================================================
# STEP 7 — DOWNSAMPLE AND UPSAMPLE
# =============================================================================
#
# PURPOSE: change spatial resolution between U-Net levels.
#   Downsample: [B, C, H, W] → [B, C, H/2, W/2]
#   Upsample:   [B, C, H, W] → [B, C, H*2, W*2]
#
# THREE OPTIONS FOR DOWNSAMPLING — decision and reasoning:
#
#   Option A — MaxPool2d(2):
#     Simple. Retains the strongest activation per 2x2 region.
#     Problem: non-differentiable at ties. Throws away information.
#     Used in old CNNs (VGG, AlexNet). Not ideal for generative models.
#
#   Option B — AvgPool2d(2):
#     Smoother than MaxPool. Differentiable.
#     Problem: still throws away ~75% of computed features (averages them out).
#     The network has no control over what gets kept.
#
#   Option C — Strided Conv2d(C, C, 3, stride=2, padding=1) [OUR CHOICE]:
#     Learnable downsampling. The network learns what to keep and what to discard.
#     Preserves more information than pooling.
#     Standard in all modern diffusion U-Nets (DDPM, DDIM, ADM, DiT).
#     Cost: a few extra parameters per level. Worth it.
#
# THREE OPTIONS FOR UPSAMPLING:
#
#   Option A — ConvTranspose2d (transposed conv / "deconv"):
#     Learnable upsampling. BUT causes "checkerboard artifacts" — periodic
#     patterns in the output due to uneven overlap in the transposed convolution.
#     Famously problematic for image generation (Odena et al. 2016).
#
#   Option B — nn.Upsample(scale_factor=2, mode='bilinear') + Conv [OUR CHOICE]:
#     First upsample with bilinear interpolation (smooth, no artifacts).
#     Then apply a learned 3x3 conv to refine the upsampled features.
#     Avoids checkerboard artifacts completely.
#     Standard approach in all modern diffusion U-Nets.
#
#   Option C — PixelShuffle:
#     Rearranges channels into spatial dimensions. Elegant but less common
#     in diffusion models. Good for super-resolution. Overkill here.


class Downsample(nn.Module):
    """
    Halve spatial dimensions using strided convolution.
    Channels are unchanged.
    [B, C, H, W] → [B, C, H/2, W/2]
    """

    def __init__(self, channels: int):
        super().__init__()
        # stride=2: conv moves 2 pixels at a time → output is half the size
        # kernel_size=3, padding=1: standard setup
        # Output H = (H + 2*padding - kernel) / stride + 1
        #          = (H + 2 - 3) / 2 + 1 = (H-1)/2 + 1 = H/2 (for even H)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """
    Double spatial dimensions using bilinear upsampling + conv.
    Channels are unchanged.
    [B, C, H, W] → [B, C, H*2, W*2]
    """

    def __init__(self, channels: int):
        super().__init__()
        # 3x3 conv to refine after upsampling. padding=1 preserves spatial dims.
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: bilinear upsampling — smooth, no checkerboard artifacts
        # scale_factor=2: doubles H and W
        # mode='nearest' is faster and good enough for our resolution.
        # 'bilinear' with align_corners=False is the alternative — smoother
        # but slightly slower. For CIFAR-32 the difference is negligible.
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        # Step 2: learned conv to refine the upsampled features
        return self.conv(x)
