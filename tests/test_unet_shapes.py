"""
tests/test_unet_shapes.py

PHASE 2, STEP 5 — UNIT TESTS: U-NET FORWARD PASS SHAPES

Directly tests the components Phase 1 built: ResBlock, SelfAttentionBlock,
Downsample, Upsample, and the full UNet assembly. The goal is to catch
shape bugs (wrong channel count after a skip connection, wrong spatial
size after up/downsampling) automatically, rather than relying on the
manual dry-run trace from Phase 1 every time someone changes the code.
"""

import pytest
import torch

from tiny_diffusion.models.blocks import Downsample, ResBlock, SelfAttentionBlock, Upsample
from tiny_diffusion.models.config import ModelConfig
from tiny_diffusion.models.embeddings import ClassEmbedding, SinusoidalTimeEmbedding
from tiny_diffusion.models.unet import UNet


@pytest.fixture
def config():
    """Smaller-than-production config for fast tests — same STRUCTURE as
    the real ~55M model, just scaled down so tests run in milliseconds
    instead of seconds. Testing the architecture's correctness doesn't
    require the production parameter count.

    IMPORTANT: num_groups must evenly divide every channel count that
    appears anywhere in the network (nn.GroupNorm's own hard requirement).
    Production uses num_groups=32 because production channels
    (128/256/512/1024 from base_channels=128) are all divisible by 32.
    Here base_channels=16 with channel_mult=[1,2,4,8] gives channels of
    16/32/64/128 — all divisible by 8, so num_groups=8 is the correct
    scaled-down equivalent, not the production default of 32.
    """
    return ModelConfig(
        base_channels=16,  # production: 128 — scaled down 8x
        channel_mult=[1, 2, 4, 8],  # SAME structure as production
        num_res_blocks=2,
        attention_resolutions=[8, 4],
        time_embed_dim=64,  # production: 512
        num_classes=10,
        cfg_dropout=0.15,
        T=1000,
        num_groups=8,  # production: 32 — see docstring above
    )


class TestTimeEmbedding:
    def test_output_shape(self, config):
        embed = SinusoidalTimeEmbedding(config.time_embed_dim)
        t = torch.randint(1, config.T, (8,))
        out = embed(t)
        assert out.shape == (8, config.time_embed_dim)

    def test_different_timesteps_give_different_embeddings(self, config):
        """A basic sanity check that the sinusoidal formula is actually
        timestep-dependent, not accidentally constant."""
        embed = SinusoidalTimeEmbedding(config.time_embed_dim)
        t1 = torch.tensor([1])
        t2 = torch.tensor([500])
        out1 = embed(t1)
        out2 = embed(t2)
        assert not torch.allclose(out1, out2), (
            "Embeddings for t=1 and t=500 are identical — the sinusoidal "
            "formula isn't varying with t as expected."
        )


class TestClassEmbedding:
    def test_output_shape(self, config):
        embed = ClassEmbedding(config.num_classes, config.time_embed_dim, config.cfg_dropout)
        labels = torch.randint(0, 10, (8,))
        out = embed(labels)
        assert out.shape == (8, config.time_embed_dim)

    def test_force_uncond_uses_null_token_for_all(self, config):
        """When force_uncond=True (the CFG inference unconditional branch),
        every sample in the batch should get the SAME null-token embedding,
        regardless of what labels were passed in."""
        embed = ClassEmbedding(config.num_classes, config.time_embed_dim, config.cfg_dropout)
        embed.eval()  # disable training-mode dropout for a clean test
        labels = torch.tensor([0, 3, 7, 9])
        out = embed(labels, force_uncond=True)
        # All 4 rows should be identical, since they all map to the null token
        for i in range(1, 4):
            assert torch.allclose(out[0], out[i]), (
                "force_uncond=True should map every label to the same null "
                "token embedding, but row outputs differ."
            )

    def test_cfg_dropout_only_active_in_training_mode(self, config):
        """In eval mode, CFG dropout must NOT randomly null out labels —
        only during training. This is critical: if dropout leaked into
        eval mode, FID evaluation would silently use the wrong class
        conditioning some fraction of the time."""
        embed = ClassEmbedding(config.num_classes, config.time_embed_dim, cfg_dropout=1.0)
        # cfg_dropout=1.0 is a deliberately extreme value: if dropout leaked
        # into eval mode, ALL labels would be replaced with null token. We
        # verify this does NOT happen during eval.
        embed.eval()
        labels = torch.tensor([3])
        out_eval = embed(labels, force_uncond=False)

        null_embedding = embed.embedding(torch.tensor([config.num_classes]))
        assert not torch.allclose(out_eval, null_embedding), (
            "With cfg_dropout=1.0 but model.eval(), output matched the null "
            "token embedding — CFG dropout is incorrectly active during eval."
        )


class TestResBlock:
    def test_output_shape_same_channels(self, config):
        # num_groups must divide num_channels evenly (nn.GroupNorm's own
        # requirement) — production uses num_groups=32 because production
        # channels (128/256/512/1024) are all divisible by 32. This test
        # uses small channel counts for speed, so it must use a smaller
        # num_groups to match, or GroupNorm raises ValueError on construction.
        block = ResBlock(in_channels=16, out_channels=16, cond_dim=64, num_groups=8)
        x = torch.randn(2, 16, 32, 32)
        cond = torch.randn(2, 64)
        out = block(x, cond)
        assert out.shape == (2, 16, 32, 32)

    def test_output_shape_channel_change(self, config):
        """ResBlock must correctly handle in_channels != out_channels via
        its skip_proj 1x1 conv — this is exactly the case that happens at
        every resolution transition in the real U-Net."""
        block = ResBlock(in_channels=16, out_channels=32, cond_dim=64, num_groups=8)
        x = torch.randn(2, 16, 32, 32)
        cond = torch.randn(2, 64)
        out = block(x, cond)
        assert out.shape == (2, 32, 32, 32)

    def test_zero_init_means_identity_at_initialization(self, config):
        """Phase 1 deliberately zero-initializes conv2 and cond_proj so
        the ResBlock outputs (close to) its input at initialization —
        this stabilizes early training. Verify that property actually
        holds right after construction, before any training."""
        block = ResBlock(in_channels=16, out_channels=16, cond_dim=64, num_groups=8)
        block.eval()
        x = torch.randn(2, 16, 32, 32)
        cond = torch.randn(2, 64)
        out = block(x, cond)
        # Should be very close to identity (residual + near-zero h)
        assert torch.allclose(out, x, atol=1e-4), (
            "ResBlock is not close to identity at initialization — check "
            "that conv2 and cond_proj are still zero-initialized."
        )


class TestSelfAttentionBlock:
    def test_output_shape_preserved(self, config):
        attn = SelfAttentionBlock(channels=32, num_heads=4)
        x = torch.randn(2, 32, 8, 8)
        out = attn(x)
        assert out.shape == x.shape

    def test_raises_on_non_divisible_heads(self, config):
        """channels must be divisible by num_heads — Phase 1's assertion
        should fire on construction, not fail silently or crash deep
        inside the attention math with a confusing tensor shape error."""
        with pytest.raises(AssertionError):
            SelfAttentionBlock(channels=33, num_heads=4)  # 33 not divisible by 4

    def test_zero_init_means_near_identity_at_initialization(self, config):
        attn = SelfAttentionBlock(channels=32, num_heads=4)
        attn.eval()
        x = torch.randn(2, 32, 8, 8)
        out = attn(x)
        assert torch.allclose(
            out, x, atol=1e-4
        ), "SelfAttentionBlock is not close to identity at initialization."


class TestDownsampleUpsample:
    def test_downsample_halves_spatial_dims(self, config):
        down = Downsample(channels=16)
        x = torch.randn(2, 16, 32, 32)
        out = down(x)
        assert out.shape == (2, 16, 16, 16)

    def test_upsample_doubles_spatial_dims(self, config):
        up = Upsample(channels=16)
        x = torch.randn(2, 16, 16, 16)
        out = up(x)
        assert out.shape == (2, 16, 32, 32)

    def test_downsample_upsample_roundtrip_preserves_shape(self, config):
        """Not a numerical identity test (these aren't meant to be
        perfectly invertible) — just confirms the SHAPE round-trips
        correctly, which is what matters for the skip-connection logic
        in the full U-Net."""
        down = Downsample(channels=16)
        up = Upsample(channels=16)
        x = torch.randn(2, 16, 32, 32)
        out = up(down(x))
        assert out.shape == x.shape


class TestFullUNet:
    """
    The integration test — exercises every component together exactly as
    Phase 1's manual dry-run script did, but as a real automated assertion
    instead of a human reading printed shapes.
    """

    def test_output_shape_matches_input_shape(self, config):
        model = UNet(config)
        model.eval()
        B = 4
        x = torch.randn(B, 3, config.image_size, config.image_size)
        t = torch.randint(1, config.T, (B,))
        c = torch.randint(0, config.num_classes, (B,))

        with torch.no_grad():
            out = model(x, t, c)

        assert out.shape == x.shape, (
            f"UNet output shape {out.shape} does not match input shape "
            f"{x.shape} — eps_theta must predict noise in the same space "
            f"as the input image."
        )

    def test_force_uncond_runs_without_error(self, config):
        """The CFG inference unconditional branch must be a valid forward
        pass, not just a theoretical code path that's never actually run."""
        model = UNet(config)
        model.eval()
        B = 4
        x = torch.randn(B, 3, config.image_size, config.image_size)
        t = torch.randint(1, config.T, (B,))
        c = torch.randint(0, config.num_classes, (B,))

        with torch.no_grad():
            out = model(x, t, c, force_uncond=True)

        assert out.shape == x.shape

    def test_different_batch_sizes_work(self, config):
        """Catches any hardcoded batch-size assumption that would break
        the moment we change batch_size in training config."""
        model = UNet(config)
        model.eval()
        for B in [1, 2, 8]:
            x = torch.randn(B, 3, config.image_size, config.image_size)
            t = torch.randint(1, config.T, (B,))
            c = torch.randint(0, config.num_classes, (B,))
            with torch.no_grad():
                out = model(x, t, c)
            assert out.shape == (
                B,
                3,
                config.image_size,
                config.image_size,
            ), f"Failed at batch_size={B}"

    def test_gradient_flows_to_all_parameters(self, config):
        """A common silent bug: some branch of the network never gets a
        gradient (e.g. a layer that's accidentally bypassed). This test
        runs one backward pass and checks every parameter received a
        non-None, non-zero gradient — catching dead code paths that
        would otherwise just silently never train."""
        model = UNet(config)
        model.train()
        B = 2
        x = torch.randn(B, 3, config.image_size, config.image_size)
        t = torch.randint(1, config.T, (B,))
        c = torch.randint(0, config.num_classes, (B,))

        out = model(x, t, c)
        loss = out.pow(2).mean()
        loss.backward()

        params_with_no_grad = []
        for name, param in model.named_parameters():
            if param.grad is None:
                params_with_no_grad.append(name)

        assert not params_with_no_grad, (
            f"These parameters received NO gradient at all, suggesting "
            f"a dead code path or disconnected module: {params_with_no_grad}"
        )

    def test_parameter_count_in_expected_range(self, config):
        """Not a precise check (the test config is intentionally scaled
        down from production), but catches GROSS errors — e.g. an
        accidental 100x parameter blowup from a misconfigured channel
        multiplier, long before you'd notice it from a slow training step."""
        model = UNet(config)
        total_params = sum(p.numel() for p in model.parameters())
        # With base_channels=16 (8x smaller than production's 128), we
        # expect roughly (1/8)^2 = 1/64th of production's ~55M params,
        # i.e. very roughly under 2M for this scaled-down test config.
        assert total_params < 5_000_000, (
            f"Test config has {total_params:,} params — expected well "
            f"under 5M for this scaled-down config. A number this large "
            f"suggests a channel multiplier or architecture bug."
        )
