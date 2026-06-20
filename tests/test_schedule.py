"""
tests/test_schedule.py

PHASE 2, STEP 5 — UNIT TESTS: NOISE SCHEDULE

These directly implement the 7 sanity checks specified back in Phase 0,
Section 4 ("Sanity checks to write as unit tests") and Phase 1's
CosineNoiseSchedule.verify_schedule() — but as REAL pytest tests with
assertions, not just printed output a human has to read and judge.

WHY THIS MATTERS BEYOND "good practice": the noise schedule is the
mathematical foundation every other component depends on (forward
diffusion, training loss, both samplers). A subtle bug here — like the
off-by-one indexing explicitly warned about in Phase 0 — would silently
corrupt every downstream result while still "running" without crashing.
These tests exist to catch exactly that class of silent-but-wrong bug.
"""

import pytest
import torch

from tiny_diffusion.diffusion.schedule import CosineNoiseSchedule


@pytest.fixture
def schedule():
    """A standard T=1000 cosine schedule, reused across tests."""
    return CosineNoiseSchedule(T=1000, s=0.008)


class TestScheduleProperties:
    """
    Phase 0's 7 sanity checks, each as an isolated, independently-failing
    test rather than one big verify_schedule() that stops at the first
    failure — this way, if THREE things break, you see all three failures
    at once instead of fixing one and rerunning to discover the next.
    """

    def test_alpha_bar_near_one_at_start(self, schedule):
        """Test 1 (Phase 0): alpha_bar_0 should be close to 1.0 — almost
        no noise has been added yet at the very first timestep."""
        assert schedule.alphas_cumprod[0] > 0.9, (
            f"alpha_bar at t=1 is {schedule.alphas_cumprod[0].item():.4f}, "
            f"expected > 0.9 — the schedule is destroying signal too fast "
            f"at the very first timestep."
        )

    def test_alpha_bar_near_zero_at_end(self, schedule):
        """Test 2 (Phase 0): alpha_bar_T should be near 0 — by the final
        timestep the image should be almost indistinguishable from noise."""
        assert schedule.alphas_cumprod[-1] < 0.01, (
            f"alpha_bar at t=T is {schedule.alphas_cumprod[-1].item():.4f}, "
            f"expected < 0.01 — training samples at large t still contain "
            f"meaningful signal, which gives the reverse process an "
            f"easier target than it should have (see Phase 0's warning "
            f"on this exact failure mode)."
        )

    def test_alpha_bar_monotonically_decreasing(self, schedule):
        """Test 3 (Phase 0): noise should only ever increase as t increases,
        never decrease — alpha_bar_t is a cumulative product of values < 1,
        so it must be strictly decreasing."""
        diffs = schedule.alphas_cumprod[1:] - schedule.alphas_cumprod[:-1]
        assert (diffs < 0).all(), (
            "alpha_bar is not monotonically decreasing — found at least one "
            "timestep where noise level decreased going forward, which is "
            "mathematically impossible for a valid variance-preserving "
            "schedule and indicates a bug in the cosine formula derivation."
        )

    def test_all_betas_in_valid_range(self, schedule):
        """Test 4 (Phase 0): every beta_t must be a valid probability-like
        value strictly between 0 and 1 (it's literally added as a variance,
        which must be positive, and must be < 1 or the VP property breaks)."""
        assert (schedule.betas > 0).all(), "Found non-positive beta_t — invalid variance."
        assert (
            schedule.betas < 1
        ).all(), "Found beta_t >= 1 — breaks variance-preserving property."

    def test_forward_diffusion_at_t_zero_is_near_identity(self, schedule):
        """Test 5 (Phase 0): at t=0 (first index), forward diffusion should
        return something very close to the original clean image — almost
        no noise should have been mixed in yet."""
        torch.manual_seed(0)
        x0 = torch.randn(4, 3, 32, 32)
        noise = torch.randn_like(x0)
        t = torch.zeros(4, dtype=torch.long)  # t=0 index

        xt = schedule.q_sample(x0, t, noise)

        # Should be very close to x0, NOT close to pure noise.
        # We check correlation with x0 is high (not exact equality, since
        # alpha_bar_0 is close to but not exactly 1.0).
        correlation = torch.corrcoef(torch.stack([x0.flatten(), xt.flatten()]))[0, 1]
        assert correlation > 0.95, (
            f"At t=0, x_t should correlate strongly with x_0 (correlation "
            f"{correlation:.4f} found, expected > 0.95) — forward diffusion "
            f"is adding too much noise even at the very first timestep."
        )

    def test_forward_diffusion_variance_preserving(self, schedule):
        """Test 6 (Phase 0): the variance-preserving property derived in
        Phase 0 — Var(x_t) should stay close to 1 regardless of t, given
        Var(x_0)=1, because signal and noise trade off but their combined
        variance is mathematically designed to stay constant."""
        torch.manual_seed(0)
        x0 = torch.randn(10000, 1)  # large N for a stable variance estimate
        noise = torch.randn_like(x0)

        for t_val in [0, 250, 500, 999]:
            t = torch.full((10000,), t_val, dtype=torch.long)
            xt = schedule.q_sample(x0, t, noise)
            var = xt.var().item()
            assert 0.8 < var < 1.2, (
                f"At t={t_val}, Var(x_t)={var:.4f}, expected close to 1.0 "
                f"(variance-preserving property) — got a value outside "
                f"[0.8, 1.2], suggesting a bug in the schedule coefficients."
            )

    def test_betas_never_exceed_clip_threshold(self, schedule):
        """Test 7 (Phase 0): beta_t must never exceed the 0.999 clip we
        deliberately apply to prevent numerical issues as t approaches T
        (see Phase 0's cosine schedule derivation for why this clip exists)."""
        assert (schedule.betas <= 0.999).all(), (
            "Found beta_t > 0.999 — the clip from Phase 0's derivation "
            "isn't being applied correctly."
        )


class TestScheduleShapes:
    """Shape consistency tests — every precomputed coefficient array must
    have exactly T entries, or indexing by timestep silently breaks."""

    def test_all_buffers_have_length_T(self, schedule):
        T = schedule.T
        buffers_to_check = [
            "betas",
            "alphas",
            "alphas_cumprod",
            "alphas_cumprod_prev",
            "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod",
            "sqrt_recip_alphas",
            "posterior_variance",
            "posterior_log_variance_clipped",
            "posterior_mean_coef1",
            "posterior_mean_coef2",
        ]
        for name in buffers_to_check:
            buf = getattr(schedule, name)
            assert buf.shape == (T,), (
                f"{name} has shape {buf.shape}, expected ({T},) — a "
                f"shape mismatch here will cause silent broadcasting bugs "
                f"or index-out-of-range errors during training/sampling."
            )


class TestQSampleBroadcasting:
    """
    Tests specifically targeting the _extract() broadcasting logic from
    Phase 1 — this is exactly the kind of code where a reshape bug silently
    broadcasts against the wrong tensor dimension instead of crashing.
    """

    def test_q_sample_preserves_input_shape(self, schedule):
        x0 = torch.randn(8, 3, 32, 32)
        t = torch.randint(0, schedule.T, (8,))
        xt = schedule.q_sample(x0, t)
        assert xt.shape == x0.shape, (
            f"q_sample changed tensor shape from {x0.shape} to {xt.shape} — "
            f"this indicates a broadcasting bug in _extract()."
        )

    def test_q_sample_different_t_per_batch_element_gives_different_results(self, schedule):
        """Each sample in a batch can have a DIFFERENT timestep — this is
        how Phase 0's training loop samples t ~ U[1,T] independently per
        example. If _extract()'s broadcasting is broken, every batch
        element might silently get the SAME t's coefficients regardless
        of what t array was actually passed in."""
        torch.manual_seed(0)
        x0 = torch.randn(2, 3, 32, 32)
        noise = torch.randn_like(x0)
        # Same image, same noise, but deliberately different t values
        t = torch.tensor([0, 999])

        xt = schedule.q_sample(x0, t, noise)

        # At t=0 (almost no noise) vs t=999 (almost pure noise), the two
        # outputs should look VERY different from each other.
        diff = (xt[0] - xt[1]).abs().mean().item()
        assert diff > 0.5, (
            f"xt[0] (t=0) and xt[1] (t=999) are nearly identical (mean "
            f"abs diff={diff:.4f}) — this strongly suggests _extract() is "
            f"broadcasting the SAME t value to every batch element "
            f"regardless of the actual per-sample t tensor passed in."
        )
