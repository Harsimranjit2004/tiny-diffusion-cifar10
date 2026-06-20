"""
scripts/smoke_test_training.py

PHASE 3 — CPU-ONLY SMOKE TEST

Runs 3 real training steps end-to-end on CPU with a tiny model and tiny
batch — NOT a real training run (CPU would take forever for the real
~55M model), just a fast mechanical check that:
  - forward diffusion produces correctly-shaped noisy images
  - the model forward pass runs without shape errors
  - loss computes and backward() succeeds
  - EMA update runs without error
  - gradient norm computation works
  - instability detection doesn't crash on real data

WHY THIS EXISTS SEPARATELY FROM pytest's test_unet_shapes.py: those tests
check INDIVIDUAL components in isolation. This script checks the actual
INTEGRATION — the exact sequence of operations train.py's real loop
performs, end to end, on a tiny scale. Catching an integration bug here
(CPU, seconds) is much cheaper than discovering it after spinning up a
Kaggle GPU session and waiting for the import + data download.

RUN THIS BEFORE EVERY REAL GPU TRAINING SESSION as a fast pre-flight check.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from tiny_diffusion.diffusion.schedule import CosineNoiseSchedule
from tiny_diffusion.models.config import ModelConfig
from tiny_diffusion.models.ema import EMA
from tiny_diffusion.models.unet import UNet
from tiny_diffusion.utils.seed import set_seed
from tiny_diffusion.utils.tracking import detect_instability


def main() -> None:
    print("=" * 70)
    print("PHASE 3 — CPU SMOKE TEST (3 training steps, tiny model)")
    print("=" * 70)

    set_seed(42, deterministic=False)  # deterministic=False: speed over
    # exactness for a quick smoke test — we're checking "does it crash",
    # not "is it bit-reproducible" here.

    device = torch.device("cpu")
    print(f"\n[smoke test] device: {device}")

    # Tiny config — same structure as production, scaled down for CPU speed
    config = ModelConfig(
        base_channels=16,
        channel_mult=[1, 2, 4, 8],
        num_res_blocks=2,
        attention_resolutions=[8, 4],
        time_embed_dim=64,
        num_classes=10,
        cfg_dropout=0.15,
        T=1000,
        num_groups=8,  # must divide all channel counts — see Phase 2 Step 5's
        # test fixture bug and fix for why this matters
    )

    model = UNet(config).to(device)
    model.train()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[smoke test] model built: {total_params:,} params (tiny test config)")

    schedule = CosineNoiseSchedule(T=config.T).to(device)
    ema = EMA(model, decay=0.999)
    optimizer = AdamW(model.parameters(), lr=2e-4)

    # Fake batch — no real CIFAR-10 needed for this mechanical check
    B = 4
    images = torch.randn(B, 3, 32, 32, device=device)
    labels = torch.randint(0, 10, (B,), device=device)

    grad_norm_history = []

    print("\n[smoke test] running 3 training steps...")
    for step in range(3):
        t = torch.randint(0, schedule.T, (B,), device=device)
        noise = torch.randn_like(images)
        x_t = schedule.q_sample(images, t, noise)

        assert x_t.shape == images.shape, "q_sample changed tensor shape!"

        optimizer.zero_grad(set_to_none=True)
        noise_pred = model(x_t, t, labels)

        assert noise_pred.shape == images.shape, "model output shape mismatch!"

        loss = F.mse_loss(noise_pred, noise)
        loss.backward()

        # Gradient norm BEFORE optimizer step (matches train.py's real
        # ordering — this is the instability-detection signal)
        total_norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm_sq += p.grad.data.norm(2).item() ** 2
        grad_norm = total_norm_sq**0.5

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        ema.update(model, step)

        grad_norm_history.append(grad_norm)
        is_unstable = detect_instability(grad_norm, grad_norm_history)

        print(
            f"  step {step}: loss={loss.item():.4f}  "
            f"grad_norm={grad_norm:.4f}  unstable={is_unstable}"
        )

    # Verify EMA context manager works (used by sample generation in train.py)
    print("\n[smoke test] testing EMA context manager...")
    with ema.apply(model):
        with torch.no_grad():
            test_out = model(images, torch.zeros(B, dtype=torch.long), labels)
    print(f"  EMA-mode forward pass output shape: {test_out.shape}")

    print("\n[smoke test] testing force_uncond (CFG inference path)...")
    with torch.no_grad():
        uncond_out = model(images, torch.zeros(B, dtype=torch.long), labels, force_uncond=True)
    print(f"  force_uncond forward pass output shape: {uncond_out.shape}")

    print("\n" + "=" * 70)
    print("SMOKE TEST PASSED — training loop mechanics work end to end.")
    print("Safe to proceed to a real GPU session (Kaggle/Colab) for actual training.")
    print("=" * 70)


if __name__ == "__main__":
    main()
