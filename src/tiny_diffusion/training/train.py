"""
src/tiny_diffusion/training/train.py

PHASE 3 — TRAINING LOOP WITH FULL OBSERVABILITY

This wires together every piece built across Phase 1 (architecture),
Phase 2 (MLflow tracking, DVC, Hydra config), and this phase's data
pipeline into the actual training loop.

WHAT "FULL OBSERVABILITY" MEANS CONCRETELY, PER PHASE 2 STEP 2's SCHEMA:
  - every `log_step_metrics_every` steps: loss, ema_loss, grad_norm, lr
  - every `log_system_metrics_every` steps: GPU util/memory, iter time
  - every epoch: FID on a cheap sample batch, a sample grid image
  - automatic instability detection via grad_norm spike monitoring
  - checkpointing: periodic (for resumability) + best-by-FID (for MLflow)

WHY THIS FUNCTION IS LONG: a training loop genuinely has many concerns
(data, forward pass, loss, backward, EMA update, logging, checkpointing,
instability detection) that all happen on the same critical path every
step. Splitting these into many tiny functions called once each would
add indirection without adding clarity — this is a case where a longer,
well-commented function is more readable than scattered abstraction.
"""

import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.optim import AdamW

from tiny_diffusion.data.cifar10 import denormalize, get_dataloader
from tiny_diffusion.diffusion.schedule import CosineNoiseSchedule
from tiny_diffusion.models.config import ModelConfig
from tiny_diffusion.models.ema import EMA
from tiny_diffusion.models.unet import UNet
from tiny_diffusion.utils import tracking
from tiny_diffusion.utils.seed import set_seed


def build_model_config(cfg: DictConfig) -> ModelConfig:
    """
    Convert the Hydra-resolved config (cfg.experiment.model.*) into a real
    Phase 1 ModelConfig dataclass. This is the bridge between Hydra's
    DictConfig (what train.py receives) and the dataclass UNet actually
    expects in its constructor — kept as one explicit function rather
    than passing the raw DictConfig into UNet, so UNet's constructor
    signature stays decoupled from Hydra entirely (UNet should be usable
    even by someone who's never heard of Hydra).
    """
    m = cfg.experiment.model
    return ModelConfig(
        image_size=m.image_size,
        in_channels=m.in_channels,
        base_channels=m.base_channels,
        channel_mult=list(m.channel_mult),
        num_res_blocks=m.num_res_blocks,
        attention_resolutions=list(m.attention_resolutions),
        num_heads=m.num_heads,
        time_embed_dim=m.time_embed_dim,
        num_classes=m.num_classes,
        cfg_dropout=m.cfg_dropout,
        T=cfg.experiment.schedule.T,
        schedule=cfg.experiment.schedule.type,
        num_groups=m.num_groups,
        out_channels=m.out_channels,
    )


def compute_grad_norm(model: torch.nn.Module) -> float:
    """
    L2 norm of gradients across ALL parameters, computed BEFORE clipping.

    WHY BEFORE CLIPPING SPECIFICALLY: this is the instability SIGNAL we
    want to log and monitor (Phase 2 Step 2's detect_instability function).
    If we logged the post-clip norm, every spike would be artificially
    capped at grad_clip_norm and invisible in the metric — we'd see a
    flat line even during genuine instability. Logging the PRE-clip norm
    is what makes the spike visible in MLflow's chart.
    """
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm_sq += p.grad.data.norm(2).item() ** 2
    return float(total_norm_sq**0.5)


@torch.no_grad()
def generate_sample_grid(
    model: torch.nn.Module,
    schedule: CosineNoiseSchedule,
    ema: EMA,
    num_classes: int,
    image_size: int,
    device: torch.device,
    normalize_mean: Tuple[float, float, float],
    normalize_std: Tuple[float, float, float],
    num_samples_per_class: int = 4,
    ddim_steps: int = 50,
) -> torch.Tensor:
    """
    Generate a grid of samples using EMA weights, for the per-epoch
    visual sanity check (Phase 2 Step 2's log_sample_grid).

    WHY EMA WEIGHTS, NOT TRAINING WEIGHTS: Phase 0 Section 8 and Phase 1's
    EMA class docstring both establish this — EMA weights are what you
    evaluate and generate from, never the raw training weights. The gap
    between EMA-FID and train-FID is real and well-documented; using
    train weights here would give a misleadingly pessimistic view of
    actual model quality during training.

    WHY DDIM AT 50 STEPS HERE (not the full DDPM 1000-step or DDIM-250
    "default eval" from Phase 0's sampler comparison table): this sample
    grid is generated EVERY EPOCH purely as a fast visual sanity check
    during training, not a quality benchmark — Phase 4's actual evaluation
    pipeline does the careful multi-step-count comparison. 50 steps here
    keeps epoch-end sampling fast enough not to meaningfully slow training.
    """
    model.eval()

    # Temporarily swap to EMA weights for sampling
    with ema.apply(model):
        num_classes_to_show = min(num_classes, 10)
        batch_size = num_classes_to_show * num_samples_per_class

        # Build class labels: [0,0,0,0, 1,1,1,1, ..., 9,9,9,9]
        labels = torch.arange(num_classes_to_show, device=device).repeat_interleave(
            num_samples_per_class
        )

        # Start from pure noise — this IS x_T in Phase 0's notation
        x = torch.randn(batch_size, 3, image_size, image_size, device=device)

        # Simple DDIM sampling loop (deterministic, eta=0) — see Phase 0
        # Section 6 for the full derivation this implements.
        timesteps = torch.linspace(schedule.T - 1, 0, ddim_steps, device=device).long()

        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_prev = timesteps[i + 1]
            t_batch = t.expand(batch_size)

            eps_pred = model(x, t_batch, labels)

            abar_t = schedule.alphas_cumprod[t]
            abar_t_prev = (
                schedule.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
            )

            x0_pred = (x - torch.sqrt(1 - abar_t) * eps_pred) / torch.sqrt(abar_t)
            x0_pred = x0_pred.clamp(-1, 1)  # see Phase 0's warning on why this matters

            direction = torch.sqrt(1 - abar_t_prev) * eps_pred
            x = torch.sqrt(abar_t_prev) * x0_pred + direction

    model.train()

    # Convert back to viewable [0,1] range for the saved image
    return denormalize(x, normalize_mean, normalize_std)


def train(cfg: DictConfig) -> None:
    """
    The main training entry point. Called from scripts/train.py once
    Hydra has resolved the full config.
    """
    # ── Setup ──────────────────────────────────────────────────────────────
    set_seed(cfg.experiment.seed, deterministic=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}")

    model_config = build_model_config(cfg)
    model = UNet(model_config).to(device)

    schedule = CosineNoiseSchedule(
        T=cfg.experiment.schedule.T,
        s=cfg.experiment.schedule.get("cosine_s", 0.008),
    ).to(device)
    schedule.verify_schedule()

    ema = EMA(model, decay=cfg.experiment.training.ema_decay)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.experiment.training.lr,
        betas=(cfg.experiment.training.adam_beta1, cfg.experiment.training.adam_beta2),
        weight_decay=cfg.experiment.training.weight_decay,
    )

    train_loader = get_dataloader(
        root="data/raw",
        train=True,
        batch_size=cfg.experiment.training.batch_size,
        normalize_mean=tuple(cfg.experiment.data.normalize_mean),
        normalize_std=tuple(cfg.experiment.data.normalize_std),
        random_horizontal_flip=cfg.experiment.data.random_horizontal_flip,
        random_crop_padding=cfg.experiment.data.random_crop_padding,
        num_workers=cfg.experiment.data.num_workers,
        pin_memory=cfg.experiment.data.pin_memory,
    )

    # Mixed precision setup — see Phase 0/ML Systems textbook connection:
    # fp16 autocast roughly halves memory bandwidth per step, the same
    # mechanism Phase 5's quantization study measures post-training, but
    # here applied DURING training for speed/memory headroom on T4/P100.
    use_amp = cfg.experiment.training.mixed_precision and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── MLflow run setup ───────────────────────────────────────────────────
    tracking.init_tracking(experiment_name="tiny-diffusion-cifar10")

    run_tags = dict(cfg.experiment.tags) if "tags" in cfg.experiment else {}

    with tracking.start_run(run_name=cfg.experiment.experiment_name, tags=run_tags):
        tracking.log_config(model_config.to_dict(), prefix="model.")
        tracking.log_config(
            {
                "lr": cfg.experiment.training.lr,
                "batch_size": cfg.experiment.training.batch_size,
                "num_epochs": cfg.experiment.training.num_epochs,
                "ema_decay": cfg.experiment.training.ema_decay,
                "mixed_precision": use_amp,
            },
            prefix="training.",
        )
        tracking.log_seed(cfg.experiment.seed)

        # ── Training state ───────────────────────────────────────────────
        global_step = 0
        # NOTE: "save checkpoint as best-by-FID" tracking (the
        # outputs/checkpoints/best.pt that dvc.yaml's evaluate stage
        # depends on) is intentionally NOT implemented yet — it needs a
        # real FID number to compare against, which only exists once
        # Phase 4's fid.py is built. Re-introduce a best_fid tracking
        # variable here when wiring that in, rather than carrying a
        # dead placeholder in the meantime (flake8 correctly flagged the
        # earlier unused `best_fid = float("inf")` line as dead code).
        grad_norm_history: list = []

        num_epochs = cfg.experiment.training.num_epochs
        log_step_every = cfg.experiment.training.log_step_metrics_every
        log_system_every = cfg.experiment.training.log_system_metrics_every
        grad_clip_norm = cfg.experiment.training.grad_clip_norm
        checkpoint_every = cfg.experiment.training.checkpoint_every_n_steps

        ema_loss_value: Optional[float] = None  # smoothed metric, see tracking.py's
        # docstring distinguishing this from
        # the model-weights EMA

        print(
            f"[train] starting training: {num_epochs} epochs, " f"{len(train_loader)} steps/epoch"
        )

        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            for batch_idx, (images, labels) in enumerate(train_loader):
                step_start_time = time.time()

                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                B = images.shape[0]

                # ── Forward diffusion (Phase 0 Section 1) ───────────────────
                # Sample a random timestep PER EXAMPLE in the batch — this
                # is exactly the t ~ U[1,T] sampling from Phase 0's L_simple
                # loss derivation.
                t = torch.randint(0, schedule.T, (B,), device=device)
                noise = torch.randn_like(images)
                x_t = schedule.q_sample(images, t, noise)

                # ── Forward pass + loss (Phase 0 Section 3) ─────────────────
                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    noise_pred = model(x_t, t, labels)
                    loss = F.mse_loss(noise_pred, noise)
                    # ^ this single line IS Phase 0's entire L_simple —
                    # everything else in this function is infrastructure
                    # around this one mathematical statement.

                # ── Backward pass ────────────────────────────────────────────
                scaler.scale(loss).backward()

                # Unscale before computing grad norm — otherwise the norm
                # is inflated by the AMP loss-scaling factor and the
                # instability-detection thresholds would be meaningless.
                scaler.unscale_(optimizer)
                grad_norm = compute_grad_norm(model)

                # Gradient clipping — caps the norm at grad_clip_norm to
                # prevent a single bad batch from destabilizing training.
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

                scaler.step(optimizer)
                scaler.update()

                # ── EMA update (Phase 1's EMA class, warmup-aware) ──────────
                ema.update(model, global_step)

                # ── Instability detection (Phase 2 Step 2) ──────────────────
                grad_norm_history.append(grad_norm)
                if len(grad_norm_history) > 50:
                    grad_norm_history.pop(0)  # keep a rolling window of last 50

                is_unstable = tracking.detect_instability(grad_norm, grad_norm_history)
                if is_unstable:
                    print(
                        f"[WARNING] step {global_step}: grad_norm spike "
                        f"detected ({grad_norm:.2f}) — possible instability"
                    )

                # ── Step-level metrics logging ───────────────────────────────
                loss_value = loss.item()
                ema_loss_value = (
                    loss_value
                    if ema_loss_value is None
                    else 0.98 * ema_loss_value + 0.02 * loss_value
                )

                if global_step % log_step_every == 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    tracking.log_step_metrics(
                        step=global_step,
                        loss=loss_value,
                        ema_loss=ema_loss_value,
                        grad_norm=grad_norm,
                        learning_rate=current_lr,
                    )

                # ── System metrics logging ───────────────────────────────────
                if global_step % log_system_every == 0:
                    iter_time = time.time() - step_start_time
                    tracking.log_system_metrics(step=global_step, iteration_time_sec=iter_time)

                # ── Periodic checkpointing (resumability, via DVC add) ──────
                if global_step > 0 and global_step % checkpoint_every == 0:
                    ckpt_path = Path(f"outputs/checkpoints/step_{global_step:07d}.pt")
                    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "ema_state_dict": ema.state_dict(),
                            "global_step": global_step,
                            "epoch": epoch,
                        },
                        ckpt_path,
                    )
                    print(f"[train] saved checkpoint: {ckpt_path}")
                    # NOTE: actually `dvc add`-ing this checkpoint is a
                    # separate concern from training itself — see Phase 3's
                    # checkpoint.py module for the DVC integration, kept
                    # OUT of the hot training loop so a slow DVC operation
                    # never blocks the next training step.

                global_step += 1

            # ── End of epoch: expensive evaluation (Phase 2 Step 2 schema) ──
            epoch_time = time.time() - epoch_start_time
            print(
                f"[train] epoch {epoch} complete in {epoch_time:.1f}s "
                f"(loss={loss_value:.4f}, ema_loss={ema_loss_value:.4f})"
            )

            # NOTE: real FID computation (Phase 4's fid.py) is intentionally
            # NOT wired in yet — this phase focuses on the training loop's
            # observability infrastructure. epoch_metrics logging is called
            # here with a placeholder so the MLflow schema is exercised
            # end-to-end; Phase 4 replaces this with the real FID pipeline.
            tracking.log_epoch_metrics(epoch=epoch, extra_metrics={"epoch_time_sec": epoch_time})

        print(f"[train] training complete. Total steps: {global_step}")
