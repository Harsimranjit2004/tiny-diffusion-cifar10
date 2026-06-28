"""
src/tiny_diffusion/training/train.py

PHASE 3 — TRAINING LOOP WITH FULL OBSERVABILITY

This wires together every piece built across Phase 1 (architecture),
Phase 2 (MLflow tracking, DVC, Hydra config), and this phase's data
pipeline into the actual training loop.

WHY THIS VERSION FOLLOWS THE STANDARD DDPM TRAINING RECIPE, NOT A
CUSTOM ONE: across six real training runs we accumulated a cosine LR
scheduler, a "loss > 5x best_ema_loss" explosion heuristic, a
30-consecutive-bad-step auto-recovery mechanism, and early stopping —
each layered on reactively after a crash. That custom logic introduced
its own bugs (a CUDA OOM, then an UnboundLocalError) faster than it
fixed real problems. This version instead follows the loop structure
used by HuggingFace diffusers' train_unconditional.py and lucidrains'
denoising-diffusion-pytorch — both converge on the same small set of
ingredients for stable diffusion training:
  1. LR WARMUP (linear ramp over the first ~500 steps) — diffusion
     models are sensitive to large updates before the optimizer's Adam
     moment estimates have stabilised; warmup is the standard fix for
     early-training instability, not a runtime detector after the fact.
  2. Gradient clipping at a STANDARD value (1.0) — every reference
     implementation uses this; not a value we need to keep retuning.
  3. A single NaN/Inf guard on the loss — skip that one batch's
     optimizer step, nothing more elaborate. This is the ONE piece of
     defensive code every production training script actually has,
     because a single corrupted batch is a real, mundane occurrence
     that costs nothing to skip — but there is no "is this exploded
     relative to history" heuristic anywhere in the reference
     implementations, because it doesn't reliably distinguish real
     instability from a step in a noisy data point.
  4. fp32, no AMP — T4 + this model size showed real AMP overflow
     issues across two separate runs; fp32 has shown zero numerical
     problems and fits in memory at batch_size=64.
  5. EMA with its own internal warmup (Phase 1's EMA class) — unchanged.
If THIS standard recipe is still unstable, that is a real signal about
the architecture or data, worth knowing — not something to paper over
with more reactive runtime logic.

CHECKPOINT BACKEND — DVC (Colab/local) vs Vertex AI (GCS FUSE):
  Vertex AI has a BUILT-IN checkpoint mechanism: torch.save() to the
  /gcs/ FUSE-mounted path writes directly into Cloud Storage — durable
  the instant the call returns, no DVC, no OAuth needed.

  On Colab/local, DVC add/push/pull functions are still used for
  cross-session persistence via Google Drive.
"""

import math
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from tiny_diffusion.data.cifar10 import denormalize, get_dataloader
from tiny_diffusion.diffusion.schedule import CosineNoiseSchedule
from tiny_diffusion.models.config import ModelConfig
from tiny_diffusion.models.ema import EMA
from tiny_diffusion.models.unet import UNet
from tiny_diffusion.training.checkpoint import (
    cleanup_old_checkpoints,
    dvc_add_checkpoint,
    dvc_pull_latest_checkpoint,
    dvc_push_checkpoint,
)
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


def get_lr_warmup_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int) -> LambdaLR:
    """
    Linear LR warmup from 0 to the optimizer's base LR over `warmup_steps`,
    then flat at the base LR forever after.

    WHY THIS, NOT A FANCIER SCHEDULE: this is the exact warmup shape used
    in HuggingFace diffusers' get_constant_schedule_with_warmup and in
    most DDPM reference training scripts. Diffusion models are sensitive
    to large early updates before Adam's moment estimates have stabilised
    — warmup directly addresses that instead of reacting to it after a
    spike has already happened.
    """

    def lr_lambda(current_step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, current_step / warmup_steps)

    return LambdaLR(optimizer, lr_lambda)


def compute_grad_norm(model: torch.nn.Module) -> float:
    """
    L2 norm of gradients across ALL parameters, computed BEFORE clipping.

    WHY BEFORE CLIPPING SPECIFICALLY: this is the instability SIGNAL we
    want to log and monitor. If we logged the post-clip norm, every spike
    would be artificially capped at grad_clip_norm and invisible in the
    metric — we'd see a flat line even during genuine instability.
    """
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm_sq += p.grad.data.norm(2).item() ** 2
    result = float(total_norm_sq**0.5)
    # A non-finite norm is only possible if the loss itself was already
    # non-finite (caught separately, below) — guard anyway so a stray Inf
    # never poisons grad_norm_history's stdev computation downstream.
    return result if math.isfinite(result) else 0.0


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
    visual sanity check.

    WHY EMA WEIGHTS, NOT TRAINING WEIGHTS: EMA weights are what you
    evaluate and generate from, never the raw training weights — the gap
    between EMA-FID and train-FID is real and well-documented.

    WHY DDIM AT 50 STEPS HERE: this sample grid is a fast visual sanity
    check during training, not a quality benchmark — Phase 4's actual
    evaluation pipeline does the careful multi-step-count comparison.
    """
    model.eval()

    with ema.apply(model):
        num_classes_to_show = min(num_classes, 10)
        batch_size = num_classes_to_show * num_samples_per_class

        labels = torch.arange(num_classes_to_show, device=device).repeat_interleave(
            num_samples_per_class
        )

        x = torch.randn(batch_size, 3, image_size, image_size, device=device)

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
            x0_pred = x0_pred.clamp(-1, 1)

            direction = torch.sqrt(1 - abar_t_prev) * eps_pred
            x = torch.sqrt(abar_t_prev) * x0_pred + direction

    model.train()

    return denormalize(x, normalize_mean, normalize_std)


def is_running_on_vertex_ai() -> bool:
    """
    Detect Vertex AI Custom Training Job environment.

    WHY CLOUD_ML_PROJECT_ID SPECIFICALLY: Vertex AI sets this
    unconditionally in every custom training job container. Never set on
    Kaggle, Colab, or a local machine.
    """
    return "CLOUD_ML_PROJECT_ID" in os.environ


def get_checkpoint_dir() -> Path:
    """
    Return the correct checkpoint directory for whichever environment
    we're actually running in.

    Vertex AI: AIP_CHECKPOINT_DIR is a gs:// URI, converted to its FUSE-
    mounted local equivalent under /gcs/ — torch.save() there writes
    directly into durable Cloud Storage, no extra sync step.

    Colab/local: outputs/checkpoints — uses DVC add/push/pull for
    cross-session persistence via Google Drive.
    """
    if is_running_on_vertex_ai():
        aip_checkpoint_dir = os.environ.get("AIP_CHECKPOINT_DIR", "")
        if aip_checkpoint_dir.startswith("gs://"):
            gcs_path = aip_checkpoint_dir[len("gs://") :]
            return Path("/gcs") / gcs_path
        print(
            "[train] WARNING — running on Vertex AI but AIP_CHECKPOINT_DIR "
            "is unset or not a gs:// URI. Falling back to local-only "
            "checkpoint storage (no cross-session persistence)."
        )
        return Path("outputs/checkpoints")

    return Path("outputs/checkpoints")


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """
    Find the most recent checkpoint by step number, for automatic resume.
    """
    if not checkpoint_dir.exists():
        return None
    checkpoints = sorted(
        checkpoint_dir.glob("step_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    return checkpoints[-1] if checkpoints else None


def load_checkpoint_for_resume(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: EMA,
    device: torch.device,
) -> Tuple[int, int]:
    """
    Restore full training state from a checkpoint — model weights,
    optimizer state (Adam's momentum/variance buffers), and EMA shadow
    weights, so resuming doesn't cause a cold-start "bump" in the loss.

    Returns:
        (global_step, epoch) to resume from
    """
    print(f"[train] resuming from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    ema.load_state_dict(checkpoint["ema_state_dict"])

    global_step = checkpoint["global_step"]
    epoch = checkpoint["epoch"]

    print(f"[train] resumed at global_step={global_step}, epoch={epoch}")
    return global_step, epoch


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

    # ── LR warmup ────────────────────────────────────────────────────────────
    # WHY THIS IS THE STANDARD FIX FOR EARLY-TRAINING INSTABILITY: every
    # reference DDPM training script (HuggingFace diffusers, lucidrains)
    # uses a linear warmup before the optimizer is trusted with the full
    # LR. This is the proactive fix; our earlier attempts at reactive
    # "detect and recover from instability" code never matched a warmup's
    # simplicity or reliability.
    warmup_steps = cfg.experiment.training.get("lr_warmup_steps", 500)
    lr_scheduler = get_lr_warmup_scheduler(optimizer, warmup_steps)

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

    # WHY mixed_precision DEFAULTS TO OFF: AMP loss-scale overflow caused
    # two real NaN/Inf crashes on this model+GPU combination. fp32 has
    # shown zero numerical issues. The plumbing is kept (one-line config
    # flip to re-enable) for anyone training a larger model on a GPU with
    # more headroom, but the recommended value here is false.
    use_amp = cfg.experiment.training.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── MLflow run setup ───────────────────────────────────────────────────
    tracking.init_tracking(experiment_name="tiny-diffusion-cifar10")

    run_tags = dict(cfg.experiment.tags) if "tags" in cfg.experiment else {}

    # ── Resume detection ────────────────────────────────────────────────────
    checkpoint_dir = get_checkpoint_dir()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if not is_running_on_vertex_ai():
        dvc_pull_latest_checkpoint(checkpoint_dir)

    resume_checkpoint = find_latest_checkpoint(checkpoint_dir)

    if resume_checkpoint is not None:
        run_tags["resumed_from"] = resume_checkpoint.name
        print(f"[train] found existing checkpoint, will resume: {resume_checkpoint}")
    else:
        print("[train] no existing checkpoint found, starting fresh")

    with tracking.start_run(run_name=cfg.experiment.experiment_name, tags=run_tags):
        tracking.log_config(model_config.to_dict(), prefix="model.")
        tracking.log_config(
            {
                "lr": cfg.experiment.training.lr,
                "batch_size": cfg.experiment.training.batch_size,
                "num_epochs": cfg.experiment.training.num_epochs,
                "ema_decay": cfg.experiment.training.ema_decay,
                "mixed_precision": use_amp,
                "lr_warmup_steps": warmup_steps,
            },
            prefix="training.",
        )
        tracking.log_seed(cfg.experiment.seed)

        # ── Training state ───────────────────────────────────────────────
        if resume_checkpoint is not None:
            global_step, start_epoch = load_checkpoint_for_resume(
                resume_checkpoint, model, optimizer, ema, device
            )
            # WHY WE FORCE-OVERRIDE LR AFTER RESUME: optimizer.load_state_dict()
            # restores param_groups verbatim, including the lr active when the
            # checkpoint was saved — if the config's lr changed since then,
            # the restored optimizer would silently keep the OLD lr.
            config_lr = cfg.experiment.training.lr
            for group in optimizer.param_groups:
                if group["lr"] != config_lr:
                    print(
                        f"[train] overriding resumed optimizer LR "
                        f"{group['lr']:.2e} → {config_lr:.2e} (from current config)"
                    )
                    group["lr"] = config_lr
            # Fast-forward the warmup scheduler so a resumed run doesn't
            # restart warmup from step 0 (warmup is already long past by
            # this point in any real resume).
            for _ in range(global_step):
                lr_scheduler.step()
        else:
            global_step = 0
            start_epoch = 0

        grad_norm_history: list = []
        confirmed_pushed_checkpoints: set = set()

        num_epochs = cfg.experiment.training.num_epochs
        log_step_every = cfg.experiment.training.log_step_metrics_every
        log_system_every = cfg.experiment.training.log_system_metrics_every
        grad_clip_norm = cfg.experiment.training.grad_clip_norm
        checkpoint_every = cfg.experiment.training.checkpoint_every_n_steps

        # ── Best-checkpoint tracking ─────────────────────────────────────────
        # Saved at the end of every epoch when ema_loss improves — cheap,
        # always useful regardless of how the rest of training behaves.
        best_ema_loss: float = float("inf")
        best_ckpt_path = checkpoint_dir / "best.pt"

        ema_loss_value: Optional[float] = None
        loss_value: float = float("nan")  # always defined, even if epoch 0's
        # first batch is skipped by the NaN guard before this gets a real
        # value — the end-of-epoch print below reads this unconditionally.
        steps_per_epoch = len(train_loader)

        print(f"[train] starting training: {num_epochs} epochs, {steps_per_epoch} steps/epoch")
        print(f"[train] lr_warmup_steps={warmup_steps}, grad_clip_norm={grad_clip_norm}")

        for epoch in range(start_epoch, num_epochs):
            # NOTE: resume granularity is per-EPOCH, not per-batch — an
            # honest, acceptable compromise. See git history for the full
            # reasoning if needed.
            epoch_start_time = time.time()

            for batch_idx, (images, labels) in enumerate(train_loader):
                step_start_time = time.time()

                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                B = images.shape[0]

                # ── Forward diffusion ────────────────────────────────────────
                t = torch.randint(0, schedule.T, (B,), device=device)
                noise = torch.randn_like(images)
                x_t = schedule.q_sample(images, t, noise)

                # ── Forward pass + loss ──────────────────────────────────────
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    noise_pred = model(x_t, t, labels)
                    loss = F.mse_loss(noise_pred, noise)

                # ── NaN/Inf guard — the ONE defensive check every real
                # training script has, nothing more elaborate. A single
                # corrupted batch costs nothing to skip; we do NOT try to
                # detect "exploded but technically finite" loss values
                # here, since that heuristic doesn't reliably distinguish
                # real instability from ordinary noisy batches and was the
                # source of more bugs than it fixed.
                if not torch.isfinite(loss):
                    print(
                        f"[WARNING] step {global_step}: loss is non-finite "
                        f"({loss.item():.4f}) — skipping this batch."
                    )
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    continue

                # ── Backward pass ────────────────────────────────────────────
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = compute_grad_norm(model)

                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()

                # ── EMA update ────────────────────────────────────────────────
                ema.update(model, global_step)

                # ── Instability monitoring (logging only, no intervention) ──
                grad_norm_history.append(grad_norm)
                if len(grad_norm_history) > 50:
                    grad_norm_history.pop(0)

                if tracking.detect_instability(grad_norm, grad_norm_history):
                    print(
                        f"[WARNING] step {global_step}: grad_norm spike "
                        f"detected ({grad_norm:.2f}) — possible instability"
                    )

                # ── Step-level metrics ───────────────────────────────────────
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

                if global_step % log_system_every == 0:
                    iter_time = time.time() - step_start_time
                    tracking.log_system_metrics(step=global_step, iteration_time_sec=iter_time)

                # ── Periodic checkpointing ───────────────────────────────────
                if global_step > 0 and global_step % checkpoint_every == 0:
                    ckpt_path = checkpoint_dir / f"step_{global_step:07d}.pt"
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

                    if is_running_on_vertex_ai():
                        confirmed_pushed_checkpoints.add(str(ckpt_path))
                    else:
                        add_ok = dvc_add_checkpoint(ckpt_path)
                        if add_ok:
                            push_ok = dvc_push_checkpoint(ckpt_path)
                            if push_ok:
                                confirmed_pushed_checkpoints.add(str(ckpt_path))

                    cleanup_old_checkpoints(
                        checkpoint_dir,
                        keep_last_n=3,
                        confirmed_pushed=confirmed_pushed_checkpoints,
                    )

                global_step += 1

            # ── End of epoch ─────────────────────────────────────────────────
            epoch_time = time.time() - epoch_start_time
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"[train] epoch {epoch} complete in {epoch_time:.1f}s "
                f"(loss={loss_value:.4f}, "
                f"ema_loss={ema_loss_value if ema_loss_value is not None else float('nan'):.4f}, "
                f"lr={current_lr:.2e})"
            )

            tracking.log_epoch_metrics(epoch=epoch, extra_metrics={"epoch_time_sec": epoch_time})

            # ── Best checkpoint ──────────────────────────────────────────────
            if ema_loss_value is not None and math.isfinite(ema_loss_value):
                if ema_loss_value < best_ema_loss:
                    best_ema_loss = ema_loss_value
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "ema_state_dict": ema.state_dict(),
                            "global_step": global_step,
                            "epoch": epoch,
                            "best_ema_loss": best_ema_loss,
                        },
                        best_ckpt_path,
                    )
                    print(
                        f"[train] new best checkpoint at epoch {epoch} "
                        f"(ema_loss={best_ema_loss:.4f}) → {best_ckpt_path}"
                    )
                    tracking.log_epoch_metrics(
                        epoch=epoch,
                        extra_metrics={
                            "best_ema_loss": best_ema_loss,
                            "best_epoch": float(epoch),
                        },
                    )

        print(f"[train] training complete. Total steps: {global_step}")
        if best_ckpt_path.exists():
            print(f"[train] best checkpoint: {best_ckpt_path} (ema_loss={best_ema_loss:.4f})")
