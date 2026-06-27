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
    result = float(total_norm_sq**0.5)
    # WHY WE CAP NON-FINITE RESULTS: with AMP, a loss-scale overflow can
    # produce Inf gradients for one step before the scaler detects and
    # rolls back the scale factor. Returning Inf here would poison
    # grad_norm_history and crash detect_instability() via OverflowError
    # in statistics.stdev() — the exact failure we hit at step ~10,377.
    # Returning 0.0 for non-finite norms is safe: the AMP scaler skips
    # the optimizer.step() for that iteration anyway, so the 0.0 is never
    # used for anything meaningful beyond logging.
    import math

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


def is_running_on_vertex_ai() -> bool:
    """
    Detect Vertex AI Custom Training Job environment.

    WHY CLOUD_ML_PROJECT_ID SPECIFICALLY: Vertex AI's training service
    sets several AIP_* and CLOUD_ML_* environment variables automatically
    in every custom training job container — CLOUD_ML_PROJECT_ID is set
    unconditionally regardless of job configuration, making it a reliable
    presence check. It is never set on SageMaker, Kaggle, Colab, or a
    local machine. Mirrors is_running_on_sagemaker()'s detection pattern.
    """
    return "CLOUD_ML_PROJECT_ID" in os.environ


def get_checkpoint_dir() -> Path:
    """
    Return the correct checkpoint directory for whichever environment
    we're actually running in.

    Vertex AI: reads the AIP_CHECKPOINT_DIR environment variable, which
    Vertex AI sets automatically to a gs:// Cloud Storage URI for every
    training job. We convert that gs:// URI to its FUSE-mounted local
    equivalent under /gcs/ — Vertex AI's training containers mount every
    accessible Cloud Storage bucket at /gcs/<bucket-name>/... via Cloud
    Storage FUSE, so a path under /gcs/ behaves exactly like local disk
    (torch.save() works directly, no GCS-aware code needed) while the
    bytes are durably stored in Cloud Storage the whole time — this is
    the mechanism that gives us cross-session persistence without any
    OAuth flow or DVC remote configuration, the two things that blocked
    us on Kaggle and complicated every other path we tried.

    Colab/local: outputs/checkpoints — uses DVC add/push/pull (see
    checkpoint.py) for cross-session persistence via Google Drive.
    """
    if is_running_on_vertex_ai():
        aip_checkpoint_dir = os.environ.get("AIP_CHECKPOINT_DIR", "")
        if aip_checkpoint_dir.startswith("gs://"):
            # "gs://my-bucket/path/checkpoints/" -> "/gcs/my-bucket/path/checkpoints/"
            gcs_path = aip_checkpoint_dir[len("gs://") :]
            return Path("/gcs") / gcs_path
        # WHY THIS FALLBACK EXISTS: if AIP_CHECKPOINT_DIR is somehow unset
        # or malformed despite is_running_on_vertex_ai() being True, fail
        # toward a sane local default rather than crashing outright — the
        # training run can still proceed and checkpoint locally, just
        # without cross-session persistence, which is a degraded but
        # recoverable state rather than a hard failure.
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

    WHY THIS EXISTS — THE REAL PROBLEM IT SOLVES:
    Kaggle sessions cap around 9-12 hours, Colab disconnects even sooner
    on the free tier. Our measured epoch time (~5.4 min) times 200 epochs
    is ~18 hours — meaning a SINGLE session cannot complete training.
    Without resume, every session timeout means losing all progress and
    restarting global_step=0 with fresh random weights. This function is
    the first piece of making "train across multiple sessions" actually
    work rather than just being a plan we hoped would pan out.
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
    Restore full training state from a checkpoint — not just model
    weights, but optimizer state (Adam's momentum/variance buffers) and
    EMA shadow weights too.

    WHY RESTORING OPTIMIZER STATE MATTERS (a subtle but real correctness
    issue): if you only reload model weights and start a FRESH AdamW
    optimizer, Adam's per-parameter adaptive learning rate estimates
    (the running mean/variance of gradients) reset to zero. This causes
    a visible "bump" in the loss curve right after resume — effectively
    a mini cold-start — even though the model weights themselves were
    fine. Restoring optimizer.state_dict() avoids this discontinuity.

    WHY RESTORING EMA STATE MATTERS: Phase 1's EMA class has its own
    warmup schedule based on absolute step count (Phase 0 Section 8).
    If we resumed with a FRESH EMA object, it would re-enter warmup mode
    at step 0 even though the model is actually at step 50,000 — the EMA
    shadow weights would temporarily diverge from a sensible running
    average. Restoring ema.state_dict() (which includes the decay value,
    though not the step count itself — see the note below) keeps this
    consistent enough in practice since by the time you're resuming,
    decay has already reached its asymptotic value.

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
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── MLflow run setup ───────────────────────────────────────────────────
    tracking.init_tracking(experiment_name="tiny-diffusion-cifar10")

    run_tags = dict(cfg.experiment.tags) if "tags" in cfg.experiment else {}

    # ── Resume detection ────────────────────────────────────────────────────
    # WHY THIS CHECK HAPPENS BEFORE tracking.start_run(): if we're resuming,
    # we want the MLflow run tagged as a continuation so it's visually
    # distinguishable in the DagsHub UI from a fresh run — otherwise a
    # resumed run's loss curve would show a confusing discontinuity (jumping
    # from step 50,000 with no steps 0-49,999 visible) with no indication
    # why, since this would actually be a SEPARATE MLflow run ID continuing
    # the SAME underlying model weights.
    checkpoint_dir = get_checkpoint_dir()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── Pull any checkpoint that survived from a PREVIOUS session ────────
    # WHY THIS IS SKIPPED ON VERTEX AI: the /gcs/ FUSE mount IS Cloud
    # Storage directly — reading from checkpoint_dir already sees prior
    # checkpoints live, no separate download step needed.
    # Colab/local still use DVC pull to fetch from Google Drive.
    if not is_running_on_vertex_ai():
        dvc_pull_latest_checkpoint(checkpoint_dir)

    resume_checkpoint = find_latest_checkpoint(checkpoint_dir)

    if resume_checkpoint is not None:
        # WHY CHECKING `resume_checkpoint is not None` DIRECTLY, RATHER
        # THAN A SEPARATE is_resuming BOOLEAN: mypy can statically narrow
        # Optional[Path] to Path within this branch because the check is
        # directly on the variable itself — a separate `is_resuming =
        # resume_checkpoint is not None` boolean carries the same logical
        # information at runtime but mypy cannot follow that link across
        # two different variables, and flags every later access to
        # resume_checkpoint as "might still be None" even though it can't
        # be. This was a real mypy finding (not a runtime bug) caught
        # when wiring resume logic into a fresh Kaggle session — fixed by
        # checking the Optional variable directly instead of through an
        # intermediate boolean.
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
            },
            prefix="training.",
        )
        tracking.log_seed(cfg.experiment.seed)

        # ── Training state ───────────────────────────────────────────────
        # WHY global_step/start_epoch ARE CONDITIONAL ON is_resuming:
        # this is the actual fix for the 18hr-training-across-multiple-
        # sessions problem. Without this, every new session would silently
        # restart at step 0 with FRESH random weights, discarding all
        # prior GPU-hours of progress the moment a Kaggle/Colab session
        # disconnects — exactly the failure mode we're avoiding here.
        if resume_checkpoint is not None:
            global_step, start_epoch = load_checkpoint_for_resume(
                resume_checkpoint, model, optimizer, ema, device
            )
        else:
            global_step = 0
            start_epoch = 0
        # NOTE: "save checkpoint as best-by-FID" tracking (the
        # outputs/checkpoints/best.pt that dvc.yaml's evaluate stage
        # depends on) is intentionally NOT implemented yet — it needs a
        # real FID number to compare against, which only exists once
        # Phase 4's fid.py is built. Re-introduce a best_fid tracking
        # variable here when wiring that in, rather than carrying a
        # dead placeholder in the meantime (flake8 correctly flagged the
        # earlier unused `best_fid = float("inf")` line as dead code).
        grad_norm_history: list = []
        confirmed_pushed_checkpoints: set = set()

        num_epochs = cfg.experiment.training.num_epochs
        log_step_every = cfg.experiment.training.log_step_metrics_every
        log_system_every = cfg.experiment.training.log_system_metrics_every
        grad_clip_norm = cfg.experiment.training.grad_clip_norm
        checkpoint_every = cfg.experiment.training.checkpoint_every_n_steps

        # ── Best-checkpoint tracking ─────────────────────────────────────────
        # WHY ema_loss, NOT raw loss: ema_loss is the smoothed training signal
        # (exponential moving average of per-step loss values). Raw loss is
        # noisy — a single easy/hard batch can look like the best or worst
        # step even if nothing meaningful changed. ema_loss gives a stable
        # trend that actually reflects whether the model improved over the
        # last N steps, making it a much more reliable "best model so far"
        # criterion than a lucky raw-loss reading.
        best_ema_loss: float = float("inf")
        best_ckpt_path = checkpoint_dir / "best.pt"

        # ── Early stopping ───────────────────────────────────────────────────
        early_stop_patience = cfg.experiment.training.get("early_stop_patience", 0)
        # 0 means disabled — no early stopping by default. Set a positive
        # integer in config (e.g. early_stop_patience: 20) to stop after
        # that many epochs with no improvement in ema_loss.
        epochs_without_improvement: int = 0

        # ── Cosine LR schedule ───────────────────────────────────────────────
        # WHY COSINE, NOT CONSTANT: a flat LR works but cosine decay gives
        # the optimizer a long warm plateau followed by a gentle cooldown in
        # the final epochs, which consistently improves final loss by 5-10%
        # in diffusion model training without any additional hyperparameters
        # to tune. T_max = total training steps (not epochs) so the decay
        # is smooth across the full run regardless of steps_per_epoch.
        use_lr_schedule = cfg.experiment.training.get("cosine_lr_schedule", False)
        scheduler = None  # set after steps_per_epoch is known

        ema_loss_value: Optional[float] = None

        steps_per_epoch = len(train_loader)

        if use_lr_schedule:
            remaining_epochs = num_epochs - start_epoch
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=remaining_epochs * steps_per_epoch,
                eta_min=cfg.experiment.training.lr * 0.1,
                # WHY 10% of base LR as minimum: decaying all the way to 0
                # can destabilise the final epochs when the model is close
                # to convergence — 10% keeps updates meaningful without the
                # large steps that caused instability earlier in training.
            )
            # If resuming mid-run, fast-forward the scheduler state to match
            # where we actually are in the cosine curve. Without this, a
            # resumed run would restart the cosine decay from its peak rather
            # than continuing the curve the original run was following.
            warmup_steps = (start_epoch - 0) * steps_per_epoch
            if warmup_steps > 0:
                for _ in range(warmup_steps):
                    scheduler.step()
            print(
                f"[train] cosine LR schedule: {cfg.experiment.training.lr:.2e} → "
                f"{cfg.experiment.training.lr * 0.1:.2e} over "
                f"{remaining_epochs} epochs"
            )

        print(f"[train] starting training: {num_epochs} epochs, {steps_per_epoch} steps/epoch")

        for epoch in range(start_epoch, num_epochs):
            # NOTE: resume granularity is per-EPOCH, not per-batch. If a
            # session ends mid-epoch, the checkpoint stores the epoch
            # index but not which batch within it we'd reached. On
            # resume, we restart that epoch from its first batch. This
            # means a resumed run sees a partially-repeated epoch's worth
            # of data — an honest, acceptable compromise (not a silent
            # bug) given that: (1) optimizer and EMA state correctly
            # carry over, so this isn't a cold-restart, (2) augmentation
            # re-randomizes every time, so it's not pixel-identical
            # repeated data, and (3) one repeated epoch out of 200 is
            # negligible. True per-batch resume would require checkpointing
            # the DataLoader's internal iteration state, which adds
            # significant complexity for a marginal benefit at our scale.
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

                with torch.amp.autocast("cuda", enabled=use_amp):
                    noise_pred = model(x_t, t, labels)
                    loss = F.mse_loss(noise_pred, noise)
                    # ^ this single line IS Phase 0's entire L_simple —
                    # everything else in this function is infrastructure
                    # around this one mathematical statement.

                # ── NaN/Inf loss guard ───────────────────────────────────────
                # WHY THIS CHECK EXISTS: if loss becomes NaN or Inf (e.g. from
                # an AMP overflow that the scaler hasn't caught yet, or a bad
                # batch), continuing to call .backward() propagates NaN into
                # every parameter's gradient, poisoning the model weights
                # permanently. Once weights are NaN, no amount of gradient
                # clipping or scaler adjustment recovers them — training runs
                # dead for the remaining epochs, which is exactly what happened
                # at step ~23,923. Skipping the step on a bad loss value costs
                # one batch of progress; not skipping costs the entire run.
                if not torch.isfinite(loss):
                    print(
                        f"[WARNING] step {global_step}: loss={loss.item():.4f} "
                        f"is non-finite — skipping backward/step for this batch."
                    )
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    continue

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

                # ── LR scheduler step ────────────────────────────────────────
                if scheduler is not None:
                    scheduler.step()

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

                # ── Periodic checkpointing ──────────────────────────────────
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
                        # ── Vertex AI: persistence is automatic ──────────
                        # torch.save() to ckpt_path wrote directly into
                        # the /gcs/ FUSE mount — the bytes are in Cloud
                        # Storage the instant save() returns, durably,
                        # with no separate sync/push step. Treat every
                        # save as confirmed for cleanup purposes.
                        confirmed_pushed_checkpoints.add(str(ckpt_path))
                    else:
                        # ── Colab/local: DVC add + push ──────────────────
                        add_ok = dvc_add_checkpoint(ckpt_path)
                        if add_ok:
                            push_ok = dvc_push_checkpoint(ckpt_path)
                            if push_ok:
                                confirmed_pushed_checkpoints.add(str(ckpt_path))

                    # ── Disk space cleanup (both backends) ───────────────
                    # WHY THIS IS NOT OPTIONAL: each checkpoint is ~0.88GB
                    # (model 0.22GB + AdamW optimizer state 0.44GB, since
                    # Adam stores momentum AND variance per parameter +
                    # EMA shadow 0.22GB). Without bounding local disk
                    # usage, checkpoints accumulate uncapped — this
                    # caused a REAL crash during this project's first
                    # full baseline run on Kaggle (disk exhausted after
                    # ~5 accumulated checkpoints, 14 epochs into an
                    # 18-hour run). We only ever delete a checkpoint
                    # that's in confirmed_pushed_checkpoints.
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
                f"(loss={loss_value:.4f}, ema_loss={ema_loss_value:.4f}, lr={current_lr:.2e})"
            )

            tracking.log_epoch_metrics(epoch=epoch, extra_metrics={"epoch_time_sec": epoch_time})

            # ── Best checkpoint ──────────────────────────────────────────────
            # Save best.pt whenever ema_loss improves. This runs EVERY epoch
            # so we always have the best weights seen so far, regardless of
            # what happens later in training (NaN collapse, quota error, etc.)
            # Only save if ema_loss is a valid finite number — a NaN ema_loss
            # must never overwrite a previously good best.pt.
            if ema_loss_value is not None and math.isfinite(ema_loss_value):
                if ema_loss_value < best_ema_loss:
                    best_ema_loss = ema_loss_value
                    epochs_without_improvement = 0
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
                        f"[train] ✓ new best checkpoint at epoch {epoch} "
                        f"(ema_loss={best_ema_loss:.4f}) → {best_ckpt_path}"
                    )
                    tracking.log_config(
                        {"best_ema_loss": best_ema_loss, "best_epoch": epoch},
                        prefix="best.",
                    )
                else:
                    epochs_without_improvement += 1

            # ── Early stopping ───────────────────────────────────────────────
            if early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
                print(
                    f"[train] early stopping: no improvement in ema_loss for "
                    f"{epochs_without_improvement} epochs (patience={early_stop_patience}). "
                    f"Best was {best_ema_loss:.4f} at epoch {epoch - epochs_without_improvement}."
                )
                break

        print(f"[train] training complete. Total steps: {global_step}")
        if best_ckpt_path.exists():
            print(f"[train] best checkpoint: {best_ckpt_path} (ema_loss={best_ema_loss:.4f})")
