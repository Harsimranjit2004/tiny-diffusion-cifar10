"""
=============================================================================
src/tiny_diffusion/utils/tracking.py

PHASE 2, STEP 2 — MLFLOW EXPERIMENT TRACKING
=============================================================================

WHAT THIS MODULE DOES:
  Wraps MLflow so the rest of the codebase never calls `mlflow.xxx()` directly.
  Instead, training/eval/quantization code calls our own small API
  (init_tracking, log_params, log_step_metrics, log_epoch_metrics, etc.)
  This indirection matters for two reasons:
    1. If we ever swap MLflow for WandB (Phase 2 Step 2 also asks us to
       evaluate this), only THIS file changes — not every call site across
       training.py, evaluation.py, quantization scripts, etc.
    2. We can enforce OUR experiment schema (what gets logged, with what
       naming convention) in one place rather than scattered across the
       codebase, which is how schemas drift and become inconsistent.

WHY DAGSHUB AS THE BACKEND (recap from our earlier discussion):
  DagsHub hosts a real MLflow-compatible tracking server for free. We do
  NOT run `mlflow server` ourselves — there's no server to manage, no port
  to keep open, no process to babysit on Kaggle/Colab. We just point
  mlflow.set_tracking_uri() at DagsHub's URL and authenticate via env vars.
  This persists across ephemeral Kaggle/Colab sessions automatically,
  because the data lives on DagsHub's servers, not on the local disk that
  gets wiped when the session ends.

THE EXPERIMENT SCHEMA — what we log, and why each category exists:

  PARAMS (logged once per run, at the start):
    - All architecture hyperparameters (base_channels, channel_mult, etc.)
    - All training hyperparameters (lr, batch_size, optimizer settings)
    - All noise schedule parameters (T, schedule type, cosine_s)
    - Git commit hash (so every run is traceable to exact code state)
    - Random seed
    WHY: params are immutable facts about the run's configuration. MLflow
    lets you filter/sort/compare runs by any param in its UI — this is how
    you do an ablation study without manually maintaining a spreadsheet.

  METRICS PER STEP (logged every N training steps):
    - loss (raw MSE loss this step)
    - ema_loss (smoothed loss, exponential moving average of the metric
      itself — NOT the same as EMA model weights, a common confusion)
    - grad_norm (L2 norm of gradients before clipping — instability signal)
    - learning_rate (in case of LR scheduling, track what it actually was)
    WHY: step-level granularity is what lets you SEE instability — a loss
    spike at step 4,231 is invisible in an epoch-averaged metric but
    obvious in a step-level MLflow chart.

  METRICS PER EPOCH (logged once per epoch — more expensive to compute):
    - fid_1k (FID computed on a cheap 1,000-sample batch — full 10k+ FID
      is too slow to run every epoch, see Phase 4 for the full pipeline)
    - sample_grid (an actual image artifact — a grid of generated samples
      logged as a PNG, so you can SEE quality progression, not just infer
      it from a number)
    WHY: FID and visual samples are expensive (require running the full
    reverse diffusion sampler), so we compute them at epoch granularity,
    not step granularity.

  SYSTEM METRICS (logged every N steps, alongside training metrics):
    - gpu_utilization_pct
    - gpu_memory_used_gb
    - iteration_time_sec
    WHY: this is how we diagnose "are we compute-bound or data-bound" —
    directly from your ML Systems textbook's Roofline framing. If GPU
    utilization is low while iteration time is high, we're data-loading
    bound, not compute bound — a completely different fix.

  ARTIFACTS (files logged to MLflow, not just scalar numbers):
    - model checkpoints (.pt files)
    - sample grids (.png files)
    - loss curve plots (.png, generated periodically)
    - the noise schedule visualization (alpha_bar_t curve, logged ONCE
      at the start of training to sanity-check the schedule visually)

  TAGS (logged once per run — used for FILTERING runs in the UI):
    - architecture_variant (e.g. "baseline", "wider_channels")
    - quantization_level (e.g. "fp32", "fp16", "int8_dynamic") — Phase 5
    - sampler_type (e.g. "ddpm", "ddim") — Phase 4
    WHY tags vs params: tags are specifically designed for filtering/
    grouping in the MLflow UI ("show me all int8 runs"), while params are
    for exact-value comparison. Using tags for categorical groupings keeps
    the run comparison table readable instead of one giant flat list.
=============================================================================
"""

import os
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Optional

import mlflow

# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class TrackingConfig:
    """
    Controls how often we log each category — directly reflects the
    'what to log every 10 steps / 100 steps / epoch' structure from
    the original project plan.
    """

    log_step_metrics_every: int = 10  # loss, grad_norm — cheap, log often
    log_system_metrics_every: int = 100  # GPU util/mem — slightly more overhead
    # FID + sample grids happen once per epoch in the training loop itself,
    # not on a step-count interval, so there's no field for that here.


# =============================================================================
# INITIALIZATION
# =============================================================================


def get_git_commit_hash() -> str:
    """
    Capture the exact git commit a run was trained under.

    WHY THIS MATTERS FOR REPRODUCIBILITY: a checkpoint is only meaningfully
    reproducible if you know EXACTLY which version of the code produced it.
    "I changed the ResBlock slightly after this run" is a common way
    experiments become impossible to reproduce. Logging the commit hash as
    an MLflow param means every run is permanently traceable to exact code.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        commit_hash = result.stdout.strip()

        # Also check for uncommitted changes — if the working tree is dirty,
        # the commit hash alone doesn't fully describe what code actually ran.
        dirty_check = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        is_dirty = len(dirty_check.stdout.strip()) > 0

        return f"{commit_hash}{'-dirty' if is_dirty else ''}"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown-not-a-git-repo"


def init_tracking(experiment_name: str = "tiny-diffusion-cifar10") -> None:
    """
    Call this ONCE at the start of your training script, before any
    logging calls. Reads MLFLOW_TRACKING_URI / USERNAME / PASSWORD from
    environment variables (which come from your local .env file — see
    Phase 2 Step 2's .env setup instructions).

    Args:
        experiment_name: groups related runs together in the MLflow UI.
                         We use ONE experiment name for this whole project;
                         individual ablations are distinguished by tags
                         and params within that experiment, not by creating
                         a new "experiment" per ablation.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise EnvironmentError(
            "MLFLOW_TRACKING_URI not set. Did you create .env from "
            ".env.example and load it? On Kaggle/Colab, set this as a "
            "secret/env var at the start of your notebook session — see "
            "Phase 2 Step 2 instructions for the exact DagsHub URL format."
        )

    # WHY WE EMBED CREDENTIALS INTO THE URI INSTEAD OF RELYING ON THE
    # SEPARATE MLFLOW_TRACKING_USERNAME / PASSWORD ENV VARS:
    # In containerised environments (Vertex AI custom training jobs), mlflow
    # 2.x does not reliably pick up those two env vars when making the initial
    # REST call to set_experiment() — the server returns a 401 even when both
    # vars are present and correct. Embedding credentials directly in the URI
    # as https://user:pass@host/path is the guaranteed HTTP Basic Auth path
    # that works across every mlflow version and every container runtime,
    # because the credentials travel with every request rather than being
    # looked up separately. This is the same pattern we use for the git clone
    # URL (DAGSHUB_TOKEN embedded in the HTTPS URL) — non-interactive
    # containers need credentials baked in, not fetched via a side channel.
    username = os.environ.get("MLFLOW_TRACKING_USERNAME", "")
    password = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")
    if username and password and "://" in tracking_uri and "@" not in tracking_uri:
        scheme, rest = tracking_uri.split("://", 1)
        tracking_uri = f"{scheme}://{username}:{password}@{rest}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Redact credentials from the printed URI so the token never appears in logs.
    logged_uri = tracking_uri
    if password and password in logged_uri:
        logged_uri = logged_uri.replace(password, "***TOKEN***")
    print(f"[tracking] MLflow tracking URI: {logged_uri}")
    print(f"[tracking] Experiment: {experiment_name}")


# =============================================================================
# RUN LIFECYCLE
# =============================================================================


def start_run(
    run_name: str,
    tags: Optional[Dict[str, str]] = None,
) -> "mlflow.ActiveRun":
    """
    Start a new MLflow run. Returns the MLflow run context manager —
    use this with `with start_run(...) as run:` in training scripts.

    Args:
        run_name: human-readable name shown in the MLflow UI
                  (e.g. "baseline_55M_cosine_t1000")
        tags: categorical metadata for filtering runs in the UI.
              Example: {"architecture_variant": "baseline",
                        "quantization_level": "fp32",
                        "sampler_type": "ddpm"}
    """
    run_tags = tags or {}
    # Always attach git commit — this is non-negotiable for every run,
    # not optional metadata, per the reproducibility reasoning above.
    run_tags["git_commit"] = get_git_commit_hash()

    return mlflow.start_run(run_name=run_name, tags=run_tags)


# =============================================================================
# LOGGING: PARAMS (once per run)
# =============================================================================


def log_config(config_dict: Dict[str, Any], prefix: str = "") -> None:
    """
    Log a full config dict as MLflow params, once at the start of a run.

    Args:
        config_dict: typically model_config.to_dict() merged with
                     training/data config dicts — everything that defines
                     "what exactly was this run."
        prefix: optional namespace prefix, e.g. "model." or "training."
               to avoid name collisions when logging multiple config
               objects (model config + training config + data config all
               in the same run).

    WHY FLATTEN INTO A SINGLE LOG CALL: MLflow's log_params() takes a dict
    and logs every key-value pair as a separate param in one network call,
    rather than N separate log_param() calls — much faster, especially
    relevant since we may be on a slow Kaggle/Colab network connection.
    """
    if prefix:
        config_dict = {f"{prefix}{k}": v for k, v in config_dict.items()}
    mlflow.log_params(config_dict)


def log_seed(seed: int) -> None:
    """Log the random seed — part of the params schema, kept separate
    since it's not part of any single config dataclass."""
    mlflow.log_param("seed", seed)


# =============================================================================
# LOGGING: STEP-LEVEL METRICS (every N steps)
# =============================================================================


def log_step_metrics(
    step: int,
    loss: float,
    ema_loss: Optional[float] = None,
    grad_norm: Optional[float] = None,
    learning_rate: Optional[float] = None,
) -> None:
    """
    Log per-step training metrics. Call this every `log_step_metrics_every`
    steps (see TrackingConfig) — NOT every single step, since that would
    flood MLflow with network calls and slow down training for no real
    observability benefit (no human reads metrics at single-step granularity
    anyway; the chart resolution is the same whether you log every step or
    every 10th step, but the network overhead is 10x different).

    Args:
        step: global training step (not epoch — this is the x-axis for
              the loss curve, which needs step-level resolution to show
              instability, per the schema reasoning above)
        loss: raw MSE loss this step
        ema_loss: exponentially-smoothed loss VALUE for cleaner trend
                  visualization. THIS IS NOT THE SAME THING AS EMA MODEL
                  WEIGHTS (Phase 1's EMA class) — naming collision to be
                  careful about. This EMA is just a smoothed metric for
                  plotting; the model-weights EMA is a completely separate
                  concept that affects what weights you use for sampling.
        grad_norm: L2 norm of gradients before clipping — spikes here are
                  often the first visible sign of training instability,
                  often BEFORE the loss itself visibly spikes.
        learning_rate: log the actual LR at this step, useful if using
                       any scheduler (warmup, cosine decay, etc.)
    """
    metrics = {"loss": loss}
    if ema_loss is not None:
        metrics["ema_loss"] = ema_loss
    if grad_norm is not None:
        metrics["grad_norm"] = grad_norm
    if learning_rate is not None:
        metrics["learning_rate"] = learning_rate

    mlflow.log_metrics(metrics, step=step)


def detect_instability(
    grad_norm: float, grad_norm_history: list, spike_threshold: float = 4.0
) -> bool:
    """
    Automatic gradient norm spike detection — flags potential training
    instability without requiring a human to stare at a chart in real time.

    Args:
        grad_norm: current step's gradient norm
        grad_norm_history: list of recent grad_norm values (e.g. last 50 steps)
        spike_threshold: how many standard deviations above the recent
                         mean counts as a "spike." 4.0 is a deliberately
                         conservative threshold — gradient norms are
                         naturally noisy, and a lower threshold (e.g. 2.0)
                         would flag normal variance as false positives.

    Returns:
        True if this step's grad_norm is anomalously high relative to
        recent history — caller should log a warning and consider this
        a signal worth investigating (not necessarily stopping training).
    """
    if len(grad_norm_history) < 10:
        # Not enough history yet to establish a baseline — never flag
        # instability in the first 10 logged steps of a run.
        return False

    import statistics

    mean = statistics.mean(grad_norm_history)
    stdev = statistics.stdev(grad_norm_history) if len(grad_norm_history) > 1 else 0

    if stdev == 0:
        return False

    z_score = (grad_norm - mean) / stdev
    return bool(z_score > spike_threshold)


# =============================================================================
# LOGGING: SYSTEM METRICS (every N steps)
# =============================================================================


def get_gpu_metrics() -> Dict[str, float]:
    """
    Query current GPU utilization and memory usage.

    WHY THIS CONNECTS TO THE ROOFLINE MODEL (ML Systems textbook):
    GPU utilization % tells you whether you're compute-bound or
    memory/data-bound. High iteration time + LOW gpu_utilization means
    the GPU is sitting idle waiting for data — a data-loading bottleneck,
    not a compute bottleneck. The fix for that (more DataLoader workers,
    pin_memory=True) is completely different from the fix for genuine
    compute-bound slowness (smaller model, fewer attention layers, etc.)
    Without logging this metric, you'd misdiagnose which problem you have.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return {}

        memory_used_gb = torch.cuda.memory_allocated() / 1e9
        memory_reserved_gb = torch.cuda.memory_reserved() / 1e9

        # torch doesn't expose utilization % directly — that requires
        # nvidia-ml-py (pynvml). We try it, but fall back gracefully
        # since it's not installed by default on all platforms.
        utilization_pct = None
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            utilization_pct = util.gpu
        except ImportError:
            pass

        metrics = {
            "gpu_memory_allocated_gb": memory_used_gb,
            "gpu_memory_reserved_gb": memory_reserved_gb,
        }
        if utilization_pct is not None:
            metrics["gpu_utilization_pct"] = float(utilization_pct)
        return metrics

    except ImportError:
        return {}


def log_system_metrics(step: int, iteration_time_sec: float) -> None:
    """
    Log GPU utilization, memory, and iteration timing.
    Call every `log_system_metrics_every` steps (less frequent than loss,
    since querying GPU state has more overhead than reading a loss tensor).
    """
    metrics = get_gpu_metrics()
    metrics["iteration_time_sec"] = iteration_time_sec
    mlflow.log_metrics(metrics, step=step)


# =============================================================================
# LOGGING: EPOCH-LEVEL METRICS (once per epoch — expensive)
# =============================================================================


def log_epoch_metrics(
    epoch: int,
    fid_1k: Optional[float] = None,
    extra_metrics: Optional[Dict[str, float]] = None,
) -> None:
    """
    Log epoch-level evaluation metrics. FID requires running the full
    reverse diffusion sampler on ~1000 images — expensive — so this is
    epoch granularity, not step granularity. See Phase 4 for the full
    FID pipeline; this function just logs whatever number that pipeline
    produces.
    """
    metrics = {}
    if fid_1k is not None:
        metrics["fid_1k"] = fid_1k
    if extra_metrics:
        metrics.update(extra_metrics)
    if metrics:
        mlflow.log_metrics(metrics, step=epoch)


def log_sample_grid(image_path: str, epoch: int) -> None:
    """
    Log a generated sample grid image as an MLflow artifact.

    Args:
        image_path: path to a PNG file (e.g. produced by torchvision's
                    make_grid + save_image on a batch of generated samples)
        epoch: used to namespace the artifact path so each epoch's
              sample grid is kept separately rather than overwritten,
              letting you scrub through quality progression in the UI.
    """
    mlflow.log_artifact(image_path, artifact_path=f"samples/epoch_{epoch:04d}")


# =============================================================================
# LOGGING: ARTIFACTS (checkpoints, plots, schedule visualization)
# =============================================================================


def log_checkpoint(checkpoint_path: str, step: int) -> None:
    """
    Log a model checkpoint file as an MLflow artifact.

    NOTE: for LARGE checkpoints (our ~55M param model is roughly 220MB in
    fp32), logging every checkpoint to MLflow as well as DVC is redundant
    storage. Our policy (finalized in Phase 2 Step 3, DVC pipeline design):
    MLflow gets the BEST checkpoint only (by FID), DVC gets every periodic
    checkpoint for full training resumability. This function is called
    selectively by the training loop, not on every checkpoint save.
    """
    mlflow.log_artifact(checkpoint_path, artifact_path=f"checkpoints/step_{step:07d}")


def log_noise_schedule_plot(plot_path: str) -> None:
    """
    Log the alpha_bar_t visualization ONCE at the start of training —
    a visual sanity check that the cosine schedule looks correct before
    spending 6 hours of T4 time training against it. See Phase 1's
    CosineNoiseSchedule.verify_schedule() for the numerical checks this
    plot visually complements.
    """
    mlflow.log_artifact(plot_path, artifact_path="schedule")


# =============================================================================
# SMOKE TEST — verify the whole module works end to end
# =============================================================================


def run_smoke_test() -> None:
    """
    Minimal end-to-end test: init tracking, start a run, log a few
    params and metrics, end the run. Run this FIRST after setting up
    your .env file, before writing the real training loop — confirms
    DagsHub connectivity and credentials work before you depend on them
    during an actual 6-hour training run.
    """
    print("=" * 70)
    print("MLFLOW TRACKING SMOKE TEST")
    print("=" * 70)

    init_tracking(experiment_name="tiny-diffusion-cifar10-smoketest")

    with start_run(
        run_name="smoke_test",
        tags={"architecture_variant": "smoke_test", "purpose": "verify_connectivity"},
    ) as run:
        print(f"\n[smoke test] Started run: {run.info.run_id}")

        # Log fake params
        log_config({"base_channels": 128, "channel_mult": "[1,2,4,8]"}, prefix="model.")
        log_seed(42)
        print("[smoke test] Logged params")

        # Log fake step metrics across a few fake steps
        grad_norm_history = []
        for step in range(0, 100, 10):
            fake_loss = 1.0 / (step + 1)
            fake_grad_norm = 2.0 + (0.1 * step)
            log_step_metrics(
                step=step,
                loss=fake_loss,
                ema_loss=fake_loss * 1.05,
                grad_norm=fake_grad_norm,
                learning_rate=2e-4,
            )
            grad_norm_history.append(fake_grad_norm)
        print("[smoke test] Logged step metrics")

        # Log fake system metrics
        log_system_metrics(step=50, iteration_time_sec=0.35)
        print("[smoke test] Logged system metrics (or skipped gracefully if no GPU)")

        # Log fake epoch metrics
        log_epoch_metrics(epoch=1, fid_1k=85.3)
        print("[smoke test] Logged epoch metrics")

        # Test instability detection
        is_unstable = detect_instability(grad_norm=50.0, grad_norm_history=grad_norm_history)
        print(f"[smoke test] Instability detection test (forced spike): {is_unstable}")

        print("\n[smoke test] Run URL: check your DagsHub repo's Experiments tab")
        print(f"[smoke test] Run ID: {run.info.run_id}")

    print("\n" + "=" * 70)
    print("SMOKE TEST COMPLETE — check DagsHub UI to confirm the run appears.")
    print("=" * 70)


if __name__ == "__main__":
    run_smoke_test()
