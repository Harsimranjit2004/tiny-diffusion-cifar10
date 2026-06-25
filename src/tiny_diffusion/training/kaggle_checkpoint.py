"""
src/tiny_diffusion/training/kaggle_checkpoint.py

PHASE 3 — KAGGLE DATASETS CHECKPOINT BACKEND

WHY THIS EXISTS — THE FULL STORY:
  We needed checkpoint persistence across Kaggle's 9-hour session cap.
  DVC + Google Drive was the first attempt — it failed because Google's
  OAuth consent flow requires a human clicking "Allow" in a browser,
  which is fundamentally incompatible with Kaggle's non-interactive
  "Save & Run All" batch execution (confirmed via direct testing: the
  push call hung for the full 300s timeout waiting on an OAuth redirect
  that could never complete). We then tried AWS SageMaker Training Jobs
  with native S3 checkpoint sync — mechanically this worked end-to-end
  (IAM role, S3 bucket, job submission all succeeded), but AWS denied
  our GPU training-job quota request outright, citing insufficient
  account usage history, and confirmed via direct quota checks that
  EVERY GPU instance type (g4dn, g5) shows 0 training-job quota on this
  account — not a per-instance-type issue, a blanket new-account
  restriction.

  Kaggle's own Datasets feature sidesteps both problems: authentication
  uses a simple API token (like DagsHub's, not an OAuth browser flow),
  and there's no AWS quota involved since we never leave Kaggle's
  infrastructure. This is the third and (for now) final backend we
  support, alongside DVC (kept for potential future use elsewhere) and
  SageMaker (kept for if/when the AWS quota gets approved).

HOW THIS WORKS MECHANICALLY:
  - At the END of a session (or periodically during training), we
    upload outputs/checkpoints/ as a new VERSION of a Kaggle Dataset
    using kagglehub.dataset_upload().
  - At the START of the next session, the notebook adds that dataset as
    an INPUT (done once, manually, via Kaggle's "Add Data" UI — this is
    the one piece that isn't fully automatable, since attaching a
    dataset as notebook input is a notebook-configuration action, not
    a runtime API call) — then our code copies its contents from the
    read-only input path into outputs/checkpoints/ before training starts.

WHY THIS DESIGN, NOT A PER-CHECKPOINT UPLOAD:
  Unlike DVC's per-file add+push, Kaggle Dataset versioning works at the
  GRANULARITY OF THE WHOLE DATASET — every dataset_upload() call creates
  a new version containing everything in the directory, not an
  incremental diff of one file. Uploading on every single checkpoint
  interval (e.g. every 1000 steps) would create many large dataset
  versions in quick succession and waste a lot of upload bandwidth
  re-sending checkpoints that haven't changed. Instead, we upload ONCE
  near the end of a session (see train.py's integration point) — the
  local cleanup logic still bounds disk usage during the session itself,
  and only the survivors at session-end get uploaded.
"""

import os
import shutil
from pathlib import Path
from typing import Optional


def is_running_on_kaggle() -> bool:
    """
    Detect Kaggle notebook environment.

    WHY KAGGLE_KERNEL_RUN_TYPE SPECIFICALLY: Kaggle sets this environment
    variable automatically in every notebook session (values like
    "Interactive" or "Batch") — it is never present on SageMaker, Colab,
    or a local machine. Mirrors the same detection pattern as
    is_running_on_sagemaker() in train.py (SM_TRAINING_ENV).
    """
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def restore_checkpoints_from_kaggle_dataset(
    checkpoint_dir: Path,
    dataset_input_path: Optional[Path] = None,
) -> bool:
    """
    Copy checkpoint files from a Kaggle Dataset (attached as notebook
    input) into the local checkpoint_dir, at the START of a session,
    BEFORE find_latest_checkpoint() looks for anything to resume from.

    WHY dataset_input_path DEFAULTS TO A SPECIFIC KAGGLE PATH: when you
    attach a dataset as notebook input via the UI, Kaggle mounts it
    read-only at /kaggle/input/<dataset-slug>/. We can't know the exact
    slug in advance (it depends on what you named the dataset), so the
    caller (train.py) must either pass this explicitly or we fall back
    to scanning /kaggle/input/ for ANY folder containing checkpoint files
    — see the fallback logic below.

    Returns True if checkpoints were found and restored, False if none
    were found (e.g. genuinely the first-ever session, nothing to
    restore yet — this is a normal, non-error condition).
    """
    if dataset_input_path is None:
        # Scan /kaggle/input/ for any attached dataset containing our
        # checkpoint files — this lets the SAME code work regardless of
        # what you happened to name the dataset when creating it.
        kaggle_input_root = Path("/kaggle/input")
        if not kaggle_input_root.exists():
            print(
                "[kaggle_checkpoint] /kaggle/input does not exist — "
                "not running on Kaggle, or no datasets attached."
            )
            return False

        candidate_dirs = [
            d for d in kaggle_input_root.iterdir() if d.is_dir() and any(d.glob("step_*.pt"))
        ]
        if not candidate_dirs:
            print(
                "[kaggle_checkpoint] No attached dataset contains "
                "checkpoint files — this is normal for a first-ever "
                "session with nothing to restore yet."
            )
            return False
        if len(candidate_dirs) > 1:
            print(
                f"[kaggle_checkpoint] WARNING — multiple attached "
                f"datasets contain checkpoint files: {candidate_dirs}. "
                f"Using the first one found ({candidate_dirs[0]}) — "
                f"if this is wrong, pass dataset_input_path explicitly."
            )
        dataset_input_path = candidate_dirs[0]

    if not dataset_input_path.exists():
        print(f"[kaggle_checkpoint] {dataset_input_path} does not exist.")
        return False

    checkpoint_files = list(dataset_input_path.glob("step_*.pt"))
    if not checkpoint_files:
        print(f"[kaggle_checkpoint] No checkpoint files found in " f"{dataset_input_path}.")
        return False

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for f in checkpoint_files:
        dest = checkpoint_dir / f.name
        shutil.copy2(f, dest)
        print(f"[kaggle_checkpoint] restored {f.name} from Kaggle Dataset")

    print(
        f"[kaggle_checkpoint] restored {len(checkpoint_files)} "
        f"checkpoint(s) from {dataset_input_path}"
    )
    return True


def upload_checkpoints_to_kaggle_dataset(
    checkpoint_dir: Path,
    dataset_handle: str,
) -> bool:
    """
    Upload the current contents of checkpoint_dir as a new version of a
    Kaggle Dataset, for persistence across sessions.

    Args:
        checkpoint_dir: local directory containing checkpoint files
        dataset_handle: "<your-kaggle-username>/<dataset-slug>", e.g.
                        "harsimranjit2004/tiny-diffusion-checkpoints" —
                        must already exist as a dataset (create it once
                        manually via the Kaggle UI before first use; this
                        function only creates NEW VERSIONS of an existing
                        dataset, not the dataset itself, since dataset
                        creation requires UI-driven metadata setup the
                        kagglehub API doesn't fully replace).

    WHY THIS NEVER RAISES ON FAILURE: matches the same fault-tolerance
    pattern as checkpoint.py's DVC functions — a failed upload (network
    issue, API rate limit) should warn loudly but never crash hours of
    training progress that's still safely on local disk.
    """
    try:
        import kagglehub
    except ImportError:
        print(
            "[kaggle_checkpoint] WARNING — kagglehub not installed. "
            "Install with: pip install kagglehub. "
            "Checkpoints will NOT persist past this session."
        )
        return False

    try:
        kagglehub.dataset_upload(
            dataset_handle,
            str(checkpoint_dir),
        )
        print(
            f"[kaggle_checkpoint] uploaded {checkpoint_dir} to " f"Kaggle Dataset {dataset_handle}"
        )
        return True
    except Exception as e:
        # WHY A BARE Exception CATCH HERE (unlike checkpoint.py's more
        # specific subprocess exception types): kagglehub's upload can
        # fail in several different ways (network, auth, rate limit,
        # malformed handle) that aren't all cleanly typed exceptions in
        # its current API — we genuinely want to catch anything here
        # and warn rather than let an unexpected exception type crash
        # what's otherwise a non-critical, best-effort persistence step.
        print(
            f"[kaggle_checkpoint] WARNING — upload FAILED: {e}\n"
            f"  Training will continue; this checkpoint will only "
            f"exist on local (ephemeral) disk for this session."
        )
        return False
