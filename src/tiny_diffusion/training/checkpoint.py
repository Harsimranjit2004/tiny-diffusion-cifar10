"""
src/tiny_diffusion/training/checkpoint.py

PHASE 3 — CHECKPOINT DVC INTEGRATION

WHY THIS IS A SEPARATE MODULE FROM train.py'S HOT LOOP:
  train.py's training loop calls torch.save() directly (fast, local disk
  write) on every checkpoint interval — that has to stay fast since it's
  on the critical path of every Nth training step. Actually running
  `dvc add` + `dvc push` involves hashing the file and a network call to
  Google Drive, which is much slower and would stall training if called
  inline. This module's functions are meant to be called PERIODICALLY but
  OUT OF BAND — e.g. from a background thread, or explicitly between
  epochs rather than between steps — never from inside the per-step loop.

WHEN TO ACTUALLY CALL THESE (now, in Phase 3, vs. deferred):
  For now, training.py only does the local torch.save(). Actually wiring
  DVC push calls into the live training loop is something we'll do
  carefully once we're running real multi-hour T4 training sessions
  (next step after this), since the failure modes of "DVC push fails
  mid-training on a flaky Colab network connection" need to be handled
  without crashing the whole training run. The functions below exist and
  are tested now so they're ready to wire in deliberately.
"""

import subprocess
from pathlib import Path


def dvc_pull_latest_checkpoint(checkpoint_dir: Path, timeout_sec: int = 300) -> bool:
    """
    Pull whatever checkpoints exist in the DVC remote down to local disk,
    at the START of a training session — BEFORE find_latest_checkpoint()
    looks for anything to resume from.

    WHY THIS FUNCTION IS THE MISSING PIECE FOR CROSS-SESSION RESUME:
    Kaggle/Colab wipe local disk between sessions. A checkpoint saved via
    torch.save() + dvc_add_checkpoint() + dvc_push_checkpoint() on Day 1
    exists ONLY on Google Drive (the DVC remote) once that session ends —
    NOT on local disk anymore. Day 2's fresh session clones the repo (git
    only — code and .dvc pointer files, never the actual checkpoint
    bytes) but has NO local outputs/checkpoints/*.pt files at all. Without
    this pull step, find_latest_checkpoint() correctly finds nothing —
    not because resume logic is broken, but because the checkpoint
    genuinely isn't on local disk yet. This was a real gap identified
    before any GPU hours were spent on a real multi-day run — exactly
    the right time to catch it.

    HOW THIS WORKS MECHANICALLY: `dvc pull` reads the .dvc pointer files
    that DID survive (they're tiny, git-tracked — see Phase 2 Step 3's
    dvc.yaml/.gitignore fixes) and uses them to fetch the actual checkpoint
    bytes from Google Drive, placing them at the exact path the pointer
    file specifies — which is exactly where find_latest_checkpoint() looks.

    Returns True if pull succeeded (or there was nothing to pull — both
    are non-error conditions), False only on a genuine pull failure.
    """
    if not checkpoint_dir.exists() or not any(checkpoint_dir.glob("*.dvc")):
        print(
            f"[checkpoint] no .dvc pointer files found in {checkpoint_dir} — "
            f"nothing to pull (this is normal for a brand new project with "
            f"no prior training sessions)."
        )
        return True

    try:
        subprocess.run(
            ["dvc", "pull", str(checkpoint_dir)],
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout_sec,
        )
        print(f"[checkpoint] dvc pull succeeded for {checkpoint_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(
            f"[checkpoint] WARNING — dvc pull FAILED for {checkpoint_dir}: "
            f"{e.stderr}\n"
            f"  Training will proceed from scratch since no local "
            f"checkpoint could be retrieved. If you expected to resume, "
            f"check your DVC remote connectivity before continuing."
        )
        return False
    except subprocess.TimeoutExpired:
        print(
            f"[checkpoint] WARNING — dvc pull TIMED OUT after {timeout_sec}s "
            f"for {checkpoint_dir} — proceeding from scratch."
        )
        return False


def dvc_add_checkpoint(checkpoint_path: Path) -> bool:
    """
    Run `dvc add` on a checkpoint file — stages it for DVC tracking
    (creates the .dvc pointer file) without pushing to the remote yet.

    Returns True on success, False on failure (logged, not raised) —
    a failed DVC add should warn loudly but never crash an active
    training run over what's fundamentally a bookkeeping operation.
    """
    try:
        subprocess.run(
            ["dvc", "add", str(checkpoint_path)],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        print(f"[checkpoint] dvc add succeeded: {checkpoint_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[checkpoint] WARNING — dvc add FAILED for {checkpoint_path}: " f"{e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        print(f"[checkpoint] WARNING — dvc add TIMED OUT for {checkpoint_path}")
        return False


def dvc_push_checkpoint(checkpoint_path: Path, timeout_sec: int = 300) -> bool:
    """
    Push a DVC-tracked checkpoint to the remote (Google Drive).

    WHY A LONGER TIMEOUT THAN dvc_add (300s vs 120s): pushing actually
    transfers the file over the network to Google Drive — for a ~220MB
    fp32 checkpoint on a typical Colab/Kaggle connection, this can
    legitimately take a few minutes. add is purely local hashing/bookkeeping
    and should be fast; push is the genuinely slow network operation.
    """
    try:
        subprocess.run(
            ["dvc", "push", str(checkpoint_path) + ".dvc"],
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout_sec,
        )
        print(f"[checkpoint] dvc push succeeded: {checkpoint_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[checkpoint] WARNING — dvc push FAILED for {checkpoint_path}: " f"{e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        print(
            f"[checkpoint] WARNING — dvc push TIMED OUT for {checkpoint_path} "
            f"after {timeout_sec}s — will retry on next checkpoint interval, "
            f"but check your network connection if this persists."
        )
        return False


def cleanup_old_checkpoints(
    checkpoint_dir: Path, keep_last_n: int = 3, confirmed_pushed: set | None = None
) -> None:
    """
    Delete local checkpoint files beyond the most recent N, keeping disk
    usage bounded during long training runs.

    WHY THIS IS SAFE EVEN THOUGH WE DELETE LOCAL FILES: by the time this
    runs, older checkpoints should already be DVC-pushed to Google Drive
    (the actual persistent copy) — deleting the LOCAL copy doesn't lose
    data, it just frees disk space on the ephemeral Kaggle/Colab session.

    WHY confirmed_pushed IS A SEPARATE PARAMETER FROM JUST CHECKING THE
    .dvc POINTER FILE EXISTS: an earlier version of this function only
    checked for the .dvc pointer's existence, which confirms `dvc add`
    succeeded — but NOT that `dvc push` actually completed. If push
    failed (e.g. a network blip) while add succeeded, the old logic
    would still delete the local file, losing the checkpoint entirely
    since it never made it to the remote either. The caller (train.py)
    now tracks which checkpoints were ACTUALLY confirmed pushed and
    passes that set in — cleanup only ever deletes checkpoints with a
    confirmed successful push, never merely a confirmed add.
    """
    checkpoints = sorted(checkpoint_dir.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)
    if len(checkpoints) <= keep_last_n:
        return

    confirmed_pushed = confirmed_pushed or set()
    to_delete = checkpoints[:-keep_last_n]
    for ckpt in to_delete:
        if str(ckpt) not in confirmed_pushed:
            print(
                f"[checkpoint] SKIPPING deletion of {ckpt} — push to DVC "
                f"remote was not confirmed successful. Deleting it now "
                f"would risk losing this checkpoint entirely if it isn't "
                f"actually on the remote."
            )
            continue
        ckpt.unlink()
        dvc_pointer = Path(str(ckpt) + ".dvc")
        if dvc_pointer.exists():
            dvc_pointer.unlink()
        print(f"[checkpoint] deleted local copy (confirmed pushed, safe): {ckpt}")
