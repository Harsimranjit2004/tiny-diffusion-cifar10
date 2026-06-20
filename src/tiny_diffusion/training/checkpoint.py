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


def cleanup_old_checkpoints(checkpoint_dir: Path, keep_last_n: int = 3) -> None:
    """
    Delete local checkpoint files beyond the most recent N, keeping disk
    usage bounded during long training runs.

    WHY THIS IS SAFE EVEN THOUGH WE DELETE LOCAL FILES: by the time this
    runs, older checkpoints should already be DVC-pushed to Google Drive
    (the actual persistent copy) — deleting the LOCAL copy doesn't lose
    data, it just frees disk space on the ephemeral Kaggle/Colab session.
    If a push failed for a given checkpoint, we deliberately skip deleting
    it (see the dvc_pushed_paths check) rather than silently losing data
    that exists nowhere else.
    """
    checkpoints = sorted(checkpoint_dir.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)
    if len(checkpoints) <= keep_last_n:
        return

    to_delete = checkpoints[:-keep_last_n]
    for ckpt in to_delete:
        dvc_pointer = Path(str(ckpt) + ".dvc")
        if not dvc_pointer.exists():
            print(
                f"[checkpoint] SKIPPING deletion of {ckpt} — no .dvc pointer "
                f"found, meaning it was never successfully dvc-added. "
                f"Deleting it now would lose this checkpoint entirely."
            )
            continue
        ckpt.unlink()
        print(f"[checkpoint] deleted local copy (DVC-tracked, safe): {ckpt}")
