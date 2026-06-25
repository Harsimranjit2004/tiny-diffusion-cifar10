"""
scripts/train.py

PHASE 3 — REAL TRAINING ENTRY POINT

Phase 2 Step 4 used this file purely to validate Hydra config composition
(printing the resolved config, no actual training). Now that Phase 3's
real training loop exists in src/tiny_diffusion/training/train.py, this
script's job is to load .env, load the config via Hydra, and hand off
to the real training loop.

WHY load_dotenv() IS CALLED HERE, EXPLICITLY, BEFORE ANYTHING ELSE:
  tracking.py's init_tracking() reads MLFLOW_TRACKING_URI directly from
  os.environ — it does NOT load .env itself (kept that way so tracking.py
  has no opinion about WHERE env vars come from — could be .env locally,
  could be Kaggle/Colab secrets, could be CI environment variables). This
  bit us during the first real Kaggle run: we wrote a .env file but never
  actually loaded it into the process environment before running this
  script directly (as opposed to the Phase 2 Step 2 smoke test, which
  called load_dotenv() manually in an inline Python snippet). Calling it
  here, once, at the top of the actual entry point, fixes this permanently
  for every future run — local, Kaggle, or Colab — without requiring
  whoever runs this script to remember an extra manual step.

WHY THIS SCRIPT CLONES THE REPO AT RUNTIME, BEFORE ANY tiny_diffusion
IMPORT — THE FULL STORY OF HOW WE GOT HERE:
  The first real run against actual Vertex AI infrastructure failed
  with `ModuleNotFoundError: No module named 'tiny_diffusion'`.
  CustomTrainingJob's script_path parameter stages ONLY that single
  file into the container — confirmed via Vertex AI SDK GitHub issue
  #1093, which states plainly that "a CustomTrainingJob method reads a
  training script from a local directory source only" — meaning it
  bundles JUST the script into the container, not the surrounding
  project tree, unlike SageMaker's source_dir parameter.

  We tried two fixes that turned out to be wrong, in order:
  1. Adding "." to launch_vertex_training.py's `requirements` list,
     hoping pip would install our project locally. This actively
     CRASHED the job: Vertex AI's internal packager
     (_TrainingScriptPythonPackager, confirmed via its GitHub PR
     history) generates its OWN wrapper setup.py and feeds our
     `requirements` list directly into that setup.py's install_requires
     field — which only accepts real PyPI-style specifiers, never a
     local path. Removed from launch_vertex_training.py entirely.
  2. Inserting src/ into sys.path, assuming the script's surrounding
     directory tree got staged alongside it. This was never actually
     verified against real infrastructure (would have cost real GPU
     minutes to test) and, given finding #1093 above, is very likely
     ALSO broken — if only the single script gets bundled, src/ never
     exists in the container's filesystem at all, making the sys.path
     insertion point at a path that simply isn't there.

  THE ACTUAL FIX: clone the project's own git repo (DagsHub) at the
  very start of this script's execution, then import from that fresh
  clone. This sidesteps Vertex AI's local-packaging ambiguity entirely
  — it doesn't matter what gets staged alongside this single script,
  since we fetch everything we need ourselves, the same way Kaggle/
  Colab sessions already do (recall: every other platform in this
  project clones the repo at session start; Vertex AI is the only one
  where the framework was supposed to package local code FOR us, and
  that assumption turned out to be false). This is unambiguous because
  it depends only on git and network access, which every Vertex AI
  training container has by default — no uncertain SDK packaging
  behavior involved.

  WHY THIS IS GATED ON is_running_on_vertex_ai(): on Kaggle/Colab/
  local, the project is ALREADY cloned/present — re-cloning would be
  redundant at best, and could overwrite local uncommitted work at
  worst. This clone-at-runtime behavior is scoped specifically to the
  one platform where it's actually needed.

USAGE:
  python scripts/train.py
      -> trains with configs/experiment/baseline.yaml

  python scripts/train.py experiment.training.lr=0.0001
      -> same experiment, overridden learning rate
      NOTE: the `experiment.` prefix is REQUIRED — configs/experiment/
      baseline.yaml composes model/training/data/schedule UNDER the
      `experiment` key (see configs/config.yaml's defaults: list), so
      every override path must mirror that nesting exactly. Forgetting
      the prefix produces Hydra's "Key 'training' is not in struct" error
      — a real mistake we hit during Phase 3's first Kaggle run.

  python scripts/train.py --multirun experiment.training.lr=1e-4,2e-4,5e-4
      -> three separate training runs, one per LR (Phase 5's mechanism
         for the quantization ablation study uses this same multirun
         pattern, just sweeping quantization method instead of LR)
"""

import os
import subprocess
import sys
from pathlib import Path


def _is_running_on_vertex_ai() -> bool:
    """
    Local copy of the same detection logic used in train.py's
    is_running_on_vertex_ai() — duplicated here (not imported) because
    this check must run BEFORE we can import anything from
    tiny_diffusion at all, which is precisely the chicken-and-egg
    problem this whole clone-at-runtime mechanism exists to solve.
    """
    return "CLOUD_ML_PROJECT_ID" in os.environ


def _ensure_project_cloned() -> Path:
    """
    On Vertex AI specifically, clone the project repo fresh so
    tiny_diffusion is genuinely importable, sidestepping the local-
    packaging ambiguity documented in this file's module docstring.

    Returns the path to clone into, which gets added to sys.path.

    WHY THE CLONE URL EMBEDS A TOKEN: the first real attempt against
    actual Vertex AI infrastructure failed with `fatal: could not read
    Username for 'https://dagshub.com': No such device or address` —
    git's way of saying it needs interactive credentials and there's no
    human present in this container to provide them. This is the exact
    same class of problem as Google Drive's OAuth flow being
    incompatible with Kaggle's non-interactive batch mode, just for git
    HTTPS auth instead. The fix is the same pattern we used successfully
    for Kaggle's own clone step earlier in this project: embed
    username:token directly in the URL so git authenticates
    non-interactively, no prompt involved.

    WHY THESE CREDENTIALS COME FROM ENVIRONMENT VARIABLES, NOT HARDCODED:
    launch_vertex_training.py passes DAGSHUB_USERNAME and
    DAGSHUB_TOKEN into the job's environment_variables — read from your
    LOCAL shell environment when you submit the job, never committed to
    this file or to git, for the same reason MLFLOW_TRACKING_PASSWORD
    has never been hardcoded anywhere in this project.
    """
    dagshub_username = os.environ.get("DAGSHUB_USERNAME")
    dagshub_token = os.environ.get("DAGSHUB_TOKEN")

    default_repo = "dagshub.com/Harsimranjit2004/tiny-diffusion-cifar10.git"
    if dagshub_username and dagshub_token:
        repo_url = os.environ.get(
            "TRAINING_REPO_URL",
            f"https://{dagshub_username}:{dagshub_token}@{default_repo}",
        )
    else:
        # WHY WE DON'T RAISE IMMEDIATELY HERE: fall through to the
        # unauthenticated URL and let git's own error message (the one
        # we already saw and diagnosed) explain the problem clearly if
        # credentials genuinely are missing — clearer than a generic
        # "credentials not set" error from this function that doesn't
        # show the actual git failure underneath it.
        print(
            "[train.py] WARNING — DAGSHUB_USERNAME / DAGSHUB_TOKEN not "
            "set in this container's environment. Clone will likely "
            "fail with a git authentication error if the repo is "
            "private."
        )
        repo_url = os.environ.get("TRAINING_REPO_URL", f"https://{default_repo}")

    clone_dir = Path("/tmp/tiny-diffusion-cifar10")

    if clone_dir.exists():
        print(
            f"[train.py] {clone_dir} already exists, skipping clone "
            f"(likely a resumed/retried job in the same container)."
        )
        return clone_dir / "src"

    # WHY THE PRINTED MESSAGE REDACTS THE URL: repo_url contains the
    # token when credentials are set — printing it directly would leak
    # the token into Vertex AI's job logs, which are not meant to be
    # treated as a secrets store. We print a redacted form instead.
    redacted_url = repo_url
    if dagshub_token:
        redacted_url = repo_url.replace(dagshub_token, "***TOKEN***")
    print(
        f"[train.py] Running on Vertex AI — cloning {redacted_url} "
        f"into {clone_dir} so tiny_diffusion is importable..."
    )
    result = subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(clone_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # WHY stderr IS NOT PRINTED RAW HERE: git's own error output can
        # echo back the URL it tried to clone (including the embedded
        # token) in some failure modes — redact defensively before this
        # ever reaches Vertex AI's logs, same reasoning as the redacted
        # print above.
        safe_stderr = result.stderr
        if dagshub_token:
            safe_stderr = safe_stderr.replace(dagshub_token, "***TOKEN***")
        raise RuntimeError(
            f"git clone failed (exit {result.returncode}): {safe_stderr}\n"
            "Set TRAINING_REPO_URL env var if the default URL above is "
            "wrong, or if the repo requires authentication this "
            "container doesn't have."
        )
    print("[train.py] Clone succeeded.")
    return clone_dir / "src"


if _is_running_on_vertex_ai():
    _src_path = _ensure_project_cloned()
    if str(_src_path) not in sys.path:
        sys.path.insert(0, str(_src_path))

# WHY noqa: E402 ON EVERY IMPORT BELOW THIS POINT: these imports must
# come AFTER the conditional sys.path modification above by design —
# tiny_diffusion may not be importable until that block runs on Vertex
# AI specifically (see this file's module docstring for the full
# ModuleNotFoundError story). flake8's "module level import not at top
# of file" check is correct that this violates the usual convention,
# but moving these imports earlier would defeat the entire purpose of
# this fix, so we suppress the warning explicitly rather than silence
# it project-wide or restructure away from a fix we've already
# verified is necessary.
import hydra  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from tiny_diffusion.training.train import train  # noqa: E402

# Load .env BEFORE hydra.main() runs — environment variables need to be
# in os.environ before tracking.init_tracking() is called inside train().
# WHY load_dotenv() IS SAFE TO CALL UNCONDITIONALLY, EVEN IF .env DOESN'T
# EXIST: python-dotenv's load_dotenv() silently no-ops if the file is
# missing rather than raising — so this is safe to leave in even in CI
# environments (Phase 7) where secrets come from actual environment
# variables set by the CI runner, not a local .env file.
load_dotenv()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
