"""
scripts/launch.py

INTERACTIVE VERTEX AI LAUNCHER

Run this instead of launch_vertex_training.py directly. It:
  1. Loads any values already set in .env
  2. Prompts interactively for anything that is still missing or blank
  3. Validates every GCP prerequisite (auth, APIs, GPU quota, bucket)
  4. Shows a full summary of what is about to run and asks for confirmation
  5. Submits the job

Why this exists: launch_vertex_training.py requires MLFLOW_TRACKING_PASSWORD
to be pre-set in your shell and silently reads PROJECT_ID / REGION / etc. from
hardcoded constants. This script surfaces every required value, checks GCP
state, and lets you correct anything before a real GPU job starts.

Usage:
  python scripts/launch.py
"""

import os
import subprocess
import sys
from getpass import getpass
from pathlib import Path

# ── Colour helpers (no deps) ─────────────────────────────────────────────────

RED = "\033[91m"
GRN = "\033[92m"
YLW = "\033[93m"
BLU = "\033[94m"
DIM = "\033[2m"
RST = "\033[0m"
BOLD = "\033[1m"


def ok(msg: str) -> None:
    print(f"  {GRN}✓{RST} {msg}")


def warn(msg: str) -> None:
    print(f"  {YLW}⚠{RST}  {msg}")


def err(msg: str) -> None:
    print(f"  {RED}✗{RST} {msg}")


def section(title: str) -> None:
    print(f"\n{BOLD}{BLU}── {title} {'─' * (54 - len(title))}{RST}")


def abort(msg: str) -> None:
    err(msg)
    print(f"\n{RED}Aborted — fix the issue above and re-run.{RST}\n")
    sys.exit(1)


# ── .env loader (manual parse — avoids python-dotenv import at launcher level) ─


def load_dotenv_into_os(env_path: Path) -> dict:
    """
    Parse .env into a dict and inject into os.environ for any key not already
    set by the shell. Returns the dict of values found in the file.
    """
    found: dict = {}
    if not env_path.exists():
        return found
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        found[key] = value
        if key not in os.environ:
            os.environ[key] = value
    return found


# ── Prompt helpers ────────────────────────────────────────────────────────────


def prompt(label: str, default: str = "", secret: bool = False) -> str:
    """
    Ask the user for a value. Shows the default (redacted if secret).
    Returns the default if the user just presses Enter.
    """
    display_default = ("***set***" if default else "not set") if secret else (default or "not set")
    suffix = f" [{display_default}]: " if default else ": "
    full_label = f"    {BOLD}{label}{RST}{suffix}"
    if secret:
        val = getpass(full_label)
    else:
        val = input(full_label).strip()
    return val if val else default


def prompt_bool(label: str, default: bool = True) -> bool:
    """Ask a yes/no question. Returns bool."""
    yn = "Y/n" if default else "y/N"
    raw = input(f"    {BOLD}{label}{RST} [{yn}]: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes")


# ── GCP checks ────────────────────────────────────────────────────────────────


def check_gcloud_adc() -> bool:
    """Return True if Application Default Credentials are valid."""
    r = subprocess.run(
        ["gcloud", "auth", "application-default", "print-access-token"],
        capture_output=True,
    )
    return r.returncode == 0


def get_gcloud_project() -> str:
    r = subprocess.run(
        ["gcloud", "config", "get", "project"],
        capture_output=True,
        text=True,
    )
    return r.stdout.strip() if r.returncode == 0 else ""


def check_api_enabled(project: str, api: str) -> bool:
    r = subprocess.run(
        ["gcloud", "services", "list", "--enabled", f"--project={project}", f"--filter=name:{api}"],
        capture_output=True,
        text=True,
    )
    return api in r.stdout


def check_t4_quota(project: str, region: str) -> float:
    """Return the NVIDIA_T4_GPUS quota limit for the given region, or 0."""
    r = subprocess.run(
        [
            "gcloud",
            "compute",
            "regions",
            "describe",
            region,
            f"--project={project}",
            "--format=value(quotas)",
        ],
        capture_output=True,
        text=True,
    )
    for part in r.stdout.split(";"):
        if (
            "NVIDIA_T4_GPUS" in part
            and "VWS" not in part
            and "PREEMPTIBLE" not in part
            and "COMMITTED" not in part
        ):
            try:
                return float(part.split("'limit':")[1].split(",")[0].strip())
            except (IndexError, ValueError):
                pass
    return 0.0


def check_bucket_exists(project: str, bucket_name: str) -> bool:
    r = subprocess.run(
        ["gcloud", "storage", "buckets", "describe", f"gs://{bucket_name}", f"--project={project}"],
        capture_output=True,
    )
    return r.returncode == 0


def create_bucket(project: str, bucket_name: str, region: str) -> bool:
    r = subprocess.run(
        [
            "gcloud",
            "storage",
            "buckets",
            "create",
            f"gs://{bucket_name}",
            f"--project={project}",
            f"--location={region}",
        ],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    print(f"\n{BOLD}{'═' * 60}{RST}")
    print(f"{BOLD}  Vertex AI Training Launcher{RST}")
    print(f"{BOLD}{'═' * 60}{RST}")

    # ── Step 1: load .env ────────────────────────────────────────────────────
    section("Load .env")
    env_path = Path(__file__).parent.parent / ".env"
    dotenv_vals = load_dotenv_into_os(env_path)
    if dotenv_vals:
        ok(f"Loaded {len(dotenv_vals)} values from {env_path.name}")
    else:
        warn(f".env not found at {env_path} — will prompt for all values")

    # ── Step 2: collect required credentials ─────────────────────────────────
    section("Credentials")

    mlflow_password = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")
    if not mlflow_password:
        print(f"  {YLW}MLFLOW_TRACKING_PASSWORD not set.{RST}")
        print(f"  {DIM}Get your token: DagsHub → Settings → Access Tokens{RST}")
        mlflow_password = prompt("DagsHub token (MLFLOW_TRACKING_PASSWORD)", secret=True)
        if not mlflow_password:
            abort("DagsHub token is required — MLflow tracking will fail without it.")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_password
    else:
        ok("MLFLOW_TRACKING_PASSWORD  (already set)")

    dagshub_token = os.environ.get("DAGSHUB_TOKEN", mlflow_password)
    os.environ["DAGSHUB_TOKEN"] = dagshub_token  # keep in sync
    ok("DAGSHUB_TOKEN             (same as MLflow password)")

    dagshub_username = os.environ.get("DAGSHUB_USERNAME", "")
    if not dagshub_username:
        dagshub_username = prompt("DagsHub username (DAGSHUB_USERNAME)")
        if not dagshub_username:
            abort("DagsHub username is required for the git clone inside the container.")
        os.environ["DAGSHUB_USERNAME"] = dagshub_username
    else:
        ok(f"DAGSHUB_USERNAME          → {dagshub_username}")

    mlflow_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        f"https://dagshub.com/{dagshub_username}/tiny-diffusion-cifar10.mlflow",
    )
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri
    ok(f"MLFLOW_TRACKING_URI       → {mlflow_uri}")

    # ── Step 3: GCP config ───────────────────────────────────────────────────
    section("GCP Configuration")

    # Import here so the script can still print errors before the venv check
    try:
        from launch_vertex_training import (  # noqa: PLC0415
            ACCELERATOR_COUNT,
            ACCELERATOR_TYPE,
            BUCKET_NAME,
            CONTAINER_URI,
            MACHINE_TYPE,
            PROJECT_ID,
            REGION,
            VALIDATION_RUN,
        )
    except ImportError:
        # Fallback: read constants directly if this script is run from the repo root
        PROJECT_ID = "tiny-diffusion-training"
        REGION = "us-central1"
        BUCKET_NAME = f"{PROJECT_ID}-checkpoints"
        MACHINE_TYPE = "n1-standard-4"
        ACCELERATOR_TYPE = "NVIDIA_TESLA_T4"
        ACCELERATOR_COUNT = 1
        CONTAINER_URI = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-3.py310:latest"
        VALIDATION_RUN = True

    ok(f"Project  → {PROJECT_ID}")
    ok(f"Region   → {REGION}")
    ok(f"Machine  → {MACHINE_TYPE} + {ACCELERATOR_COUNT}× {ACCELERATOR_TYPE}")
    ok(f"Bucket   → gs://{BUCKET_NAME}")
    ok(f"Container→ {CONTAINER_URI}")

    # ── Step 4: GCP prerequisite checks ─────────────────────────────────────
    section("GCP Prerequisite Checks")

    # 4a. gcloud ADC
    if check_gcloud_adc():
        ok("Application Default Credentials are valid")
    else:
        err("gcloud ADC not set up — run:")
        print(f"        {DIM}gcloud auth application-default login{RST}")
        if not prompt_bool(
            "Open a shell command for you to run? (you'll need to re-run this script after)",
            default=False,
        ):
            abort("gcloud ADC required before submitting a job.")
        else:
            subprocess.run(["gcloud", "auth", "application-default", "login"])
            if not check_gcloud_adc():
                abort("ADC still not valid after login attempt.")
            ok("Application Default Credentials now valid")

    # 4b. active gcloud project
    active_project = get_gcloud_project()
    if active_project == PROJECT_ID:
        ok(f"Active gcloud project matches → {active_project}")
    elif active_project:
        warn(f"Active gcloud project is '{active_project}', expected '{PROJECT_ID}'")
        warn("Continuing — the Python SDK uses PROJECT_ID from the script, not gcloud config.")
    else:
        warn("Could not read active gcloud project — continuing anyway.")

    # 4c. APIs
    for api in ["aiplatform.googleapis.com", "compute.googleapis.com"]:
        if check_api_enabled(PROJECT_ID, api):
            ok(f"{api} enabled")
        else:
            err(f"{api} is NOT enabled")
            print(f"        {DIM}Run: gcloud services enable {api} --project={PROJECT_ID}{RST}")
            abort(f"Enable {api} and re-run.")

    # 4d. GPU quota
    t4_quota = check_t4_quota(PROJECT_ID, REGION)
    if t4_quota >= 1.0:
        ok(f"NVIDIA_T4_GPUS quota in {REGION} → {t4_quota:.0f} (sufficient)")
    else:
        abort(
            f"NVIDIA_T4_GPUS quota in {REGION} is {t4_quota} — need at least 1.\n"
            "    Request a quota increase in the GCP console:\n"
            "    IAM & Admin → Quotas → filter NVIDIA_T4_GPUS → Edit Quotas."
        )

    # 4e. GCS bucket
    if check_bucket_exists(PROJECT_ID, BUCKET_NAME):
        ok(f"GCS bucket gs://{BUCKET_NAME} exists")
    else:
        warn(f"GCS bucket gs://{BUCKET_NAME} does not exist — creating it...")
        if create_bucket(PROJECT_ID, BUCKET_NAME, REGION):
            ok(f"Created gs://{BUCKET_NAME} in {REGION}")
        else:
            abort(
                f"Failed to create gs://{BUCKET_NAME}.\n"
                "    Check that your account has Storage Admin on project "
                f"'{PROJECT_ID}'."
            )

    # ── Step 5: job configuration ────────────────────────────────────────────
    section("Job Configuration")

    validation_run = VALIDATION_RUN
    print(f"  VALIDATION_RUN is currently {BOLD}{'True' if validation_run else 'False'}{RST}")
    if validation_run:
        ok("Will run 1 epoch to validate the full pipeline end-to-end")
        print(
            f"  {DIM}Set VALIDATION_RUN = False in launch_vertex_training.py"
            f" for the real 200-epoch run.{RST}"
        )
    else:
        warn("VALIDATION_RUN = False → this will submit the FULL ~18hr training run")
        if not prompt_bool("Are you sure you want to start the full run?", default=False):
            abort("Aborted by user — set VALIDATION_RUN = True for a safe smoke test first.")

    # ── Step 6: final summary + confirm ─────────────────────────────────────
    section("Summary")

    summary_rows = [
        ("Project", PROJECT_ID),
        ("Region", REGION),
        ("Machine", f"{MACHINE_TYPE} + {ACCELERATOR_COUNT}× {ACCELERATOR_TYPE}"),
        ("Container", CONTAINER_URI),
        ("Staging bucket", f"gs://{BUCKET_NAME}"),
        ("Checkpoint output", f"gs://{BUCKET_NAME}/output/"),
        ("MLflow URI", mlflow_uri),
        ("MLflow username", dagshub_username),
        ("DagsHub username", dagshub_username),
        ("Validation run", str(validation_run)),
    ]
    for label, value in summary_rows:
        print(f"  {DIM}{label:<22}{RST}{value}")

    print()
    if not prompt_bool("Submit job now?", default=True):
        print(f"\n{YLW}Aborted by user.{RST}\n")
        sys.exit(0)

    # ── Step 7: launch ───────────────────────────────────────────────────────
    section("Submitting Job")

    launch_script = Path(__file__).parent / "launch_vertex_training.py"
    env = {**os.environ}  # everything we've built up, including the credentials

    print(f"  Running: python {launch_script.name}\n")
    result = subprocess.run(
        [sys.executable, str(launch_script)],
        env=env,
        cwd=str(launch_script.parent.parent),  # repo root, so script_path= resolves correctly
    )

    if result.returncode == 0:
        print(f"\n{GRN}{BOLD}Job submitted successfully.{RST}")
        print(
            f"  Monitor at: https://console.cloud.google.com/vertex-ai/"
            f"training/custom-jobs?project={PROJECT_ID}\n"
        )
    else:
        abort(f"launch_vertex_training.py exited with code {result.returncode}")


if __name__ == "__main__":
    main()
