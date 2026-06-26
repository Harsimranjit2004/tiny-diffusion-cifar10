"""
scripts/launch_vertex_training.py

PHASE 3 — VERTEX AI CUSTOM TRAINING JOB LAUNCHER

WHAT THIS SCRIPT DOES: submits our existing train.py to run as a managed
Vertex AI Custom Training Job, using the SAME training code that already
runs on Kaggle/Colab/local — only the checkpoint persistence mechanism
differs (see get_checkpoint_dir() in train.py), handled automatically by
environment detection.

WHY VERTEX AI, AFTER EVERYTHING ELSE WE TRIED:
  1. Kaggle + DVC/Google Drive: Google's OAuth consent flow requires a
     human clicking "Allow" in a browser — fundamentally incompatible
     with Kaggle's non-interactive "Save & Run All" batch execution.
     Confirmed via direct testing: the push call hung for the full 300s
     timeout waiting on a redirect that could never complete.
  2. AWS SageMaker: mechanically worked end-to-end (IAM role, S3 bucket,
     job submission all succeeded) but AWS denied our GPU training-job
     quota request outright, citing insufficient account usage history.
     Confirmed via direct quota checks that EVERY GPU instance type
     (g4dn, g5) shows 0 training-job quota on this account — a blanket
     new-account restriction, not specific to the instance type.
  3. Vertex AI: this account's $415 credit balance and existing $1 of
     real usage (unlike AWS's brand-new, zero-usage account) carried
     over real GPU quota automatically — confirmed via direct quota
     check showing NVIDIA_T4_GPUS limit=1.0 in us-central1, immediately
     usable with no quota request needed at all.

WHY NO CLI-ARGUMENT BRIDGE SCRIPT IS NEEDED (unlike SageMaker):
  SageMaker's hyperparameters dict gets converted into `--key value` CLI
  args by the SDK, incompatible with Hydra's `key=value` override syntax
  — we had to write sagemaker_entry.py specifically to translate between
  the two. Vertex AI's CustomTrainingJob.run(args=[...]) passes the args
  list through to the entry script completely unmodified — we can hand
  it Hydra-style "key=value" strings directly, no translation needed.

PREREQUISITES (one-time GCP setup, do this before running this script):
  1. A Google Cloud project with billing enabled (already have:
     tiny-diffusion-training, linked to the account with $415 credit)
  2. Vertex AI API and Compute Engine API enabled (already done)
  3. gcloud CLI installed and authenticated (already done: `gcloud init`)
  4. Confirmed GPU quota (already done: NVIDIA_T4_GPUS limit=1.0 in
     us-central1 — verified via `gcloud compute regions describe`)

USAGE:
  python scripts/launch_vertex_training.py
"""

import os

from google.cloud import aiplatform, storage

# ── Configuration ────────────────────────────────────────────────────────

# WHY THIS FLAG EXISTS, AND WHY IT'S DEFINED FIRST: this is genuinely new,
# never-tested-against-real-GCP code — before committing the full ~18hr
# run, we validate the entire pipeline end-to-end on a single short
# epoch first. This MUST be defined before any other constant that
# reads it. Set to False once the validation run succeeds and you're
# ready for the real baseline training.
VALIDATION_RUN = True

PROJECT_ID = "tiny-diffusion-training"
REGION = "us-central1"

# WHY n1-standard-4 + NVIDIA_TESLA_T4 SPECIFICALLY: matches our confirmed
# quota exactly (NVIDIA_T4_GPUS limit=1.0 in us-central1) and is
# comparable to what Kaggle/Colab gave us for free — keeps our existing
# batch_size/memory assumptions valid without re-tuning anything.
MACHINE_TYPE = "n1-standard-4"
ACCELERATOR_TYPE = "NVIDIA_TESLA_T4"
ACCELERATOR_COUNT = 1

# WHY A PRE-BUILT PYTORCH CONTAINER, NOT A CUSTOM ONE: Google publishes
# ready-to-use containers with PyTorch + CUDA already correctly linked
# (the same "don't fight the platform's CUDA setup" reasoning from
# Phase 2 Step 2's package-installation decisions) — building our own
# container image adds complexity with no benefit here, since none of
# our dependencies need anything unusual.
CONTAINER_URI = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-3.py310:latest"

# WHY THIS BUCKET NAMING: Vertex AI requires a "staging bucket" to
# upload your source code and a separate (or same) bucket for
# checkpoint output. We use one bucket for both, scoped under this
# project, created automatically on first use if it doesn't exist.
BUCKET_NAME = f"{PROJECT_ID}-checkpoints"
STAGING_BUCKET = f"gs://{BUCKET_NAME}"
CHECKPOINT_GCS_URI = f"gs://{BUCKET_NAME}/checkpoints/"
MLFLOW_URI = "https://dagshub.com/Harsimranjit2004/tiny-diffusion-cifar10.mlflow"


def ensure_bucket_exists(bucket_name: str, region: str) -> None:
    """
    Create the GCS bucket if it doesn't already exist.

    WHY THIS FUNCTION EXISTS: unlike SageMaker, which auto-provisions a
    default bucket per AWS account/region the first time you use it,
    Vertex AI does NOT create a bucket for you automatically — our
    launch script originally assumed the bucket already existed, which
    failed with a 404 "specified bucket does not exist" error on the
    very first real run against actual GCP infrastructure. This function
    closes that gap so the script is self-sufficient on a fresh project.
    """
    client = storage.Client(project=PROJECT_ID)
    if client.lookup_bucket(bucket_name) is not None:
        print(f"Bucket gs://{bucket_name} already exists.")
        return

    print(f"Bucket gs://{bucket_name} does not exist — creating it in {region}...")
    client.create_bucket(bucket_name, location=region)
    print(f"Created gs://{bucket_name}")


def main() -> None:
    mlflow_password = os.environ.get("MLFLOW_TRACKING_PASSWORD")
    if not mlflow_password:
        raise EnvironmentError(
            "MLFLOW_TRACKING_PASSWORD not set in your local environment. "
            "Run this script as: "
            "MLFLOW_TRACKING_PASSWORD=<your-dagshub-token> python scripts/launch_vertex_training.py"
        )

    dagshub_token = os.environ.get("DAGSHUB_TOKEN", mlflow_password)
    dagshub_username = os.environ.get("DAGSHUB_USERNAME", "Harsimranjit2004")

    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=STAGING_BUCKET,
    )

    ensure_bucket_exists(BUCKET_NAME, REGION)

    print(f"Project: {PROJECT_ID}")
    print(f"Region: {REGION}")
    print(f"Machine: {MACHINE_TYPE} + {ACCELERATOR_COUNT}x {ACCELERATOR_TYPE}")
    print(f"Checkpoints will sync to: {CHECKPOINT_GCS_URI}")
    print(f"Validation run: {VALIDATION_RUN}")

    job = aiplatform.CustomTrainingJob(
        display_name="tiny-diffusion-baseline",
        script_path="scripts/train.py",
        container_uri=CONTAINER_URI,
        # WHY requirements LISTED HERE TOO, NOT JUST IN requirements.txt:
        # Vertex AI's CustomTrainingJob does NOT automatically read a
        # requirements.txt from the project — it only installs packages
        # explicitly listed in this `requirements` argument on top of
        # whatever the base container_uri image already provides (which
        # already includes torch/torchvision/CUDA, matching the same
        # "don't fight the platform's pre-installed GPU-linked torch"
        # principle from Phase 2 Step 2). We list everything else our
        # code needs beyond that base image.
        #
        # WHY "." IS IN THIS LIST (THE FIX FOR THE REAL FIRST-RUN BUG):
        # CustomTrainingJob's script_path stages ONLY that single file
        # into the container — it does NOT package and install the rest
        # of the project the way SageMaker's source_dir parameter does.
        # The first real run against actual GCP infrastructure failed
        # with `ModuleNotFoundError: No module named 'tiny_diffusion'`
        # because train.py's `from tiny_diffusion.training.train import
        # train` had nothing to import — the package was never installed
        # in the container. `requirements` is passed straight to pip,
        # and pip accepts a local directory path as a requirement
        # specifier (this is the documented mechanism for exactly this
        # case — see pyproject.toml at the project root, which is what
        # makes "." resolve to an installable package here, the same
        # pyproject.toml that makes `pip install -e .` work locally).
        requirements=[
            "einops==0.8.0",
            "mlflow==2.14.1",
            "datasets==2.20.0",
            "Pillow",  # datasets returns PIL images; base container may vary
            "python-dotenv==1.0.1",
            "hydra-core==1.3.2",
            "omegaconf==2.3.0",
            "nvidia-ml-py==12.560.30",
        ],
    )

    # ── Hydra overrides passed as plain CLI args ─────────────────────────
    # WHY THIS WORKS WITHOUT A TRANSLATION BRIDGE (unlike SageMaker):
    # CustomTrainingJob.run()'s args parameter is passed through to the
    # entry script's sys.argv completely unmodified — Hydra's
    # @hydra.main decorator parses these directly with no incompatible
    # intermediate format to translate, unlike SageMaker's --key value
    # hyperparameters dict which required scripts/sagemaker_entry.py.
    job_args = []
    if VALIDATION_RUN:
        job_args.append("experiment.training.num_epochs=1")
        job_args.append("experiment.training.checkpoint_every_n_steps=50")
    else:
        job_args.append("experiment.training.checkpoint_every_n_steps=1000")

    print(f"\nJob args: {job_args}")
    print("\nSubmitting training job to Vertex AI...")

    job.run(
        machine_type=MACHINE_TYPE,
        accelerator_type=ACCELERATOR_TYPE,
        accelerator_count=ACCELERATOR_COUNT,
        replica_count=1,
        args=job_args,
        environment_variables={
            "MLFLOW_TRACKING_URI": MLFLOW_URI,
            "MLFLOW_TRACKING_USERNAME": "Harsimranjit2004",
            "MLFLOW_TRACKING_PASSWORD": mlflow_password,
            "DAGSHUB_USERNAME": dagshub_username,
            "DAGSHUB_TOKEN": dagshub_token,
            # WHY AIP_CHECKPOINT_DIR IS NOT SET HERE MANUALLY: Vertex AI
            # sets this automatically for every CustomTrainingJob based
            # on base_output_dir — we don't need to (and shouldn't)
            # override it ourselves. get_checkpoint_dir() in train.py
            # reads whatever Vertex AI actually provides.
        },
        base_output_dir=f"gs://{BUCKET_NAME}/output/",
        sync=False,
        # sync=False: returns immediately after submitting the job,
        # rather than blocking your local terminal for the full ~18
        # hours. Check job status via the Cloud Console or the job
        # object's .state property.
    )

    # WHY THIS CALL IS REQUIRED BEFORE READING job.display_name OR
    # job.resource_name: with sync=False, job.run() returns control to
    # this script immediately, before Vertex AI's API has actually
    # finished CREATING the underlying server-side job resource. Reading
    # any property on the job object before that creation completes
    # raises "RuntimeError: CustomTrainingJob resource has not been
    # created" — exactly the error we hit on the first real attempt
    # against actual GCP infrastructure. wait_for_resource_creation()
    # blocks only until the job EXISTS as a resource (fast — seconds),
    # NOT until training finishes (which would defeat the purpose of
    # sync=False entirely). This is the documented fix for this exact
    # "construct and run" pattern's known timing gap.
    job.wait_for_resource_creation()

    print(f"\nJob submitted: {job.display_name}")
    print(f"Resource name: {job.resource_name}")
    print(
        "Monitor progress at: "
        f"https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}"
    )


if __name__ == "__main__":
    main()
