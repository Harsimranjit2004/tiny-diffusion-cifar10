"""
=============================================================================
PHASE 2 — STEP 1: PROJECT STRUCTURE AND ENVIRONMENT SETUP
Tiny Conditional Diffusion on CIFAR-10
=============================================================================

WHAT THIS STEP COVERS:
  1. Why this exact folder layout (reasoning for every directory)
  2. The environment problem: Kaggle vs Colab vs local — what's persistent,
     what's ephemeral, and why that dictates our structure
  3. requirements.txt vs conda vs Docker — decision with reasoning
  4. Complete seed management (Python, NumPy, PyTorch, CUDA)
  5. A setup script that works identically on Kaggle and Colab

RUN THIS FILE TO:
  - Create the full folder structure
  - Verify environment (GPU, package versions)
  - Test seed reproducibility end to end

=============================================================================
"""

# =============================================================================
# STEP 1.1 — THE ENVIRONMENT PROBLEM: WHY STRUCTURE MATTERS HERE
# =============================================================================
#
# Before deciding folder layout, we need to understand the constraint that
# shapes everything: Kaggle and Colab are EPHEMERAL.
#
# WHAT "EPHEMERAL" MEANS FOR US:
#   - Colab: disk wiped when runtime disconnects (idle timeout ~90 min,
#            hard session limit ~12 hrs). /content is gone after disconnect.
#   - Kaggle: disk wiped after each session (~12 hr GPU quota per week,
#             session ends → /kaggle/working is gone unless explicitly saved
#             as a Kaggle Dataset or Output).
#
# CONSEQUENCE FOR OUR DESIGN:
#   Anything we care about keeping must be pushed to PERSISTENT, EXTERNAL
#   storage during the run, not just saved to local disk and dealt with later.
#   This is why Phase 2 sets up DVC remote (Google Drive/DagsHub) and MLflow
#   remote tracking BEFORE training starts — if we bolt this on after a run,
#   we lose everything when the session ends.
#
# OUR FOLDER STRUCTURE THEREFORE SEPARATES:
#   - /workspace        — ephemeral, exists only during the active session
#                          (matches /kaggle/working or /content on each platform)
#   - DVC remote         — persistent, lives on Google Drive regardless of
#                          which platform/session created the artifact
#   - MLflow remote       — persistent, same reasoning, hosted on DagsHub free tier
#   - GitHub repo         — persistent, holds CODE only (never data, never
#                          checkpoints — those go through DVC, not git)


# =============================================================================
# STEP 1.2 — FOLDER STRUCTURE (the actual layout we build)
# =============================================================================
#
# tiny-diffusion-cifar10/
# │
# ├── configs/                      ← Hydra configs (Phase 2, Step 4)
# │   ├── model/
# │   ├── training/
# │   ├── data/
# │   └── experiment/
# │
# ├── src/                          ← all source code, importable as a package
# │   └── tiny_diffusion/
# │       ├── __init__.py
# │       ├── models/               ← architecture (Phase 1 code lives here)
# │       │   ├── __init__.py
# │       │   ├── unet.py
# │       │   ├── blocks.py         ← ResBlock, SelfAttentionBlock, etc.
# │       │   ├── embeddings.py     ← time + class embeddings
# │       │   └── ema.py
# │       ├── diffusion/            ← math: schedules, samplers
# │       │   ├── __init__.py
# │       │   ├── schedule.py       ← CosineNoiseSchedule
# │       │   ├── ddpm_sampler.py
# │       │   └── ddim_sampler.py
# │       ├── data/                 ← CIFAR-10 loading, augmentation
# │       │   ├── __init__.py
# │       │   └── cifar10.py
# │       ├── training/             ← training loop, checkpointing
# │       │   ├── __init__.py
# │       │   ├── train.py
# │       │   └── checkpoint.py
# │       ├── evaluation/           ← FID, sampler comparison
# │       │   ├── __init__.py
# │       │   └── fid.py
# │       ├── quantization/         ← Phase 5 — PTQ, QAT pipelines
# │       │   └── __init__.py
# │       └── utils/
# │           ├── __init__.py
# │           ├── seed.py           ← THIS FILE's seed management code lives here
# │           └── logging_utils.py
# │
# ├── tests/                        ← pytest unit tests (Phase 2, Step 5)
# │   ├── test_schedule.py
# │   ├── test_blocks.py
# │   ├── test_unet_shapes.py
# │   └── test_samplers.py
# │
# ├── scripts/                      ← standalone CLI entry points
# │   ├── train.py                  ← thin wrapper: Hydra config → training.train()
# │   ├── sample.py
# │   ├── evaluate.py
# │   └── setup_env.py              ← THIS is what we build right now
# │
# ├── data/                         ← DVC-tracked, gitignored, populated on demand
# │   ├── raw/                      ← original CIFAR-10 download
# │   └── processed/                ← preprocessed tensors (optional cache)
# │
# ├── outputs/                      ← DVC-tracked, gitignored
# │   ├── checkpoints/
# │   ├── samples/                  ← generated sample grids
# │   └── quantized_models/
# │
# ├── .dvc/                         ← DVC internal config (Phase 2, Step 3)
# ├── dvc.yaml                      ← DVC pipeline definition
# ├── params.yaml                   ← DVC-tracked hyperparameters (mirrors Hydra)
# │
# ├── .github/workflows/            ← CI/CD (Phase 7)
# │   └── ci.yml
# │
# ├── .pre-commit-config.yaml       ← black, isort, flake8, mypy (Phase 2, Step 5)
# ├── pyproject.toml                ← tool configs (black, isort, mypy settings)
# ├── requirements.txt              ← pinned dependencies
# ├── .gitignore
# ├── .env.example                  ← template for secrets (MLflow URI, etc.)
# └── README.md
#
# WHY src/tiny_diffusion/ AS A PACKAGE (not flat scripts):
#   Making this an installable package (pip install -e .) means:
#   1. `from tiny_diffusion.models.unet import UNet` works from anywhere —
#      no sys.path hacks, no relative import pain across notebooks/scripts.
#   2. Tests import the real package, not a copy — no drift between
#      "what we test" and "what we ship."
#   3. On Kaggle/Colab: `pip install -e .` once per session, then every
#      notebook cell and script shares the same code. Critical because
#      Kaggle/Colab notebooks otherwise encourage copy-pasted code drift.
#
# WHY scripts/ IS SEPARATE FROM src/:
#   scripts/ contains thin CLI entry points only — argument parsing and a
#   single call into src/. This keeps src/ free of argparse/Hydra boilerplate
#   so it stays unit-testable and importable as a clean library.


import os
import sys
import random
import subprocess
from pathlib import Path


# =============================================================================
# STEP 1.3 — DEPENDENCY MANAGEMENT: requirements.txt vs conda vs Docker
# =============================================================================
#
# DECISION: requirements.txt, with exact pinned versions. NOT conda. NOT Docker
# (for training). Docker comes back later for SERVING only (Phase 6).
#
# REASONING — comparing all three for THIS project's constraints:
#
# ── Conda environment ───────────────────────────────────────────────────
#   Pros: handles non-Python deps (CUDA toolkit) cleanly, isolated envs.
#   Cons: Kaggle and Colab both ship a fixed environment with PyTorch and
#         CUDA pre-installed and pre-linked to the GPU driver. Installing a
#         fresh conda env on top fights the platform's existing CUDA setup
#         and frequently breaks GPU visibility. This is a common source of
#         "torch.cuda.is_available() returns False" bugs on these platforms.
#   Verdict: wrong tool for Kaggle/Colab specifically.
#
# ── Docker ───────────────────────────────────────────────────────────────
#   Pros: perfect reproducibility, isolated, portable.
#   Cons: Kaggle and Colab do NOT let you run arbitrary Docker containers
#         for the free-tier GPU training session — you get a notebook
#         kernel, not a docker host. Docker is the right tool later for
#         the FastAPI inference server (Phase 6) which we deploy ourselves,
#         but it cannot be the training environment here.
#   Verdict: wrong tool for training on free-tier notebooks; correct tool
#            for serving later.
#
# ── requirements.txt (pip) — OUR CHOICE ────────────────────────────────
#   Pros: Both Kaggle and Colab already have a working Python + CUDA-linked
#         PyTorch. We only need to pip install the ADDITIONAL packages
#         (einops, mlflow, dvc, hydra-core, etc.) on top of what's there.
#         pip install --break-system-packages works fine on both platforms.
#         Exact version pins give us reproducibility without fighting the
#         platform's CUDA setup.
#   Cons: doesn't isolate from the platform's base environment — but that's
#         actually what we want here, since we need the platform's GPU-linked
#         torch, not a fresh one.
#   Verdict: correct tool for this specific constraint (ephemeral, GPU-linked,
#            notebook-based platforms).


REQUIREMENTS_TXT = """\
# ── Core ML (versions pinned for reproducibility) ──────────────────────────
# NOTE: torch/torchvision are usually pre-installed on Kaggle/Colab with the
# correct CUDA linkage. We pin versions here for LOCAL/CI environments where
# they are NOT pre-installed. On Kaggle/Colab, pip will see the existing
# install satisfies the pin (or we skip torch lines there — see setup_env.py).
torch==2.3.1
torchvision==0.18.1
einops==0.8.0

# ── Experiment tracking (Phase 2, Step 2) ───────────────────────────────────
mlflow==2.14.1
wandb==0.17.4

# ── Data/model versioning (Phase 2, Step 3) ─────────────────────────────────
dvc==3.51.2
dvc-gdrive==3.0.1

# ── Configuration management (Phase 2, Step 4) ──────────────────────────────
hydra-core==1.3.2
omegaconf==2.3.0

# ── Code quality (Phase 2, Step 5) ───────────────────────────────────────────
black==24.4.2
isort==5.13.2
flake8==7.1.0
mypy==1.10.1
pre-commit==3.7.1
pytest==8.2.2
pytest-cov==5.0.0

# ── Evaluation (Phase 4) ──────────────────────────────────────────────────
scipy==1.13.1          # for FID's matrix sqrt computation
pillow==10.4.0

# ── Quantization / export (Phase 5-6) ────────────────────────────────────
onnx==1.16.1
onnxruntime==1.18.1

# ── Serving (Phase 6) ──────────────────────────────────────────────────────
fastapi==0.111.0
uvicorn==0.30.1
pydantic==2.8.2
"""


def write_requirements(project_root: Path):
    """Write the pinned requirements.txt file."""
    path = project_root / "requirements.txt"
    path.write_text(REQUIREMENTS_TXT)
    print(f"  Wrote {path}")


# =============================================================================
# STEP 1.4 — .gitignore (data and checkpoints NEVER go into git)
# =============================================================================
#
# WHY: checkpoints are hundreds of MB, datasets are hundreds of MB. Git is
# built for text diffs, not binary blobs. Putting these in git would make
# the repo unclonable in reasonable time and bloat history permanently
# (git doesn't garbage-collect old blob versions by default).
# These go through DVC instead (Phase 2, Step 3), which is designed for
# exactly this — versioning large binary artifacts with pointers in git.

GITIGNORE = """\
# ── Data and model artifacts (tracked by DVC, not git) ─────────────────────
/data/raw/*
/data/processed/*
/outputs/checkpoints/*
/outputs/samples/*
/outputs/quantized_models/*
!/data/raw/.gitkeep
!/data/processed/.gitkeep
!/outputs/checkpoints/.gitkeep
!/outputs/samples/.gitkeep
!/outputs/quantized_models/.gitkeep
*.dvc
!/*.dvc.yaml

# ── Python ───────────────────────────────────────────────────────────────
__pycache__/
*.py[cod]
*.egg-info/
.eggs/
build/
dist/
.mypy_cache/
.pytest_cache/
.coverage
htmlcov/

# ── Environment / secrets ───────────────────────────────────────────────
.env
.venv/
venv/

# ── MLflow local artifacts (we use remote tracking, but ignore local mlruns) ─
mlruns/
mlartifacts/

# ── Jupyter ──────────────────────────────────────────────────────────────
.ipynb_checkpoints/
*.ipynb_meta

# ── OS ───────────────────────────────────────────────────────────────────
.DS_Store
Thumbs.db

# ── IDE ──────────────────────────────────────────────────────────────────
.vscode/
.idea/
"""


def write_gitignore(project_root: Path):
    path = project_root / ".gitignore"
    path.write_text(GITIGNORE)
    print(f"  Wrote {path}")


# =============================================================================
# STEP 1.5 — .env.example (secrets template — never commit real secrets)
# =============================================================================
#
# WHY a .env file at all: MLflow remote tracking (DagsHub) and DVC remote
# (Google Drive) need credentials. These must NEVER be hardcoded in source
# or committed to git. We commit a .env.example with placeholder values;
# each environment (your laptop, Kaggle secrets, Colab secrets) creates
# its own real .env that .gitignore excludes.

ENV_EXAMPLE = """\
# Copy this file to .env and fill in real values.
# .env is gitignored — never commit real credentials.

# ── MLflow remote tracking (DagsHub) ────────────────────────────────────────
MLFLOW_TRACKING_URI=https://dagshub.com/<your-username>/<repo-name>.mlflow
MLFLOW_TRACKING_USERNAME=<your-dagshub-username>
MLFLOW_TRACKING_PASSWORD=<your-dagshub-token>

# ── DVC remote (Google Drive) ───────────────────────────────────────────────
# Folder ID from your Google Drive URL: drive.google.com/drive/folders/<THIS_PART>
DVC_GDRIVE_FOLDER_ID=<your-gdrive-folder-id>

# ── Weights & Biases (if used instead of / alongside MLflow) ───────────────
WANDB_API_KEY=<your-wandb-api-key>
WANDB_PROJECT=tiny-diffusion-cifar10
"""


def write_env_example(project_root: Path):
    path = project_root / ".env.example"
    path.write_text(ENV_EXAMPLE)
    print(f"  Wrote {path}")


# =============================================================================
# STEP 1.6 — pyproject.toml (single source of truth for tool configs)
# =============================================================================
#
# WHY ONE FILE for black/isort/mypy/pytest config instead of separate
# setup.cfg / .flake8 / mypy.ini files:
#   pyproject.toml is the modern Python standard (PEP 518/621). Centralizing
#   config in one file means one less place to look when debugging why a
#   linter behaves a certain way, and it doubles as the package metadata
#   file that makes `pip install -e .` work (Step 1.2's reasoning).

PYPROJECT_TOML = """\
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
name = "tiny_diffusion"
version = "0.1.0"
description = "Tiny Conditional Diffusion on CIFAR-10: quantization study"
requires-python = ">=3.10"

# WHY explicit include= instead of bare auto-discovery:
# `[tool.setuptools.packages.find] where=["src"]` alone relies on setuptools
# correctly inferring this is a src-layout project. On some setuptools
# versions/platforms this silently discovers ZERO packages and fails with
# a confusing "No distribution was found" error during `pip install -e .`,
# rather than telling you discovery found nothing. Being explicit with
# include=["tiny_diffusion*"] removes the ambiguity entirely — setuptools
# is told exactly what to look for under src/, not asked to guess.
[tool.setuptools.packages.find]
where = ["src"]
include = ["tiny_diffusion*"]

[tool.setuptools.package-dir]
"" = "src"

# ── Black: code formatter ───────────────────────────────────────────────────
[tool.black]
line-length = 100
target-version = ["py310"]

# ── isort: import sorter, configured to not fight black ───────────────────
[tool.isort]
profile = "black"
line_length = 100
known_first_party = ["tiny_diffusion"]

# ── mypy: static type checker ───────────────────────────────────────────────
[tool.mypy]
python_version = "3.10"
ignore_missing_imports = true
disallow_untyped_defs = true
warn_return_any = true
warn_unused_ignores = true
# PyTorch's own type stubs are incomplete in places; this is the standard
# pragmatic setting used across most PyTorch projects.
[[tool.mypy.overrides]]
module = ["torch.*", "torchvision.*", "einops.*"]
ignore_missing_imports = true

# ── pytest ───────────────────────────────────────────────────────────────
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
addopts = "-v --tb=short"
"""


def write_pyproject(project_root: Path):
    path = project_root / "pyproject.toml"
    path.write_text(PYPROJECT_TOML)
    print(f"  Wrote {path}")


# =============================================================================
# STEP 1.7 — SEED MANAGEMENT
# =============================================================================
#
# THE PROBLEM: reproducibility requires controlling randomness in FOUR
# independent sources. Missing any one means "same seed, different result."
#
#   1. Python's random module       — used by some libraries internally
#   2. NumPy's random generator     — used by data augmentation, DVC stages
#   3. PyTorch's CPU random generator — used by nn.init, dropout, etc.
#   4. PyTorch's CUDA random generator — separate from CPU! GPU ops draw
#                                         from a different RNG state.
#
# A FIFTH SOURCE OF NON-DETERMINISM (not a seed, but related):
#   cuDNN's autotuner. By default, PyTorch lets cuDNN benchmark multiple
#   convolution algorithms and pick the fastest — but the "fastest" algorithm
#   can vary in numerical precision and even which one gets chosen can vary
#   run to run. For research reproducibility, we disable this.
#
# THE TRADEOFF: cudnn.deterministic=True can make training 10-20% SLOWER.
# Given our 6-hour budget, this matters. Our policy:
#   - During DEVELOPMENT and DEBUGGING: deterministic=True (catch real bugs,
#     not "is this difference real or just nondeterminism" confusion)
#   - During FINAL TRAINING RUNS for the paper/results: deterministic=True
#     (reproducibility is the whole point of the MLOps stack)
#   - We accept the speed cost. If T4 budget becomes too tight, this is the
#     first thing to relax, and we'll log that decision explicitly in MLflow.

def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set all random seeds for full reproducibility.

    Args:
        seed: the seed value (we'll use the same seed across all 4 sources)
        deterministic: if True, also force deterministic CUDA/cuDNN behavior.
                       Slower but exactly reproducible. See module docstring
                       above for the tradeoff reasoning.
    """
    # We import torch here (not at module top) so this file can be imported
    # for non-training utilities even in environments without torch installed.
    import numpy as np
    import torch

    # ── Source 1: Python's built-in random ──────────────────────────────────
    random.seed(seed)

    # ── Source 2: NumPy ──────────────────────────────────────────────────────
    np.random.seed(seed)

    # ── Source 3: PyTorch CPU ────────────────────────────────────────────────
    torch.manual_seed(seed)

    # ── Source 4: PyTorch CUDA (ALL GPUs, in case of multi-GPU) ─────────────
    # torch.manual_seed() alone does NOT seed CUDA — this is the #1 most
    # common reproducibility bug in PyTorch projects. Must call explicitly.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ── The cuDNN determinism setting ────────────────────────────────────────
    if deterministic:
        # Forces cuDNN to use deterministic algorithms only (no autotuning
        # variability). torch.use_deterministic_algorithms additionally makes
        # PyTorch raise an error if any operation has no deterministic
        # implementation, rather than silently falling back to nondeterministic.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        # warn_only=True: warn instead of crash for the rare op without a
        # deterministic kernel, rather than halting training entirely.
    else:
        # Default PyTorch behavior: cuDNN autotunes for speed.
        torch.backends.cudnn.benchmark = True

    # ── Environment variable for additional determinism (cuBLAS) ───────────
    # Some matrix multiply operations on GPU need this env var set BEFORE
    # CUDA context creation to be deterministic. If you set this after torch
    # has already initialized CUDA, it has no effect — set it as early as
    # possible in your script (ideally before importing torch).
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    print(f"  Seed set: {seed}  (deterministic={deterministic})")


def verify_reproducibility(seed: int = 42) -> bool:
    """
    Sanity test: run the same operation twice with the same seed,
    verify identical output. This is what we'd call from a pytest test.
    """
    import torch

    set_seed(seed, deterministic=True)
    a = torch.randn(100, 100)
    if torch.cuda.is_available():
        a = a.cuda()
        conv = torch.nn.Conv2d(3, 16, 3).cuda()
    else:
        conv = torch.nn.Conv2d(3, 16, 3)
    test_input = torch.randn(4, 3, 32, 32, device=a.device)
    out1 = conv(test_input).clone()

    set_seed(seed, deterministic=True)
    a2 = torch.randn(100, 100)
    if torch.cuda.is_available():
        a2 = a2.cuda()
    test_input2 = torch.randn(4, 3, 32, 32, device=a.device)
    out2 = conv(test_input2).clone()

    tensors_match = torch.allclose(a, a2) and torch.allclose(out1, out2)
    print(f"  Reproducibility check: {'PASSED' if tensors_match else 'FAILED'}")
    return tensors_match


# =============================================================================
# STEP 1.8 — BUILD THE FOLDER STRUCTURE
# =============================================================================

FOLDERS = [
    "configs/model",
    "configs/training",
    "configs/data",
    "configs/experiment",
    "configs/quantization",
    "src/tiny_diffusion/models",
    "src/tiny_diffusion/diffusion",
    "src/tiny_diffusion/data",
    "src/tiny_diffusion/training",
    "src/tiny_diffusion/evaluation",
    "src/tiny_diffusion/quantization",
    "src/tiny_diffusion/utils",
    "tests",
    "scripts",
    "data/raw",
    "data/processed",
    "outputs/checkpoints",
    "outputs/samples",
    "outputs/quantized_models",
    ".github/workflows",
]

# __init__.py files needed to make these importable Python packages.
# WHY explicit list rather than "every folder gets one": only src/ subfolders
# need to be Python packages. data/, outputs/, configs/ hold non-code files.
PACKAGE_INIT_FILES = [
    "src/tiny_diffusion/__init__.py",
    "src/tiny_diffusion/models/__init__.py",
    "src/tiny_diffusion/diffusion/__init__.py",
    "src/tiny_diffusion/data/__init__.py",
    "src/tiny_diffusion/training/__init__.py",
    "src/tiny_diffusion/evaluation/__init__.py",
    "src/tiny_diffusion/quantization/__init__.py",
    "src/tiny_diffusion/utils/__init__.py",
]

# .gitkeep files so git tracks otherwise-empty data/output directories
# (git does not track empty directories at all — .gitkeep is a workaround)
GITKEEP_FOLDERS = [
    "data/raw",
    "data/processed",
    "outputs/checkpoints",
    "outputs/samples",
    "outputs/quantized_models",
]


def build_folder_structure(project_root: Path):
    """Create the complete folder structure described in Step 1.2."""
    print(f"\nBuilding folder structure at: {project_root}")

    for folder in FOLDERS:
        path = project_root / folder
        path.mkdir(parents=True, exist_ok=True)

    for init_file in PACKAGE_INIT_FILES:
        path = project_root / init_file
        if not path.exists():
            path.write_text('"""tiny_diffusion package."""\n')

    for folder in GITKEEP_FOLDERS:
        path = project_root / folder / ".gitkeep"
        path.touch(exist_ok=True)

    print(f"  Created {len(FOLDERS)} folders")
    print(f"  Created {len(PACKAGE_INIT_FILES)} __init__.py files")
    print(f"  Created {len(GITKEEP_FOLDERS)} .gitkeep files")


def print_tree(project_root: Path, max_depth: int = 3):
    """Print the folder structure for visual verification."""
    print(f"\n{project_root.name}/")
    skip_dirs = {".git", "__pycache__", ".mypy_cache", ".pytest_cache"}

    def walk(directory: Path, prefix: str, depth: int):
        if depth > max_depth:
            return
        entries = sorted(
            [e for e in directory.iterdir() if e.name not in skip_dirs],
            key=lambda e: (e.is_file(), e.name),
        )
        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            print(f"{prefix}{connector}{entry.name}{'/' if entry.is_dir() else ''}")
            if entry.is_dir():
                extension = "    " if is_last else "│   "
                walk(entry, prefix + extension, depth + 1)

    walk(project_root, "", 1)


# =============================================================================
# STEP 1.9 — ENVIRONMENT DETECTION AND SETUP SCRIPT
# =============================================================================
#
# This is what actually runs at the start of every Kaggle/Colab session.
# It detects which platform we're on and adapts accordingly — because the
# two platforms differ in how they expose secrets, mount drives, and
# pre-install packages.

def detect_platform() -> str:
    """Detect whether we're running on Kaggle, Colab, or locally."""
    if os.path.exists("/kaggle/input"):
        return "kaggle"
    elif "google.colab" in sys.modules or os.path.exists("/content"):
        return "colab"
    else:
        return "local"


def verify_package_discovery(project_root: Path) -> bool:
    """
    Self-test: confirm setuptools will actually find the tiny_diffusion
    package BEFORE attempting the real `pip install -e .` / `uv pip install -e .`.

    WHY THIS STEP EXISTS:
    setuptools' `packages.find` auto-discovery can silently find zero
    packages on some setuptools versions if the src-layout isn't detected
    correctly. Without this check, you only find out something is wrong
    after a confusing build-backend traceback during the real install.
    This function runs the exact same discovery logic setuptools will use
    and reports the result directly — catches the bug in under a second.
    """
    try:
        import setuptools
    except ImportError:
        print("  setuptools not installed in current environment — skipping "
              "discovery check (will run for real during pip install).")
        return True

    src_dir = project_root / "src"
    found = setuptools.find_packages(where=str(src_dir), include=["tiny_diffusion*"])

    print(f"  setuptools.find_packages(where='src', include=['tiny_diffusion*'])")
    print(f"  -> found: {found}")

    if not found:
        print("  ✗ FAILED — discovery found zero packages.")
        print("    Check that src/tiny_diffusion/__init__.py exists.")
        return False

    expected_root = "tiny_diffusion"
    if expected_root not in found:
        print(f"  ✗ FAILED — expected '{expected_root}' in discovered packages.")
        return False

    print(f"  ✓ PASSED — {len(found)} package(s) discovered correctly.")
    return True


def setup_environment(project_root: Path):
    """
    The actual setup routine to run once per session.
    Installs additional packages on top of the platform's pre-installed
    torch/CUDA, verifies GPU, and confirms reproducibility.
    """
    platform = detect_platform()
    print(f"\nDetected platform: {platform}")

    print("\nChecking PyTorch / CUDA...")
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  VRAM: {vram_gb:.1f} GB")
    except ImportError:
        print("  PyTorch not found — install with: pip install torch torchvision")

    print("\nInstalling additional packages (mlflow, dvc, hydra, einops, etc.)...")
    extra_packages = [
        "einops", "mlflow", "dvc", "dvc-gdrive", "hydra-core", "omegaconf",
        "black", "isort", "flake8", "mypy", "pre-commit", "pytest", "pytest-cov",
    ]
    # break-system-packages needed on both Kaggle and Colab's externally
    # managed Python environments (PEP 668)
    cmd = [sys.executable, "-m", "pip", "install", "--break-system-packages", "-q"] + extra_packages
    print(f"  Running: pip install --break-system-packages -q {' '.join(extra_packages)}")
    # (Not executing automatically here — print the command so the user can
    #  run it explicitly and see output, since silent pip installs hide
    #  version conflicts that matter for reproducibility.)
    print("  -> Run the above command manually to see full install output.")

    print("\nInstalling tiny_diffusion package in editable mode...")
    print(f"  Run: pip install -e {project_root} --break-system-packages")

    print("\nVerifying seed reproducibility...")
    try:
        import torch  # noqa: F401  — just probing availability
        verify_reproducibility(seed=42)
    except ImportError:
        print("  Skipped — torch not installed in this environment.")
        print("  This will run for real on Kaggle/Colab where torch is pre-installed.")


# =============================================================================
# MAIN — run all of Step 1
# =============================================================================

if __name__ == "__main__":
    # ── Pick the right root for whichever platform this runs on ──────────────
    # Kaggle:  /kaggle/working is the only writable, session-persistent-until-
    #          session-ends directory. Anything here must be pushed to DVC/
    #          MLflow remotes before the session ends, or it's lost.
    # Colab:   /content is the equivalent — wiped on disconnect.
    # Local:   you already created and cd'd into tiny-diffusion-cifar10/ and
    #          activated venv (Phase 2 local setup steps) — build IN PLACE
    #          here, not in a nested subfolder. This matters because git was
    #          already initialized in this exact folder.
    _platform = detect_platform()
    if _platform == "kaggle":
        PROJECT_ROOT = Path("/kaggle/working/tiny-diffusion-cifar10")
    elif _platform == "colab":
        PROJECT_ROOT = Path("/content/tiny-diffusion-cifar10")
    else:
        PROJECT_ROOT = Path.cwd()

    print("=" * 70)
    print("PHASE 2, STEP 1 — PROJECT STRUCTURE AND ENVIRONMENT SETUP")
    print("=" * 70)

    build_folder_structure(PROJECT_ROOT)
    write_requirements(PROJECT_ROOT)
    write_gitignore(PROJECT_ROOT)
    write_env_example(PROJECT_ROOT)
    write_pyproject(PROJECT_ROOT)

    print("\n" + "-" * 50)
    print("FOLDER STRUCTURE")
    print("-" * 50)
    print_tree(PROJECT_ROOT)

    print("\n" + "-" * 50)
    print("PACKAGE DISCOVERY SELF-TEST")
    print("-" * 50)
    print("Validating setuptools can find tiny_diffusion BEFORE attempting")
    print("the real editable install — catches src-layout misconfiguration early.\n")
    discovery_ok = verify_package_discovery(PROJECT_ROOT)
    if not discovery_ok:
        print("\n  STOP — fix package discovery before running pip/uv install -e .")
        print("  Otherwise you'll hit a confusing 'No distribution was found' error.")

    print("\n" + "-" * 50)
    print("ENVIRONMENT CHECK")
    print("-" * 50)
    setup_environment(PROJECT_ROOT)

    print("\n" + "=" * 70)
    print("STEP 1 COMPLETE")
    print("=" * 70)
    print("""
What we built:
  [x] Full folder structure (configs/, src/, tests/, scripts/, data/, outputs/)
  [x] requirements.txt with pinned versions, reasoned against conda/Docker
  [x] .gitignore — data and checkpoints excluded from git (DVC's job instead)
  [x] .env.example — secrets template, real .env stays local and gitignored
  [x] pyproject.toml — single source of truth for black/isort/mypy/pytest
  [x] Seed management covering all 4 RNG sources + cuDNN determinism
  [x] Platform detection (Kaggle / Colab / local)

What comes next (Step 2):
  -> MLflow tracking server setup — where to host it for free, and why
  -> Experiment schema design — what exactly to log and why
""")