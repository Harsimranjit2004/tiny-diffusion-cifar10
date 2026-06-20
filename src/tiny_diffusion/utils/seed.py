"""
src/tiny_diffusion/utils/seed.py

Seed management — split out from Phase 2 Step 1's combined structure
script into its own importable module. See that phase's documentation
for the full reasoning on all four RNG sources and the cuDNN determinism
speed/reproducibility tradeoff.
"""

import os
import random


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

    tensors_match = bool(torch.allclose(a, a2) and torch.allclose(out1, out2))
    print(f"  Reproducibility check: {'PASSED' if tensors_match else 'FAILED'}")
    return tensors_match
