"""
src/tiny_diffusion/data/cifar10.py

PHASE 3 — CIFAR-10 DATA PIPELINE

WHAT THIS MODULE DOES:
  Loads CIFAR-10, applies the deterministic preprocessing (normalization)
  separately from the per-epoch random augmentations (flip, crop), and
  returns a DataLoader ready for the training loop.

WHY NORMALIZATION VALUES ARE THESE SPECIFIC NUMBERS:
  (0.4914, 0.4822, 0.4465) and (0.2470, 0.2435, 0.2616) are the empirically
  measured per-channel mean and std of the CIFAR-10 TRAINING set. These are
  widely published, standard values (computing them fresh would just
  reproduce the same numbers at the cost of an extra dataset pass) — see
  configs/data/cifar10.yaml's comment on this exact point.

WHY WE NORMALIZE TO ROUGHLY [-1, 1] RATHER THAN [0, 1]:
  Diffusion models add Gaussian noise ε ~ N(0,1) directly to the image
  tensor (Phase 0's forward process: x_t = sqrt(abar)*x_0 + sqrt(1-abar)*ε).
  For this addition to make sense, x_0 should live on a similar SCALE to
  the noise — values centered near 0 with unit-ish variance, not values
  in [0,1] which would make the early, low-noise timesteps add disproportionately
  large perturbations relative to the signal's own scale. Standard practice
  across all diffusion papers: normalize images to mean~0.

WHY AUGMENTATION HAPPENS HERE, NOT IN THE DVC preprocess STAGE:
  Phase 2 Step 3's dvc.yaml comment on the preprocess stage already
  explained this: random crop and flip must be RE-RANDOMIZED every epoch.
  Precomputing them once would mean training sees the same augmented
  version of each image every single epoch — defeating the entire purpose
  of augmentation (forcing the model to generalize past exact pixel
  patterns). Only deterministic preprocessing (normalization stats) is
  precomputed and cached.
"""

from typing import Tuple

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset


class CIFAR10Diffusion(Dataset):
    """
    Wraps torchvision's CIFAR10 dataset with diffusion-appropriate
    normalization and the CFG-compatible label scheme.

    WHY WE WRAP RATHER THAN USE torchvision.datasets.CIFAR10 DIRECTLY:
    torchvision's labels are 0-9. Our ClassEmbedding table (Phase 1) has
    11 entries (0-9 real classes + index 10 for the CFG null token). The
    dataset itself should never produce label 10 — that's something
    ClassEmbedding's forward() injects at TRAINING TIME via random dropout
    (Phase 1's cfg_dropout logic), not something baked into the data. This
    wrapper exists mainly to centralize the transform pipeline in one
    place rather than scattering transform logic across train.py.
    """

    def __init__(
        self,
        root: str = "data/raw",
        train: bool = True,
        normalize_mean: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465),
        normalize_std: Tuple[float, float, float] = (0.2470, 0.2435, 0.2616),
        random_horizontal_flip: bool = True,
        random_crop_padding: int = 4,
        download: bool = True,
    ):
        super().__init__()

        transform_list = []

        # ── Random augmentations (only applied if train=True) ────────────
        # WHY GATED ON train: we NEVER want random augmentation during
        # evaluation/sampling — that would make FID computation
        # non-reproducible run to run for reasons having nothing to do
        # with the model itself. Augmentation is a TRAINING-only concept.
        if train:
            if random_crop_padding > 0:
                # Standard CIFAR-10 augmentation: pad by N pixels on each
                # side, then randomly crop back to the original 32x32.
                # This effectively gives the model slightly different
                # spatial crops of the same image every epoch — a cheap,
                # well-validated way to reduce overfitting on a small
                # (50k image) dataset.
                transform_list.append(T.RandomCrop(32, padding=random_crop_padding))
            if random_horizontal_flip:
                # CIFAR-10 classes are mostly left-right symmetric in
                # natural variation (a horse facing left vs right is still
                # a horse) — horizontal flip is a safe augmentation here.
                # WHY NOT vertical flip: an upside-down airplane or truck
                # is not a realistic training example for THIS dataset's
                # natural image distribution — vertical flip would teach
                # the model an unrealistic prior.
                transform_list.append(T.RandomHorizontalFlip(p=0.5))

        # ── Deterministic preprocessing (always applied) ──────────────────
        transform_list.append(T.ToTensor())  # [0,255] uint8 -> [0,1] float32
        transform_list.append(T.Normalize(mean=normalize_mean, std=normalize_std))
        # WHY Normalize here gives us roughly [-1,1]: with mean~0.48 and
        # std~0.25, a pixel originally at 1.0 maps to (1.0-0.48)/0.25≈2.1,
        # and a pixel at 0.0 maps to -0.48/0.25≈-1.9 — not EXACTLY [-1,1]
        # but close enough, and using the empirically-measured stats (not
        # a hardcoded 0.5/0.5) is the more principled standard choice.

        self.transform = T.Compose(transform_list)

        self.dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=download)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, label = self.dataset[idx]  # PIL image, int label (0-9)
        image = self.transform(image)
        return image, label


def get_dataloader(
    root: str = "data/raw",
    train: bool = True,
    batch_size: int = 128,
    normalize_mean: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465),
    normalize_std: Tuple[float, float, float] = (0.2470, 0.2435, 0.2616),
    random_horizontal_flip: bool = True,
    random_crop_padding: int = 4,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Build the actual DataLoader the training loop iterates over.

    WHY num_workers AND pin_memory MATTER (connects to Phase 2 Step 2's
    GPU utilization logging and the ML Systems textbook's Roofline framing):
      num_workers>0 lets the CPU preprocess the NEXT batch while the GPU
      is still busy computing on the CURRENT batch — without this, the GPU
      sits idle during every data-loading step, exactly the "data-bound,
      not compute-bound" failure mode that log_system_metrics()'s
      gpu_utilization_pct is designed to catch.
      pin_memory=True allocates the batch in page-locked host memory,
      making the CPU->GPU transfer faster (avoids an extra copy the OS
      would otherwise do). Free speedup, standard practice.
    """
    dataset = CIFAR10Diffusion(
        root=root,
        train=train,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
        random_horizontal_flip=random_horizontal_flip,
        random_crop_padding=random_crop_padding,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,  # only shuffle during training, never during eval
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=train,
        # WHY drop_last=True only for training: dropping an incomplete
        # final batch keeps batch_size CONSTANT across every training
        # step, which matters because GroupNorm and other batch-shape-
        # sensitive ops behave most predictably with consistent batch
        # sizes. For evaluation we want every sample seen exactly once,
        # so we keep the (possibly smaller) final batch.
    )


def denormalize(
    x: torch.Tensor,
    normalize_mean: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465),
    normalize_std: Tuple[float, float, float] = (0.2470, 0.2435, 0.2616),
) -> torch.Tensor:
    """
    Reverse the normalization to get back viewable [0,1] images —
    needed every time we save a sample grid (Phase 2 Step 2's
    log_sample_grid) or compute FID, since both expect images in
    a standard displayable/measurable range, not the model's internal
    roughly-[-1,1] training scale.
    """
    mean = torch.tensor(normalize_mean, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(normalize_std, device=x.device).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)
