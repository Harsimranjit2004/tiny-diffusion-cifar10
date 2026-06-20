"""
scripts/download_data.py

PHASE 3 — DVC data_download STAGE

WHAT THIS SCRIPT DOES: downloads CIFAR-10 to data/raw/ via torchvision's
built-in downloader. This is intentionally the THINNEST possible script —
its only job is to trigger the download, matching exactly what dvc.yaml's
data_download stage declares as its output path
(data/raw/cifar-10-batches-py).

WHY THIS IS SEPARATE FROM cifar10.py's CIFAR10Diffusion class (which also
has download=True as a default): that class downloads AS A SIDE EFFECT of
being instantiated for training. This script exists so `dvc repro` can
invoke JUST the download step in isolation, satisfying the DAG dependency
declared in dvc.yaml, without needing to construct a full Dataset object
with transforms that aren't relevant to "did the raw files arrive."
"""

import torchvision


def main() -> None:
    print("Downloading CIFAR-10 to data/raw/ ...")
    # download=True triggers the fetch if not already present; if the
    # files already exist (e.g. DVC pulled them from the remote), this
    # is a fast no-op verification rather than a redundant re-download.
    torchvision.datasets.CIFAR10(root="data/raw", train=True, download=True)
    torchvision.datasets.CIFAR10(root="data/raw", train=False, download=True)
    print("Done.")


if __name__ == "__main__":
    main()
