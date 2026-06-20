"""
scripts/download_data.py

PHASE 3 — DVC data_download STAGE

WHY THIS USES THE HUGGING FACE MIRROR (uoft-cs/cifar10) INSTEAD OF
torchvision.datasets.CIFAR10's DIRECT DOWNLOAD: discovered during Phase
3's first real GPU smoke test — the original University of Toronto host
was measured at ~24-68 kB/s on both Kaggle and Colab (nearly 2 hours for
170MB), while the identical dataset on Hugging Face's CDN downloads in
~2 seconds at 50+ MB/s in the same sessions. See src/tiny_diffusion/data/
cifar10.py's module docstring for the full reasoning — this script just
needs to trigger the same download path so DVC's data_download stage
output exists before the preprocess stage runs.
"""

from datasets import load_dataset


def main() -> None:
    print("Downloading CIFAR-10 (Hugging Face mirror: uoft-cs/cifar10) to data/raw/ ...")
    load_dataset("uoft-cs/cifar10", split="train", cache_dir="data/raw")
    load_dataset("uoft-cs/cifar10", split="test", cache_dir="data/raw")
    print("Done.")


if __name__ == "__main__":
    main()
