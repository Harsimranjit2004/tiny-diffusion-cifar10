"""
scripts/train.py

PHASE 3 — REAL TRAINING ENTRY POINT

Phase 2 Step 4 used this file purely to validate Hydra config composition
(printing the resolved config, no actual training). Now that Phase 3's
real training loop exists in src/tiny_diffusion/training/train.py, this
script's job is to load the config via Hydra and hand it off.

USAGE:
  python scripts/train.py
      -> trains with configs/experiment/baseline.yaml

  python scripts/train.py training.lr=0.0001
      -> same experiment, overridden learning rate

  python scripts/train.py --multirun training.lr=1e-4,2e-4,5e-4
      -> three separate training runs, one per LR (Phase 5's mechanism
         for the quantization ablation study uses this same multirun
         pattern, just sweeping quantization method instead of LR)
"""

import hydra
from omegaconf import DictConfig

from tiny_diffusion.training.train import train


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
