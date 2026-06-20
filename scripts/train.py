"""
scripts/train.py

PHASE 2, STEP 4 — HYDRA CONFIG COMPOSITION VALIDATION

This is intentionally NOT the real training loop yet (that's Phase 3).
Its only job right now is to prove the Hydra config hierarchy we just
built actually composes correctly end to end — model + training + data +
schedule all merging into one resolved config, with command-line overrides
working as designed.

USAGE:
  python scripts/train.py
      -> runs with configs/experiment/baseline.yaml (the default)

  python scripts/train.py training.lr=0.0001
      -> overrides just the learning rate, everything else stays default

  python scripts/train.py schedule=linear
      -> swaps the ENTIRE schedule config group to linear.yaml,
         proving config groups are independently swappable

  python scripts/train.py --multirun training.lr=1e-4,2e-4,5e-4
      -> Hydra's multirun/sweep feature — runs three times, once per LR.
         This is the mechanism Phase 5's quantization ablation will use.
"""

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print("=" * 70)
    print("HYDRA CONFIG COMPOSITION CHECK")
    print("=" * 70)

    print("\nFully resolved config (this is what train.py would actually see):\n")
    print(OmegaConf.to_yaml(cfg))

    # ── Sanity checks: prove cross-config-group access works ──────────────
    print("-" * 50)
    print("ACCESS CHECKS")
    print("-" * 50)
    print(f"  cfg.experiment.experiment_name = {cfg.experiment.experiment_name}")
    print(f"  cfg.experiment.model.base_channels = {cfg.experiment.model.base_channels}")
    print(f"  cfg.experiment.training.lr = {cfg.experiment.training.lr}")
    print(f"  cfg.experiment.data.dataset_name = {cfg.experiment.data.dataset_name}")
    print(f"  cfg.experiment.schedule.type = {cfg.experiment.schedule.type}")
    print(f"  cfg.experiment.tags = {dict(cfg.experiment.tags)}")

    # ── Confirm this matches what ModelConfig (Phase 1) expects ────────────
    print("\n" + "-" * 50)
    print("PHASE 1 MODELCONFIG COMPATIBILITY CHECK")
    print("-" * 50)
    expected_model_fields = {
        "base_channels", "channel_mult", "num_res_blocks",
        "attention_resolutions", "num_heads", "time_embed_dim",
        "num_classes", "cfg_dropout", "num_groups",
        "in_channels", "out_channels", "image_size",
    }
    actual_fields = set(cfg.experiment.model.keys())
    missing = expected_model_fields - actual_fields
    extra = actual_fields - expected_model_fields

    if missing:
        print(f"  ✗ MISSING fields Phase 1's ModelConfig needs: {missing}")
    else:
        print(f"  ✓ All {len(expected_model_fields)} ModelConfig fields present")

    if extra:
        print(f"  (extra fields not used by ModelConfig: {extra})")

    print("\n" + "=" * 70)
    print("If you see this, Hydra composition is working correctly.")
    print("=" * 70)


if __name__ == "__main__":
    main()