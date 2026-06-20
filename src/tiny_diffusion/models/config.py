"""
src/tiny_diffusion/models/config.py

Architecture configuration — see Phase 0 for the derivation of every
default value, and Phase 1 for the original combined-file version this
was split out from.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    image_size: int = 32
    in_channels: int = 3

    base_channels: int = 128
    channel_mult: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    num_res_blocks: int = 2

    attention_resolutions: List[int] = field(default_factory=lambda: [8, 4])
    num_heads: int = 4

    time_embed_dim: int = 512
    num_classes: int = 10
    cfg_dropout: float = 0.15

    T: int = 1000
    schedule: str = "cosine"
    cosine_s: float = 0.008

    num_groups: int = 32
    out_channels: int = 3

    def channel_at(self, level: int) -> int:
        """Channels at resolution level (0=highest res, 3=bottleneck)."""
        return self.base_channels * self.channel_mult[level]

    def to_dict(self) -> dict:
        """For MLflow logging (Phase 2 Step 2)."""
        return {
            "image_size": self.image_size,
            "base_channels": self.base_channels,
            "channel_mult": str(self.channel_mult),
            "num_res_blocks": self.num_res_blocks,
            "attention_resolutions": str(self.attention_resolutions),
            "num_heads": self.num_heads,
            "time_embed_dim": self.time_embed_dim,
            "num_classes": self.num_classes,
            "cfg_dropout": self.cfg_dropout,
            "T": self.T,
            "schedule": self.schedule,
        }
