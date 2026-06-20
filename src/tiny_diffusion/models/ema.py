"""
src/tiny_diffusion/models/ema.py

Exponential Moving Average wrapper — split out from Phase 1's combined
file. See Phase 0, Section 8 for why EMA is load-bearing for diffusion
models specifically (conflicting per-timestep gradient directions).
"""

from typing import Any, Dict

import torch
import torch.nn as nn


class EMA:
    """
    Exponential Moving Average of model parameters.

    Usage:
        ema = EMA(model, decay=0.9999)

        # In training loop, after optimizer.step():
        ema.update(model, step)

        # For evaluation/sampling:
        with ema.apply(model):
            samples = model(...)   # uses EMA weights

        # Or explicitly:
        ema.copy_to(model)         # model now has EMA weights
        samples = model(...)
        ema.restore(model)         # restore training weights
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        # Deep copy: EMA shadow weights start identical to initial model weights.
        # .parameters() only — we don't EMA the buffers (like running_mean in BN).
        # We store as a separate dict rather than a full model copy to save memory.
        self.shadow: Dict[str, torch.Tensor] = {
            name: param.data.clone() for name, param in model.named_parameters()
        }
        self._stored_weights: Dict[str, torch.Tensor] = {}  # for the context manager

    def update(self, model: nn.Module, step: int) -> None:
        """
        Update EMA weights after one training step.

        Args:
            model: the training model (with fresh weights from optimizer.step)
            step:  current training step (for warmup)
        """
        # Warmup: effective decay grows from ~0 to self.decay over first 10k steps.
        # Formula: min(decay, (1 + step) / (10 + step))
        # At step 0:   (1+0)/(10+0)  = 0.1    → very fast update
        # At step 100: 101/110        = 0.918  → moderate
        # At step 9990: 9991/10000   = 0.9991 → near target
        decay = min(self.decay, (1 + step) / (10 + step))

        with torch.no_grad():
            for name, param in model.named_parameters():
                # EMA update: shadow = decay * shadow + (1-decay) * param
                # in-place for memory efficiency
                self.shadow[name].mul_(decay).add_(param.data, alpha=1 - decay)

    def copy_to(self, model: nn.Module) -> None:
        """Replace model parameters with EMA weights."""
        for name, param in model.named_parameters():
            param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        """Restore original training weights after using EMA for evaluation."""
        for name, param in model.named_parameters():
            param.data.copy_(self._stored_weights[name])
        self._stored_weights = {}

    class _EMAContextManager:
        """Context manager: temporarily swap to EMA weights."""

        def __init__(self, ema: "EMA", model: nn.Module) -> None:
            self.ema = ema
            self.model = model

        def __enter__(self) -> None:
            # Store training weights
            self.ema._stored_weights = {
                name: param.data.clone() for name, param in self.model.named_parameters()
            }
            # Switch to EMA weights
            self.ema.copy_to(self.model)

        def __exit__(self, *args: Any) -> None:
            # Restore training weights
            self.ema.restore(self.model)

    def apply(self, model: nn.Module) -> "EMA._EMAContextManager":
        """
        Context manager for temporary EMA weight application.

        Usage:
            with ema.apply(model):
                # model uses EMA weights here
                output = model(input)
            # model restored to training weights here
        """
        return self._EMAContextManager(self, model)

    def state_dict(self) -> Dict[str, Any]:
        """For saving EMA state in checkpoints."""
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """For loading EMA state from checkpoints."""
        self.shadow = state["shadow"]
        self.decay = state["decay"]
