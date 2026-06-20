"""
src/tiny_diffusion/models/embeddings.py

Time and class conditioning embeddings — split out from Phase 1's
combined file into its own module, per Step 1's intended package layout.
See Phase 0, Sections on time embedding and CFG for the full derivations.
"""

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """
    Converts integer timestep t -> rich embedding vector.

    Flow:
      t (integer, shape [B])
      → sinusoidal encoding  [B, half_dim*2]
      → Linear(dim, 4*dim)   [B, 4*dim]
      → SiLU activation
      → Linear(4*dim, dim)   [B, dim]
    """

    def __init__(self, dim: int):
        super().__init__()
        # dim = time_embed_dim = 512 in our config
        self.dim = dim

        # The MLP that processes sinusoidal features.
        # SiLU (Sigmoid Linear Unit) = x * sigmoid(x).
        # WHY SiLU instead of ReLU:
        #   SiLU is smooth everywhere — no dead neuron problem.
        #   DDPM and most modern diffusion models use SiLU.
        #   ReLU kills negative activations permanently; SiLU gates them softly.
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),  # expand to 4x for capacity
            nn.SiLU(),  # smooth nonlinearity
            nn.Linear(dim * 4, dim),  # project back to dim
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: integer timesteps, shape [B]
               Values in range [1, T] (1-indexed, matching math notation)
        Returns:
            embedding: shape [B, dim]
        """
        # ── Step 1: Compute sinusoidal encoding ───────────────────────────
        # half_dim = dim/2 because we produce both sin and cos components,
        # then concatenate them to get the full dim-dimensional vector.
        half_dim = self.dim // 2

        # This computes the denominators: 10000^(2i/d) for i=0..half_dim-1
        # In log space for numerical stability:
        #   log(10000^(2i/d)) = (2i/d) * log(10000)
        # Then exp() to get the actual denominators.
        #
        # torch.arange(half_dim) produces [0, 1, 2, ..., half_dim-1]
        # Dividing by half_dim gives the normalized indices 2i/d
        # (where d = half_dim*2, so 2i/d = i/half_dim)
        #
        # NOTE: this scalar constant gets its own name (log_ratio) rather
        # than reusing `emb` for both a float AND a tensor in sequence —
        # mypy correctly flagged the original version (same variable name
        # holding a float then a tensor) as a real type-safety smell, the
        # kind of thing that causes confusing bugs if this function is
        # edited later by someone who doesn't trace through every line.
        log_ratio = math.log(10000) / (half_dim - 1)
        # Shape: [half_dim]
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -log_ratio)

        # t is shape [B], emb is shape [half_dim]
        # Outer product: t[:, None] is [B, 1], emb[None, :] is [1, half_dim]
        # Result: [B, half_dim] — each row is t_i * all_denominators
        emb = t[:, None].float() * emb[None, :]

        # Concatenate sin and cos → [B, half_dim*2] = [B, dim]
        # WHY both sin and cos: sin alone is not injective (sin(x)=sin(pi-x))
        # Together they uniquely encode any angle/position.
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        # ── Step 2: MLP to mix frequencies ────────────────────────────────
        # Shape stays [B, dim] throughout the MLP.
        emb = self.mlp(emb)

        return emb  # [B, dim]


# =============================================================================
# STEP 3 — CLASS CONDITIONING WITH CFG DROPOUT
# =============================================================================
#
# PURPOSE: convert a class label (integer 0-9, or 10 for null) into
# an embedding vector of the same dimension as the time embedding.
# The two are then ADDED together and injected into every ResBlock.
#
# WHY ADDITION (not concatenation, not cross-attention):
#   - Cross-attention: designed for long sequences (text tokens). Overkill
#     for 10 class labels. Would add attention layers at every resolution.
#   - Concatenation: would double the conditioning vector, requiring bigger
#     projection layers in every ResBlock. More params, same information.
#   - Addition: time and class embeddings live in the same 512-dim space.
#     Adding them says "this point in time, for this class." Clean and cheap.
#     This is the standard approach in all DDPM class-conditional papers.
#
# CFG DROPOUT BUILT INTO THIS MODULE:
#   During training, with probability cfg_dropout, we replace the real
#   class label with label=num_classes (our null token index).
#   This is the ONLY place where CFG dropout happens.
#   The rest of the model never sees this logic — it just gets an embedding.


class ClassEmbedding(nn.Module):
    """
    Class label -> embedding vector, with CFG null token support.

    Embedding table size: num_classes + 1
      Indices 0-9:  CIFAR-10 class embeddings
      Index    10:  null token (unconditional, used by CFG)
    """

    def __init__(self, num_classes: int, embed_dim: int, cfg_dropout: float = 0.15):
        super().__init__()
        self.num_classes = num_classes
        self.cfg_dropout = cfg_dropout
        self.null_token_idx = num_classes  # index 10 = unconditional

        # +1 for the null token. The table has 11 rows total.
        # Each row is a learnable embed_dim-dimensional vector.
        # nn.Embedding is just a lookup table — no computation, just indexing.
        self.embedding = nn.Embedding(num_classes + 1, embed_dim)

    def forward(self, labels: torch.Tensor, force_uncond: bool = False) -> torch.Tensor:
        """
        Args:
            labels: class indices, shape [B], values in [0, num_classes]
                    (num_classes = null token, already set by caller for CFG inference)
            force_uncond: if True, return null embeddings for entire batch.
                          Used during CFG inference unconditional pass.
        Returns:
            class_emb: shape [B, embed_dim]
        """
        if force_uncond:
            # CFG inference: unconditional branch.
            # Replace ALL labels with null token.
            labels = torch.full_like(labels, self.null_token_idx)

        elif self.training and self.cfg_dropout > 0:
            # Training: randomly drop some labels to null token.
            # torch.rand gives uniform [0,1) — where < cfg_dropout → drop.
            drop_mask = torch.rand(labels.shape[0], device=labels.device) < self.cfg_dropout
            # Where drop_mask is True, replace with null_token_idx.
            # Where False, keep original label.
            labels = torch.where(drop_mask, self.null_token_idx, labels)

        # Simple lookup: for each label in [B], return the corresponding row
        # from the embedding table. Output: [B, embed_dim].
        return self.embedding(labels)
