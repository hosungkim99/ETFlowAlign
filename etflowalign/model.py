
"""Core ETFlowAlign model definition.

This file contains the high-level model contract only:
- inputs represent query/reference alignment context,
- outputs are time-dependent coordinate vector fields over query atoms.

The backbone is intentionally lightweight in this scaffold and should be
replaced by a stronger E(3)-equivariant architecture in later iterations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn


@dataclass
class AlignmentBatch:
    """Minimal self-contained batch container for ETFlowAlign.

    Notes:
        This avoids hard-coding external data classes in the scaffold.
        A production implementation can adapt this to PyG Data/Batch objects.
    """

    query_pos: Tensor
    query_atom_type: Tensor
    query_batch: Tensor
    reference_pos: Optional[Tensor] = None
    reference_atom_type: Optional[Tensor] = None
    reference_batch: Optional[Tensor] = None
    pocket_pos: Optional[Tensor] = None


class SimpleTimeEmbedding(nn.Module):
    """Small sinusoidal time embedding used by the scaffold model."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: Tensor) -> Tensor:
        # t: [num_graphs] in [0, 1]
        half = self.dim // 2
        freqs = torch.exp(
            torch.linspace(0.0, 1.0, half, device=t.device) * (-torch.log(torch.tensor(10000.0, device=t.device)))
        )
        phase = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return self.proj(emb)


class ETFlowAlignModel(nn.Module):
    """Time-dependent vector-field model for alignment.

    Contract:
        v = model(batch, t)
        - v.shape == batch.query_pos.shape
        - v is an equivariant velocity field over query coordinates.

    TODO:
        Replace the simple MLP backbone with an E(3)-equivariant
        reference-conditioned architecture.
    """

    def __init__(
        self,
        atom_vocab_size: int = 128,
        atom_embed_dim: int = 64,
        time_embed_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.atom_embed = nn.Embedding(atom_vocab_size, atom_embed_dim)
        self.time_embed = SimpleTimeEmbedding(time_embed_dim)

        # NOTE: this scaffold uses local coordinate + simple context pooling.
        in_dim = 3 + atom_embed_dim + time_embed_dim + 3
        self.velocity_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def _per_graph_center(self, x: Tensor, batch: Tensor) -> Tensor:
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        if num_graphs == 0:
            return x
        centers = torch.zeros(num_graphs, 3, device=x.device, dtype=x.dtype)
        counts = torch.zeros(num_graphs, 1, device=x.device, dtype=x.dtype)
        centers.index_add_(0, batch, x)
        ones = torch.ones_like(batch, dtype=x.dtype).unsqueeze(-1)
        counts.index_add_(0, batch, ones)
        centers = centers / counts.clamp_min(1.0)
        return x - centers[batch]

    def _reference_context(self, batch: AlignmentBatch) -> Tensor:
        # Returns per-query-node coarse context vector (3D) using reference COM.
        if batch.reference_pos is None or batch.reference_batch is None:
            return torch.zeros_like(batch.query_pos)

        ref = batch.reference_pos
        ref_batch = batch.reference_batch
        num_graphs = int(ref_batch.max().item()) + 1 if ref_batch.numel() > 0 else 0
        ref_centers = torch.zeros(num_graphs, 3, device=ref.device, dtype=ref.dtype)
        ref_counts = torch.zeros(num_graphs, 1, device=ref.device, dtype=ref.dtype)
        ref_centers.index_add_(0, ref_batch, ref)
        ref_counts.index_add_(0, ref_batch, torch.ones_like(ref_batch, dtype=ref.dtype).unsqueeze(-1))
        ref_centers = ref_centers / ref_counts.clamp_min(1.0)
        return ref_centers[batch.query_batch] - batch.query_pos

    def forward(self, batch: AlignmentBatch, t_graph: Tensor) -> Tensor:
        """Predict vector field over query atoms.

        Args:
            batch: alignment input.
            t_graph: [num_graphs] continuous time values in [0, 1].
        """
        q_pos_centered = self._per_graph_center(batch.query_pos, batch.query_batch)
        atom_feat = self.atom_embed(batch.query_atom_type)
        t_node = self.time_embed(t_graph)[batch.query_batch]
        ref_ctx = self._reference_context(batch)
        x = torch.cat([q_pos_centered, atom_feat, t_node, ref_ctx], dim=-1)
        return self.velocity_head(x)
