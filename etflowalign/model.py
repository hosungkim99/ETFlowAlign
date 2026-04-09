"""Core ETFlowAlign model definition.

This module implements a minimal but concrete E(3)-equivariant vector-field model
for alignment. The architecture is intentionally compact so the entire design can
be understood from this repository without external wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from .utils import safe_norm, segment_mean


@dataclass
class AlignmentBatch:
    """Minimal self-contained batch container for ETFlowAlign.

    Attributes:
        query_pos: Query ligand coordinates with shape ``[Nq, 3]``.
        query_atom_type: Integer atom type ids for query atoms ``[Nq]``.
        query_batch: Graph index for each query atom ``[Nq]``.
        reference_pos: Reference ligand coordinates ``[Nr, 3]``.
        reference_atom_type: Optional reference atom type ids ``[Nr]``.
        reference_batch: Graph index for each reference atom ``[Nr]``.
        pocket_pos: Optional pocket/receptor point coordinates.
    """

    query_pos: Tensor
    query_atom_type: Tensor
    query_batch: Tensor
    reference_pos: Optional[Tensor] = None
    reference_atom_type: Optional[Tensor] = None
    reference_batch: Optional[Tensor] = None
    pocket_pos: Optional[Tensor] = None
    query_node_attr: Optional[Tensor] = None
    reference_node_attr: Optional[Tensor] = None


class SimpleTimeEmbedding(nn.Module):
    """Sinusoidal embedding for continuous flow time t in [0, 1]."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: Tensor) -> Tensor:
        """Embed flow time for each graph.

        Args:
            t: Continuous times in ``[0, 1]`` with shape ``[B]``.

        Returns:
            Time embeddings with shape ``[B, dim]``.
        """
        half = self.dim // 2
        if half == 0:
            return t[:, None]
        freq_exp = -torch.log(torch.tensor(10000.0, device=t.device))
        freqs = torch.exp(torch.linspace(0.0, 1.0, half, device=t.device) * freq_exp)
        phase = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        if emb.size(-1) < self.dim:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return self.proj(emb)


def build_radius_edges(pos: Tensor, batch: Tensor, cutoff: float, max_neighbors: int) -> Tensor:
    """Build intra-graph radius edges with small-N dense fallback (self-contained)."""
    # src/dst collect directed edges (i -> j) within each graph.
    src, dst = [], []
    num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
    cutoff_sq = cutoff * cutoff

    for g in range(num_graphs):
        node_idx = torch.where(batch == g)[0]
        if node_idx.numel() <= 1:
            continue
        p = pos[node_idx]  # Local coordinates for graph g: [Ng, 3]
        diff = p[:, None, :] - p[None, :, :]
        dist_sq = (diff * diff).sum(-1)
        mask = (dist_sq <= cutoff_sq) & (~torch.eye(node_idx.numel(), device=pos.device, dtype=torch.bool))

        for i in range(node_idx.numel()):
            nbr_local = torch.where(mask[i])[0]
            if nbr_local.numel() > max_neighbors:
                d = dist_sq[i, nbr_local]
                nbr_local = nbr_local[torch.argsort(d)[:max_neighbors]]
            if nbr_local.numel() > 0:
                src.append(node_idx[i].repeat(nbr_local.numel()))
                dst.append(node_idx[nbr_local])

    if not src:
        return torch.empty(2, 0, dtype=torch.long, device=pos.device)
    return torch.stack([torch.cat(src), torch.cat(dst)], dim=0)


class EquivariantBlock(nn.Module):
    """Simple EGNN-style block: scalar message + relative vector aggregation."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.phi_e = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.phi_h = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.phi_x = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def forward(self, h: Tensor, x: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        """Run one equivariant message passing block.

        Args:
            h: Node scalar features ``[N, H]``.
            x: Node coordinates ``[N, 3]``.
            edge_index: Directed edges ``[2, E]``.

        Returns:
            Updated node features and coordinates.
        """
        if edge_index.numel() == 0:
            return h, x

        i, j = edge_index[0], edge_index[1]
        rij = x[i] - x[j]
        dij = torch.norm(rij, dim=-1, keepdim=True)

        e_ij = self.phi_e(torch.cat([h[i], h[j], dij], dim=-1))

        # Coordinate update (equivariant): sum alpha_ij * (x_i - x_j)
        alpha_ij = self.phi_x(e_ij)
        dx_msg = alpha_ij * rij
        dx = torch.zeros_like(x)
        dx.index_add_(0, i, dx_msg)
        x = x + dx

        # Feature update: aggregate messages to node i
        m = torch.zeros_like(h)
        m.index_add_(0, i, e_ij)
        h = h + self.phi_h(torch.cat([h, m], dim=-1))
        return h, x


class ETFlowAlignModel(nn.Module):
    """Compact E(3)-equivariant time-dependent vector field model for query atoms."""

    def __init__(
        self,
        atom_vocab_size: int = 128,
        hidden_dim: int = 128,
        time_embed_dim: int = 64,
        extra_feat_dim: int = 0,
        num_blocks: int = 4,
        edge_cutoff: float = 6.0,
        max_neighbors: int = 32,
    ) -> None:
        super().__init__()
        self.edge_cutoff = float(edge_cutoff)
        self.max_neighbors = int(max_neighbors)

        self.atom_embed = nn.Embedding(atom_vocab_size, hidden_dim)
        self.reference_atom_embed = nn.Embedding(atom_vocab_size, hidden_dim)
        self.time_embed = SimpleTimeEmbedding(time_embed_dim)
        self.extra_feat_dim = int(extra_feat_dim)
        self.query_feat_proj = nn.Linear(self.extra_feat_dim, hidden_dim) if self.extra_feat_dim > 0 else None
        self.reference_feat_proj = nn.Linear(self.extra_feat_dim, hidden_dim) if self.extra_feat_dim > 0 else None
        # [query_atom, time, ref_dir(3), ref_dist(1), pooled_ref_atom_feat]
        self.in_proj = nn.Linear(hidden_dim + time_embed_dim + 4 + hidden_dim, hidden_dim)

        self.blocks = nn.ModuleList([EquivariantBlock(hidden_dim) for _ in range(num_blocks)])
        # More expressive equivariant head than v = gate * x:
        # combine query-relative vector x and reference direction vector.
        self.out_scale_x = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.out_scale_ref = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def _reference_context(self, batch: AlignmentBatch) -> tuple[Tensor, Tensor, Tensor]:
        """Return reference-aware geometric + feature context for each query atom.

        Returns:
            direction: Unit vector from query atom to weighted reference anchor.
            distance: Norm of the query-to-anchor vector.
            ref_feat: Weighted pooled reference atom-type embedding.
        """
        if batch.reference_pos is None or batch.reference_batch is None or batch.reference_batch.numel() == 0:
            zero = torch.zeros_like(batch.query_pos)
            zeros_feat = torch.zeros(
                batch.query_pos.size(0), self.atom_embed.embedding_dim, device=batch.query_pos.device, dtype=batch.query_pos.dtype
            )
            return zero, torch.zeros(batch.query_pos.size(0), 1, device=batch.query_pos.device, dtype=batch.query_pos.dtype), zeros_feat

        # Fallback when reference atom types are not available.
        if batch.reference_atom_type is None:
            ref_atom_feat_all = torch.zeros(
                batch.reference_pos.size(0), self.atom_embed.embedding_dim, device=batch.reference_pos.device, dtype=batch.reference_pos.dtype
            )
        else:
            ref_atom_feat_all = self.reference_atom_embed(batch.reference_atom_type)
        if self.reference_feat_proj is not None and batch.reference_node_attr is not None:
            ref_atom_feat_all = ref_atom_feat_all + self.reference_feat_proj(batch.reference_node_attr)

        direction = torch.zeros_like(batch.query_pos)
        distance = torch.zeros(batch.query_pos.size(0), 1, device=batch.query_pos.device, dtype=batch.query_pos.dtype)
        ref_feat = torch.zeros(
            batch.query_pos.size(0), ref_atom_feat_all.size(-1), device=batch.query_pos.device, dtype=batch.query_pos.dtype
        )

        num_graphs = int(batch.query_batch.max().item()) + 1
        for g in range(num_graphs):
            q_idx = torch.where(batch.query_batch == g)[0]
            r_idx = torch.where(batch.reference_batch == g)[0]
            if q_idx.numel() == 0 or r_idx.numel() == 0:
                continue

            q = batch.query_pos[q_idx]   # [Nq, 3]
            r = batch.reference_pos[r_idx]  # [Nr, 3]
            rf = ref_atom_feat_all[r_idx]  # [Nr, H]

            # Query-reference soft assignment (cross-graph attention-like conditioning).
            d2 = torch.cdist(q, r, p=2) ** 2
            w = torch.softmax(-d2, dim=-1)  # [Nq, Nr]
            r_anchor = w @ r
            rf_anchor = w @ rf

            delta = r_anchor - q
            dist = safe_norm(delta, dim=-1, keepdim=True)
            direction[q_idx] = delta / dist.clamp_min(1e-8)
            distance[q_idx] = dist
            ref_feat[q_idx] = rf_anchor

        return direction, distance, ref_feat

    def forward(self, batch: AlignmentBatch, t_graph: Tensor) -> Tensor:
        """Predict query velocity field ``v_theta(x_t, t, cond)``.

        Args:
            batch: Alignment mini-batch.
            t_graph: Time per graph ``[B]``.

        Returns:
            Velocity vectors for query atoms ``[Nq, 3]``.
        """
        if batch.query_pos.numel() == 0:
            return torch.zeros_like(batch.query_pos)

        num_graphs = int(batch.query_batch.max().item()) + 1
        com = segment_mean(batch.query_pos, batch.query_batch, num_graphs)  # Per-graph center of mass.
        x = batch.query_pos - com[batch.query_batch]  # Translation-invariant coordinates.

        h_atom = self.atom_embed(batch.query_atom_type)  # Atom identity features.
        if self.query_feat_proj is not None and batch.query_node_attr is not None:
            h_atom = h_atom + self.query_feat_proj(batch.query_node_attr)
        h_t = self.time_embed(t_graph)[batch.query_batch]  # Time features expanded to nodes.
        ref_dir, ref_dist, ref_feat = self._reference_context(batch)  # Conditioning from reference geometry + atom types.
        h = self.in_proj(torch.cat([h_atom, h_t, ref_dir, ref_dist, ref_feat], dim=-1))  # Initial node states.

        edge_index = build_radius_edges(x, batch.query_batch, self.edge_cutoff, self.max_neighbors)
        for block in self.blocks:
            h, x = block(h, x, edge_index)

        scale_x = self.out_scale_x(h)
        scale_ref = self.out_scale_ref(h)
        v = scale_x * x + scale_ref * ref_dir
        return v
