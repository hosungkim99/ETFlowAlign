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
    pocket_batch: Optional[Tensor] = None
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


def _segment_mean(x: Tensor, batch: Tensor, num_graphs: int) -> Tensor:
    """Compute per-graph mean of node features/coordinates.

    Args:
        x: Node tensor ``[N, D]``.
        batch: Graph index per node ``[N]``.
        num_graphs: Number of graphs in the mini-batch.

    Returns:
        Mean tensor per graph ``[num_graphs, D]``.
    """
    out = torch.zeros(num_graphs, x.size(-1), device=x.device, dtype=x.dtype)
    cnt = torch.zeros(num_graphs, 1, device=x.device, dtype=x.dtype)
    out.index_add_(0, batch, x)
    cnt.index_add_(0, batch, torch.ones_like(batch, dtype=x.dtype).unsqueeze(-1))
    return out / cnt.clamp_min(1.0)


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
        mask = (dist_sq <= cutoff_sq) & (
            ~torch.eye(node_idx.numel(), device=pos.device, dtype=torch.bool)
        )

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


class TorchMDEquivariantTransformerBlock(nn.Module):
    """TorchMD-NET style lightweight equivariant transformer block."""

    def __init__(self, hidden_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = int(max(1, num_heads))
        self.head_dim = hidden_dim // self.num_heads
        if self.head_dim == 0 or self.head_dim * self.num_heads != hidden_dim:
            raise ValueError("hidden_dim must be divisible by num_heads for torchmd_et backbone.")

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dist_proj = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, self.num_heads))
        self.coord_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.SiLU(), nn.Linear(hidden_dim * 2, hidden_dim))

    def forward(self, h: Tensor, x: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        if edge_index.numel() == 0:
            return h, x

        i, j = edge_index[0], edge_index[1]
        rij = x[i] - x[j]
        dij = torch.norm(rij, dim=-1, keepdim=True).clamp_min(1e-8)

        q = self.q_proj(h).view(h.size(0), self.num_heads, self.head_dim)
        k = self.k_proj(h).view(h.size(0), self.num_heads, self.head_dim)
        v = self.v_proj(h).view(h.size(0), self.num_heads, self.head_dim)

        q_i = q[i]
        k_j = k[j]
        v_j = v[j]
        dist_bias = self.dist_proj(dij)  # [E, heads]
        attn_logits = (q_i * k_j).sum(dim=-1) / (self.head_dim**0.5) + dist_bias

        # Segment softmax over incoming edges for each destination node i.
        attn = torch.zeros_like(attn_logits)
        n_nodes = h.size(0)
        for node in range(n_nodes):
            mask = i == node
            if mask.any():
                attn[mask] = torch.softmax(attn_logits[mask], dim=0)

        msg = attn.unsqueeze(-1) * v_j
        agg = torch.zeros(h.size(0), self.num_heads, self.head_dim, device=h.device, dtype=h.dtype)
        agg.index_add_(0, i, msg)
        agg = agg.reshape(h.size(0), -1)

        h = h + self.o_proj(agg)
        h = self.norm(h + self.ffn(h))

        # Equivariant coordinate update from attended scalar messages.
        gate = self.coord_gate(agg[i])
        dx_msg = gate * (rij / dij)
        dx = torch.zeros_like(x)
        dx.index_add_(0, i, dx_msg)
        x = x + dx
        return h, x


class ETFlowAlignModel(nn.Module):
    """Compact E(3)-equivariant time-dependent vector field model for query atoms."""

    def __init__(
        self,
        atom_vocab_size: int = 128,
        hidden_dim: int = 128,
        time_embed_dim: int = 64,
        num_blocks: int = 4,
        edge_cutoff: float = 6.0,
        max_neighbors: int = 32,
        use_atom_index_embed: bool = False,
        use_direct_vector_head: bool = False,
        use_equivariant_basis_head: bool = False,
        max_atoms: int = 256,
    ) -> None:
        super().__init__()
        self.edge_cutoff = float(edge_cutoff)
        self.max_neighbors = int(max_neighbors)

        self.atom_embed = nn.Embedding(atom_vocab_size, hidden_dim)
        self.use_atom_index_embed = bool(use_atom_index_embed)
        self.use_direct_vector_head = bool(use_direct_vector_head)
        self.use_equivariant_basis_head = bool(use_equivariant_basis_head)
        self.max_atoms = int(max_atoms)

        if self.use_atom_index_embed:
            self.atom_index_embed = nn.Embedding(self.max_atoms, hidden_dim)

        self.time_embed = SimpleTimeEmbedding(time_embed_dim)
        self.in_proj = nn.Linear(hidden_dim + time_embed_dim + 4, hidden_dim)

        self.blocks = nn.ModuleList([EquivariantBlock(hidden_dim) for _ in range(num_blocks)])

        # Legacy production head:
        #   v_i = alpha_i x_i
        # This is E(3)-equivariant but too restrictive for many alignment flows.
        self.out_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Debug head:
        #   v_i = MLP(h_i)
        # This is expressive and useful for overfit sanity checks, but not E(3)-equivariant.
        if self.use_direct_vector_head:
            self.out_vec = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 3),
            )

        # Production-compatible expressive head:
        #   v_i = sum_k a_{ik} b_{ik}
        # where b_{ik} are equivariant vector bases and a_{ik} are invariant scalars.
        if self.use_equivariant_basis_head:
            self.out_basis_coeff = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 4),
            )

    def _reference_context(self, batch: AlignmentBatch) -> tuple[Tensor, Tensor]:
        """Return direction and distance from query atoms to reference center."""
        if batch.reference_pos is None or batch.reference_batch is None or batch.reference_batch.numel() == 0:
            zero = torch.zeros_like(batch.query_pos)
            return zero, torch.zeros(batch.query_pos.size(0), 1, device=batch.query_pos.device, dtype=batch.query_pos.dtype)

        num_graphs = int(batch.query_batch.max().item()) + 1
        ref_center = _segment_mean(batch.reference_pos, batch.reference_batch, num_graphs)
        delta = ref_center[batch.query_batch] - batch.query_pos
        dist = torch.norm(delta, dim=-1, keepdim=True)
        direction = delta / dist.clamp_min(1e-8)
        return direction, dist

    def _reference_delta_basis(self, batch: AlignmentBatch) -> Tensor:
        """Return reference-center-to-query vector basis.

        Basis:
            b_ref,i = c_ref[graph(i)] - q_i

        This vector is translation-invariant and rotation-equivariant.
        """
        if batch.reference_pos is None or batch.reference_batch is None or batch.reference_batch.numel() == 0:
            return torch.zeros_like(batch.query_pos)

        num_graphs = int(batch.query_batch.max().item()) + 1
        ref_center = _segment_mean(batch.reference_pos, batch.reference_batch, num_graphs)
        return ref_center[batch.query_batch] - batch.query_pos

    def _pocket_delta_basis(self, batch: AlignmentBatch) -> Tensor:
        """Return pocket-center-to-query vector basis.

        Basis:
            b_pocket,i = c_pocket[graph(i)] - q_i

        If pocket coordinates are missing, return zero vectors.
        """
        if batch.pocket_pos is None or batch.pocket_batch is None or batch.pocket_batch.numel() == 0:
            return torch.zeros_like(batch.query_pos)

        num_graphs = int(batch.query_batch.max().item()) + 1
        pocket_center = _segment_mean(batch.pocket_pos, batch.pocket_batch, num_graphs)
        return pocket_center[batch.query_batch] - batch.query_pos

    def _neighbor_vector_basis(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Return query-query neighbor aggregate vector basis.

        Basis:
            b_nbr,i = sum_{j in N(i)} (x_j - x_i)

        This vector is translation-invariant and rotation-equivariant.
        """
        if edge_index.numel() == 0:
            return torch.zeros_like(x)

        i, j = edge_index[0], edge_index[1]
        rij = x[j] - x[i]
        out = torch.zeros_like(x)
        out.index_add_(0, i, rij)
        return out

    def _local_atom_index(self, query_batch: Tensor) -> Tensor:
        """Return graph-local atom indices for optional debug atom-index embeddings."""
        local_index = torch.zeros_like(query_batch)
        for g in query_batch.unique(sorted=True):
            mask = query_batch == g
            n = int(mask.sum().item())
            local_index[mask] = torch.arange(n, device=query_batch.device, dtype=query_batch.dtype)
        return local_index

    def _equivariant_basis_head(self, h: Tensor, x: Tensor, batch: AlignmentBatch, edge_index: Tensor) -> Tensor:
        """Predict velocity by combining equivariant vector bases.

        The scalar coefficients are invariant because they are predicted from scalar node
        features. The bases rotate with the input coordinates. Their weighted sum is
        therefore E(3)-equivariant.

        Bases:
            0. x: query-centered coordinate basis after equivariant blocks.
            1. reference delta: reference center - query atom.
            2. pocket delta: pocket center - query atom.
            3. neighbor aggregate: sum_j (x_j - x_i).
        """
        coeff = self.out_basis_coeff(h)  # [N, 4]

        basis_query = x
        basis_reference = self._reference_delta_basis(batch)
        basis_pocket = self._pocket_delta_basis(batch)
        basis_neighbor = self._neighbor_vector_basis(x, edge_index)

        bases = torch.stack(
            [
                basis_query,
                basis_reference,
                basis_pocket,
                basis_neighbor,
            ],
            dim=1,
        )  # [N, 4, 3]

        return (coeff.unsqueeze(-1) * bases).sum(dim=1)

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
        com = _segment_mean(batch.query_pos, batch.query_batch, num_graphs)
        x = batch.query_pos - com[batch.query_batch]

        h_atom = self.atom_embed(batch.query_atom_type)

        if self.use_atom_index_embed:
            local_index = self._local_atom_index(batch.query_batch)
            if int(local_index.max().item()) >= self.max_atoms:
                raise ValueError(f"local atom index exceeds max_atoms={self.max_atoms}; increase --max-atoms.")
            h_atom = h_atom + self.atom_index_embed(local_index)

        h_t = self.time_embed(t_graph)[batch.query_batch]
        ref_dir, ref_dist = self._reference_context(batch)

        h = self.in_proj(torch.cat([h_atom, h_t, ref_dir, ref_dist], dim=-1))

        edge_index = build_radius_edges(x, batch.query_batch, self.edge_cutoff, self.max_neighbors)

        for block in self.blocks:
            h, x = block(h, x, edge_index)

        # Highest priority: debug direct head.
        # This is intentionally non-equivariant and should be used for sanity checks only.
        if self.use_direct_vector_head:
            return self.out_vec(h)

        # Production-compatible expressive equivariant head.
        if self.use_equivariant_basis_head:
            return self._equivariant_basis_head(h, x, batch, edge_index)

        # Legacy restrictive equivariant head.
        gate = self.out_gate(h)
        v = gate * x
        return v
