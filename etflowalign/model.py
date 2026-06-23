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
    pocket_atom_type: Optional[Tensor] = None
    query_node_attr: Optional[Tensor] = None
    query_bond_index: Optional[Tensor] = None
    query_bond_length: Optional[Tensor] = None
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


def _clamp_vector_norm(v: Tensor, max_norm: float) -> Tensor:
    """Scale rows of ``v`` so each has norm <= ``max_norm`` (soft ceiling, no-op when below)."""
    norm = v.norm(dim=-1, keepdim=True)
    scale = (max_norm / norm.clamp_min(1e-8)).clamp(max=1.0)
    return v * scale


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
    """Intra-graph radius edges, fully vectorized (no per-atom Python loop).

    Edge ``(i, j)`` means atom ``j`` is a neighbor of atom ``i`` (message j->i). For each
    ``i`` only the ``max_neighbors`` nearest in-graph atoms within ``cutoff`` are kept.
    """
    n = pos.size(0)
    if n == 0:
        return torch.empty(2, 0, dtype=torch.long, device=pos.device)

    diff = pos[:, None, :] - pos[None, :, :]  # [n, n, 3]
    dist_sq = (diff * diff).sum(-1)  # [n, n]
    same = batch[:, None] == batch[None, :]
    eye = torch.eye(n, dtype=torch.bool, device=pos.device)
    mask = same & (~eye) & (dist_sq <= cutoff * cutoff)

    if 0 < max_neighbors < n:
        d = dist_sq.masked_fill(~mask, float("inf"))
        topk_idx = d.topk(min(max_neighbors, n), dim=1, largest=False).indices
        keep = torch.zeros_like(mask)
        keep.scatter_(1, topk_idx, True)
        mask = mask & keep

    i, j = mask.nonzero(as_tuple=True)
    return torch.stack([i, j], dim=0)

def build_cross_edges(
    lig_pos: Tensor,
    lig_batch: Tensor,
    pkt_pos: Tensor,
    pkt_batch: Tensor,
    cutoff: float,
    max_neighbors: int,
) -> tuple[Tensor, Tensor]:
    """Ligand->pocket edges within cutoff (same graph), fully vectorized.

    Returns ``(lig_idx, pkt_idx)``; for each ligand atom only the ``max_neighbors`` nearest
    in-graph pocket atoms within ``cutoff`` are kept.
    """
    empty = torch.empty(0, dtype=torch.long, device=lig_pos.device)
    if pkt_pos is None or pkt_pos.numel() == 0 or lig_pos.numel() == 0:
        return empty, empty

    diff = lig_pos[:, None, :] - pkt_pos[None, :, :]  # [nl, npk, 3]
    dist_sq = (diff * diff).sum(-1)  # [nl, npk]
    same = lig_batch[:, None] == pkt_batch[None, :]
    mask = same & (dist_sq <= cutoff * cutoff)

    npk = pkt_pos.size(0)
    if 0 < max_neighbors < npk:
        d = dist_sq.masked_fill(~mask, float("inf"))
        topk_idx = d.topk(min(max_neighbors, npk), dim=1, largest=False).indices
        keep = torch.zeros_like(mask)
        keep.scatter_(1, topk_idx, True)
        mask = mask & keep

    lig_idx, pkt_idx = mask.nonzero(as_tuple=True)
    return lig_idx, pkt_idx


class PocketInteraction(nn.Module):
    """Cross message passing so ligand atoms can sense binding-site shape.

    The pocket is fixed context (its atoms never move). For each ligand atom this
    produces (a) an invariant feature update aggregated from nearby pocket atoms, and
    (b) an equivariant pocket-shape vector ``sum_j w_ij (p_j - x_i)`` that gives the
    rigid head directional information about the pocket surface (not just its centroid).
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        # used as the pocket node feature only when pocket atom types are unavailable
        self.fallback_token = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h_lig: Tensor, x_lig: Tensor, x_pkt: Tensor, h_pkt: Tensor, lig_idx: Tensor, pkt_idx: Tensor) -> tuple[Tensor, Tensor]:
        """``h_pkt`` are per-pocket-atom features (atom-type embedding, or the fallback token)."""
        n_lig = h_lig.size(0)
        h_update = torch.zeros_like(h_lig)
        vec = torch.zeros(n_lig, 3, device=h_lig.device, dtype=h_lig.dtype)
        if lig_idx.numel() == 0:
            return h_update, vec

        rij = x_pkt[pkt_idx] - x_lig[lig_idx]  # [E, 3], ligand -> pocket
        dij = safe_norm(rij, dim=-1, keepdim=True)  # [E, 1]
        feat = torch.cat([h_lig[lig_idx], h_pkt[pkt_idx], dij], dim=-1)

        # Mean aggregation + sigmoid gate keep both outputs bounded regardless of how many
        # pocket atoms a ligand atom sees (real pockets have 100-200 atoms). A plain sum
        # made the pocket-shape vector explode -> huge rigid velocity -> NaN.
        ones = torch.ones(lig_idx.size(0), 1, device=h_lig.device, dtype=h_lig.dtype)
        deg = torch.zeros(n_lig, 1, device=h_lig.device, dtype=h_lig.dtype)
        deg.index_add_(0, lig_idx, ones)
        deg = deg.clamp_min(1.0)
        
        h_update.index_add_(0, lig_idx, self.msg_mlp(feat))
        gate = torch.sigmoid(self.gate_mlp(feat))  # (0, 1): bounds each edge's vector contribution
        vec.index_add_(0, lig_idx, gate * rij)     # |contribution| <= cutoff
        return self.norm(h_update / deg), vec / deg

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
        dij = safe_norm(rij, dim=-1, keepdim=True)  # safe_norm: finite gradient when rij == 0

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
        num_blocks: int = 4,
        edge_cutoff: float = 6.0,
        max_neighbors: int = 32,
        use_atom_index_embed: bool = False,
        use_equivariant_basis_head: bool = False,
        use_rigid_head: bool = False,
        use_pocket_conditioning: bool = False,
        pocket_cutoff: float = 8.0,
        pocket_max_neighbors: int = 16,
        use_node_attr: bool = False,
        node_attr_dim: int = 5,
        max_atoms: int = 256,
        **_deprecated_kwargs: object,  # 구버전 체크포인트의 제거된 인자(use_direct_vector_head 등)를 흡수
    ) -> None:
        super().__init__()
        self.edge_cutoff = float(edge_cutoff)
        self.max_neighbors = int(max_neighbors)

        self.atom_embed = nn.Embedding(atom_vocab_size, hidden_dim)
        self.use_atom_index_embed = bool(use_atom_index_embed)
        self.use_equivariant_basis_head = bool(use_equivariant_basis_head)
        self.use_rigid_head = bool(use_rigid_head)
        self.use_pocket_conditioning = bool(use_pocket_conditioning)
        self.pocket_cutoff = float(pocket_cutoff)
        self.pocket_max_neighbors = int(pocket_max_neighbors)
        self.max_atoms = int(max_atoms)

        if self.use_pocket_conditioning:
            self.pocket_interaction = PocketInteraction(hidden_dim)

        # rigid head gains a 5th basis (pocket-shape vector) when pocket conditioning is on
        self._n_rigid_bases = 5 if self.use_pocket_conditioning else 4
        
        if self.use_atom_index_embed:
            self.atom_index_embed = nn.Embedding(self.max_atoms, hidden_dim)

        self.time_embed = SimpleTimeEmbedding(time_embed_dim)
        self.in_proj = nn.Linear(hidden_dim + time_embed_dim + 4, hidden_dim)

        self.blocks = nn.ModuleList([EquivariantBlock(hidden_dim) for _ in range(num_blocks)])

        # production-호환 expressive head:
        #   v_i = sum_k a_{ik} b_{ik}
        # where b_{ik} are equivariant vector bases and a_{ik} are invariant scalars.
        if self.use_equivariant_basis_head:
            self.out_basis_coeff = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 4),
            )
        # Rigid (direction A) head:
        #   v_i = omega_g x (x_i - c_g) + v_lin_g
        # A per-graph rigid-body velocity field (6 DOF per molecule). Because the field is
        # a rigid motion by construction, integrating it cannot change any intramolecular
        # distance: bond/angle geometry of the source conformer is preserved exactly.
        # omega_g and v_lin_g are built as graph-pooled combinations of equivariant vector
        # bases, so the field rotates with the input (E(3)-equivariant up to the reference
        # direction conditioning in `in_proj`).
        if self.use_rigid_head:
            self.out_rigid_omega = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, self._n_rigid_bases),
            )
            self.out_rigid_vlin = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, self._n_rigid_bases),
            )
            
    def _reference_context(self, batch: AlignmentBatch) -> tuple[Tensor, Tensor]:
        """Return direction and distance from query atoms to reference center."""
        if batch.reference_pos is None or batch.reference_batch is None or batch.reference_batch.numel() == 0:
            zero = torch.zeros_like(batch.query_pos)
            return zero, torch.zeros(batch.query_pos.size(0), 1, device=batch.query_pos.device, dtype=batch.query_pos.dtype)

        num_graphs = int(batch.query_batch.max().item()) + 1
        ref_center = _segment_mean(batch.reference_pos, batch.reference_batch, num_graphs)
        delta = ref_center[batch.query_batch] - batch.query_pos
        dist = safe_norm(delta, dim=-1, keepdim=True)
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
        """Return query-query neighbor aggregate vector basis (mean over neighbors).

        Basis:
            b_nbr,i = mean_{j in N(i)} (x_j - x_i)

        Mean (not sum) keeps the magnitude bounded regardless of neighbor count; an
        unbounded sum was a source of exploding omega -> NaN/huge loss on real data.
        Translation-invariant and rotation-equivariant.
        """
        if edge_index.numel() == 0:
            return torch.zeros_like(x)

        i, j = edge_index[0], edge_index[1]
        rij = x[j] - x[i]
        out = torch.zeros_like(x)
        out.index_add_(0, i, rij)
        deg = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
        deg.index_add_(0, i, torch.ones(i.size(0), 1, device=x.device, dtype=x.dtype))
        return out / deg.clamp_min(1.0)

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

        basis_query = x  # equivariant block을 거친 query 좌표
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
    def _rigid_head(
        self,
        h: Tensor,
        x: Tensor,
        x_in: Tensor,
        batch: AlignmentBatch,
        edge_index: Tensor,
        pocket_basis: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict a per-graph rigid-body velocity field (direction A).

        Steps:
            1. Build equivariant per-atom vector bases (same set as the basis head).
            2. Form per-atom omega/v_lin as invariant-scalar-weighted basis sums.
            3. Pool to one omega_g and v_lin_g per graph (mean preserves equivariance).
            4. Expand to a rigid field on the centered input coordinates:
                   v_i = omega_g x (x_in_i - c_g) + v_lin_g

        ``x_in`` are the centered *input* query coordinates (not the block-updated ``x``),
        so the returned field is an exact rigid motion of the actual sampler state. The
        equivariant bases are also built from ``x_in`` (and reference/pocket geometry):
        the block-updated ``x`` is unnormalized and can grow across blocks, which makes
        ``omega`` explode and the loss ill-conditioned. ``h`` still carries the learned
        message-passing context as invariant coefficients.
        """
        num_graphs = int(batch.query_batch.max().item()) + 1

        basis_list = [
            x_in,                                       # query-centered coordinate
            self._reference_delta_basis(batch),         # reference centroid -> query
            self._pocket_delta_basis(batch),            # pocket centroid -> query
            self._neighbor_vector_basis(x_in, edge_index),  # query neighbor aggregate
        ]
        if self.use_pocket_conditioning:
            # pocket-shape vector (directional surface info, not just centroid)
            basis_list.append(pocket_basis if pocket_basis is not None else torch.zeros_like(x_in))
        bases = torch.stack(basis_list, dim=1)  # [N, n_bases, 3]

        omega_atom = (self.out_rigid_omega(h).unsqueeze(-1) * bases).sum(dim=1)  # [N, 3]
        vlin_atom = (self.out_rigid_vlin(h).unsqueeze(-1) * bases).sum(dim=1)  # [N, 3]

        # Bound the per-graph rigid velocity so a large ligand or a transient weight blow-up
        # cannot produce an exploding velocity (-> NaN/huge loss). A valid geodesic rotation
        # has |omega| <= pi, so the 2*pi cap never touches legitimate targets.
        omega = _clamp_vector_norm(_segment_mean(omega_atom, batch.query_batch, num_graphs), 2.0 * torch.pi)
        vlin = _clamp_vector_norm(_segment_mean(vlin_atom, batch.query_batch, num_graphs), 30.0)

        com = _segment_mean(x_in, batch.query_batch, num_graphs)  # [G, 3]
        rel = x_in - com[batch.query_batch]  # [N, 3]
        omega_i = omega[batch.query_batch]  # [N, 3]
        vlin_i = vlin[batch.query_batch]  # [N, 3]

        return torch.cross(omega_i, rel, dim=-1) + vlin_i
    
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
        x_in = x.clone()  # centered input coords, used by the rigid head
        
        h_atom = self.atom_embed(batch.query_atom_type)

        if self.use_atom_index_embed:
            local_index = self._local_atom_index(batch.query_batch)
            if int(local_index.max().item()) >= self.max_atoms:
                raise ValueError(f"local atom index exceeds max_atoms={self.max_atoms}; increase --max-atoms.")
            h_atom = h_atom + self.atom_index_embed(local_index)

        h_t = self.time_embed(t_graph)[batch.query_batch]
        ref_dir, ref_dist = self._reference_context(batch)

        h = self.in_proj(torch.cat([h_atom, h_t, ref_dir, ref_dist], dim=-1))

        # Pocket-shape conditioning: enrich ligand features and build a directional
        # pocket-shape basis from the fixed binding-site atoms (in the ligand-COM frame).
        pocket_basis = None
        if (
            self.use_pocket_conditioning
            and batch.pocket_pos is not None
            and batch.pocket_batch is not None
            and batch.pocket_batch.numel() > 0
        ):
            xp = batch.pocket_pos - com[batch.pocket_batch]
            if batch.pocket_atom_type is not None:
                h_pkt = self.atom_embed(batch.pocket_atom_type)  # pocket chemistry (shared atom embedding)
            else:
                h_pkt = self.pocket_interaction.fallback_token.unsqueeze(0).expand(xp.size(0), -1)
            lig_idx, pkt_idx = build_cross_edges(
                x_in, batch.query_batch, xp, batch.pocket_batch,
                self.pocket_cutoff, self.pocket_max_neighbors,
            )
            h_pocket, pocket_basis = self.pocket_interaction(h, x_in, xp, h_pkt, lig_idx, pkt_idx)
            h = h + h_pocket
        
        edge_index = build_radius_edges(x, batch.query_batch, self.edge_cutoff, self.max_neighbors)

        for block in self.blocks:
            h, x = block(h, x, edge_index)

        # 방향 A: rigid-body 속도장 (적분해도 분자 내부 기하가 정확히 보존됨)
        if self.use_rigid_head:
            return self._rigid_head(h, x, x_in, batch, edge_index, pocket_basis)

        # production-호환 equivariant basis head
        if self.use_equivariant_basis_head:
            return self._equivariant_basis_head(h, x, batch, edge_index)

        raise RuntimeError(
            "출력 head가 선택되지 않았습니다: use_rigid_head 또는 use_equivariant_basis_head 중 하나를 켜세요."
        )