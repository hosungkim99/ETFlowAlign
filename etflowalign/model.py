"""ETFlowAlign 핵심 모델 정의.

이 모듈은 정렬을 위한 최소한의 구체적인 E(3)-등변 벡터 필드 모델을 구현한다.
아키텍처는 외부 래퍼 없이 이 저장소만으로 전체 설계를 이해할 수 있도록
의도적으로 간결하게 구성되어 있다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from .utils import safe_norm, segment_mean


@dataclass
class AlignmentBatch:
    """ETFlowAlign를 위한 최소한의 독립형 배치 컨테이너.

    Attributes:
        query_pos: 쿼리 리간드 좌표, 형상 ``[Nq, 3]``.
        query_atom_type: 쿼리 원자의 정수형 원자 타입 ID ``[Nq]``.
        query_batch: 각 쿼리 원자의 그래프 인덱스 ``[Nq]``.
        reference_pos: 레퍼런스 리간드 좌표 ``[Nr, 3]``.
        reference_atom_type: 선택적 레퍼런스 원자 타입 ID ``[Nr]``.
        reference_batch: 각 레퍼런스 원자의 그래프 인덱스 ``[Nr]``.
        pocket_pos: 선택적 포켓/수용체 포인트 좌표.
    """

    query_pos: Tensor
    query_atom_type: Tensor
    query_batch: Tensor
    reference_pos: Optional[Tensor] = None
    reference_atom_type: Optional[Tensor] = None
    reference_batch: Optional[Tensor] = None
    pocket_pos: Optional[Tensor] = None
    pocket_batch: Optional[Tensor] = None
    pocket_atom_Type: Optional[Tensor] = None
    query_node_attr: Optional[Tensor] = None
    query_bond_index: Optional[Tensor] = None
    query_bond_length: Optional[Tensor] = None
    reference_node_attr: Optional[Tensor] = None


class SimpleTimeEmbedding(nn.Module):
    """[0, 1] 구간의 연속 플로우 시간 t에 대한 사인 임베딩."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: Tensor) -> Tensor:
        """각 그래프에 대한 플로우 시간을 임베딩한다.

        Args:
            t: ``[0, 1]`` 구간의 연속 시간, 형상 ``[B]``.

        Returns:
            형상 ``[B, dim]``의 시간 임베딩.
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
    """노드 피처/좌표의 그래프별 평균을 계산한다.

    Args:
        x: 노드 텐서 ``[N, D]``.
        batch: 노드별 그래프 인덱스 ``[N]``.
        num_graphs: 미니배치 내 그래프 수.

    Returns:
        그래프별 평균 텐서 ``[num_graphs, D]``.
    """
    out = torch.zeros(num_graphs, x.size(-1), device=x.device, dtype=x.dtype)
    cnt = torch.zeros(num_graphs, 1, device=x.device, dtype=x.dtype)
    out.index_add_(0, batch, x)
    cnt.index_add_(0, batch, torch.ones_like(batch, dtype=x.dtype).unsqueeze(-1))
    return out / cnt.clamp_min(1.0)


def build_radius_edges(pos: Tensor, batch: Tensor, cutoff: float, max_neighbors: int) -> Tensor:
    """완전히 벡터화된 그래프 내 반경 에지 생성 (원자별 Python 루프 없음).

    에지 ``(i, j)``는 원자 ``j``가 원자 ``i``의 이웃임을 의미한다 (메시지 j->i).
    각 ``i``에 대해 ``cutoff`` 이내의 그래프 내 가장 가까운 ``max_neighbors``개 원자만 유지한다.
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
    """컷오프 이내의 리간드->포켓 에지 생성 (동일 그래프), 완전히 벡터화됨.

    ``(lig_idx, pkt_idx)``를 반환한다; 각 리간드 원자에 대해 ``cutoff`` 이내의
    그래프 내 가장 가까운 ``max_neighbors``개 포켓 원자만 유지한다.
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
    """리간드 원자가 결합 부위의 형태를 감지할 수 있도록 하는 크로스 메시지 패싱.

    포켓은 고정된 컨텍스트이다 (원자가 절대 이동하지 않음). 각 리간드 원자에 대해
    (a) 인근 포켓 원자로부터 집계된 불변 피처 업데이트와,
    (b) 포켓 표면에 대한 방향 정보를 rigid head에 제공하는 등변 포켓 형태 벡터
    ``sum_j w_ij (p_j - x_i)`` (단순 중심이 아닌)를 생성한다.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        # 포켓 원자 타입을 사용할 수 없을 때 포켓 노드 피처로 사용되는 폴백 토큰
        self.fallback_token = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h_lig: Tensor, x_lig: Tensor, x_pkt: Tensor, h_pkt: Tensor, lig_idx: Tensor, pkt_idx: Tensor) -> tuple[Tensor, Tensor]:
        """``h_pkt``는 포켓 원자별 피처이다 (원자 타입 임베딩 또는 폴백 토큰)."""
        n_lig = h_lig.size(0)
        h_update = torch.zeros_like(h_lig)
        vec = torch.zeros(n_lig, 3, device=h_lig.device, dtype=h_lig.dtype)
        if lig_idx.numel() == 0:
            return h_update, vec

        rij = x_pkt[pkt_idx] - x_lig[lig_idx]  # [E, 3], ligand -> pocket
        dij = safe_norm(rij, dim=-1, keepdim=True)  # [E, 1]
        tok = self.pocket_token.unsqueeze(0).expand(lig_idx.size(0), -1)
        feat = torch.cat([h_lig[lig_idx], tok, dij], dim=-1)

        # 평균 집계 + 시그모이드 게이트는 리간드 원자가 보는 포켓 원자 수에 관계없이
        # 두 출력을 유한 범위 내로 유지한다 (실제 포켓은 100~200개 원자를 가짐).
        # 단순 합산을 사용하면 포켓 형태 벡터가 발산 -> rigid 속도 폭발 -> NaN 발생.
        ones = torch.ones(lig_idx.size(0), 1, device=h_lig.device, dtype=h_lig.dtype)
        deg = torch.zeros(n_lig, 1, device=h_lig.device, dtype=h_lig.dtype)
        deg.index_add_(0, lig_idx, ones)
        deg = deg.clamp_min(1.0)
        
        h_update.index_add_(0, lig_idx, self.msg_mlp(feat))
        gate = torch.sigmoid(self.gate_mlp(feat))  # (0, 1): 각 에지의 벡터 기여를 범위 내로 제한
        vec.index_add_(0, lig_idx, gate * rij)     # |기여| <= cutoff
        return self.norm(h_update / deg), vec / deg

class EquivariantBlock(nn.Module):
    """단순 EGNN 스타일 블록: 스칼라 메시지 + 상대 벡터 집계."""

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
        """등변 메시지 패싱 블록 한 번을 실행한다.

        Args:
            h: 노드 스칼라 피처 ``[N, H]``.
            x: 노드 좌표 ``[N, 3]``.
            edge_index: 방향성 에지 ``[2, E]``.

        Returns:
            업데이트된 노드 피처 및 좌표.
        """
        if edge_index.numel() == 0:
            return h, x

        i, j = edge_index[0], edge_index[1]
        rij = x[i] - x[j]
        dij = safe_norm(rij, dim=-1, keepdim=True)  # safe_norm: rij == 0일 때도 유한한 기울기를 보장

        e_ij = self.phi_e(torch.cat([h[i], h[j], dij], dim=-1))

        # 좌표 업데이트 (등변): sum alpha_ij * (x_i - x_j)
        alpha_ij = self.phi_x(e_ij)
        dx_msg = alpha_ij * rij
        dx = torch.zeros_like(x)
        dx.index_add_(0, i, dx_msg)
        x = x + dx

        # 피처 업데이트: 노드 i로 메시지 집계
        m = torch.zeros_like(h)
        m.index_add_(0, i, e_ij)
        h = h + self.phi_h(torch.cat([h, m], dim=-1))
        return h, x


class TorchMDEquivariantTransformerBlock(nn.Module):
    """TorchMD-NET 스타일의 경량 등변 트랜스포머 블록."""

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

        # 각 목적지 노드 i에 대한 수신 에지의 세그먼트 소프트맥스.
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

        # 어텐션된 스칼라 메시지로부터의 등변 좌표 업데이트.
        gate = self.coord_gate(agg[i])
        dx_msg = gate * (rij / dij)
        dx = torch.zeros_like(x)
        dx.index_add_(0, i, dx_msg)
        x = x + dx
        return h, x


class ETFlowAlignModel(nn.Module):
    """쿼리 원자를 위한 간결한 E(3)-등변 시간 의존 벡터 필드 모델."""

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
        use_rigid_head: bool = False,
        use_pocket_conditioning: bool = False,
        pocket_cutoff: float = 8.0,
        pocket_max_neighbors: int = 16,
        use_node_attr: bool = False,
        node_attr_dim: int = 5,
        max_atoms: int = 256,
    ) -> None:
        super().__init__()
        self.edge_cutoff = float(edge_cutoff)
        self.max_neighbors = int(max_neighbors)

        self.atom_embed = nn.Embedding(atom_vocab_size, hidden_dim)
        self.use_atom_index_embed = bool(use_atom_index_embed)
        self.use_direct_vector_head = bool(use_direct_vector_head)
        self.use_equivariant_basis_head = bool(use_equivariant_basis_head)
        self.use_rigid_head = bool(use_rigid_head)
        self.use_pocket_conditioning = bool(use_pocket_conditioning)
        self.pocket_cutoff = float(pocket_cutoff)
        self.pocket_max_neighbors = int(pocket_max_neighbors)
        self.max_atoms = int(max_atoms)

        if self.use_pocket_conditioning:
            self.pocket_interaction = PocketInteraction(hidden_dim)

        # 포켓 컨디셔닝이 활성화되면 rigid head는 5번째 기저(포켓 형태 벡터)를 추가로 가짐
        self._n_rigid_bases = 5 if self.use_pocket_conditioning else 4
        
        if self.use_atom_index_embed:
            self.atom_index_embed = nn.Embedding(self.max_atoms, hidden_dim)

        self.time_embed = SimpleTimeEmbedding(time_embed_dim)
        self.in_proj = nn.Linear(hidden_dim + time_embed_dim + 4, hidden_dim)

        self.blocks = nn.ModuleList([EquivariantBlock(hidden_dim) for _ in range(num_blocks)])

        # 기존 프로덕션 헤드:
        #   v_i = alpha_i x_i
        # E(3)-등변이지만 많은 정렬 플로우에 대해 너무 제약적이다.
        self.out_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # 디버그 헤드:
        #   v_i = MLP(h_i)
        # 표현력이 높고 과적합 검증에 유용하지만 E(3)-등변이 아니다.
        if self.use_direct_vector_head:
            self.out_vec = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 3),
            )

        # 프로덕션 호환 표현력 있는 헤드:
        #   v_i = sum_k a_{ik} b_{ik}
        # 여기서 b_{ik}는 등변 벡터 기저이고 a_{ik}는 불변 스칼라이다.
        if self.use_equivariant_basis_head:
            self.out_basis_coeff = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 4),
            )
        # Rigid (direction A) 헤드:
        #   v_i = omega_g x (x_i - c_g) + v_lin_g
        # 그래프별 강체 속도 필드 (분자당 6 자유도). 필드가 구조적으로 강체 운동이므로,
        # 이를 적분해도 분자 내 거리가 전혀 변하지 않는다: 원본 형태의 결합/각도 구조가
        # 정확히 보존된다. omega_g와 v_lin_g는 등변 벡터 기저의 그래프 풀링 조합으로
        # 구성되므로, 필드는 입력과 함께 회전한다 (`in_proj`의 레퍼런스 방향 컨디셔닝까지
        # E(3)-등변).
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
        """쿼리 원자로부터 레퍼런스 중심까지의 방향과 거리를 반환한다."""
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
        """레퍼런스 중심에서 쿼리 원자까지의 벡터 기저를 반환한다.

        기저:
            b_ref,i = c_ref[graph(i)] - q_i

        이 벡터는 평행 이동 불변이며 회전 등변이다.
        """
        if batch.reference_pos is None or batch.reference_batch is None or batch.reference_batch.numel() == 0:
            return torch.zeros_like(batch.query_pos)

        num_graphs = int(batch.query_batch.max().item()) + 1
        ref_center = _segment_mean(batch.reference_pos, batch.reference_batch, num_graphs)
        return ref_center[batch.query_batch] - batch.query_pos

    def _pocket_delta_basis(self, batch: AlignmentBatch) -> Tensor:
        """포켓 중심에서 쿼리 원자까지의 벡터 기저를 반환한다.

        기저:
            b_pocket,i = c_pocket[graph(i)] - q_i

        포켓 좌표가 없으면 영 벡터를 반환한다.
        """
        if batch.pocket_pos is None or batch.pocket_batch is None or batch.pocket_batch.numel() == 0:
            return torch.zeros_like(batch.query_pos)

        num_graphs = int(batch.query_batch.max().item()) + 1
        pocket_center = _segment_mean(batch.pocket_pos, batch.pocket_batch, num_graphs)
        return pocket_center[batch.query_batch] - batch.query_pos

    def _neighbor_vector_basis(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """쿼리-쿼리 이웃 집계 벡터 기저를 반환한다 (이웃에 대한 평균).

        기저:
            b_nbr,i = mean_{j in N(i)} (x_j - x_i)

        합산이 아닌 평균을 사용하면 이웃 수에 관계없이 크기가 유한하게 유지된다;
        경계 없는 합산은 실제 데이터에서 omega 폭발 -> NaN/거대한 손실의 원인이었다.
        평행 이동 불변이며 회전 등변이다.
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
        """선택적 디버그 원자 인덱스 임베딩을 위한 그래프 내 로컬 원자 인덱스를 반환한다."""
        local_index = torch.zeros_like(query_batch)
        for g in query_batch.unique(sorted=True):
            mask = query_batch == g
            n = int(mask.sum().item())
            local_index[mask] = torch.arange(n, device=query_batch.device, dtype=query_batch.dtype)
        return local_index

    def _equivariant_basis_head(self, h: Tensor, x: Tensor, batch: AlignmentBatch, edge_index: Tensor) -> Tensor:
        """등변 벡터 기저를 조합하여 속도를 예측한다.

        스칼라 계수는 스칼라 노드 피처로부터 예측되므로 불변이다. 기저는 입력 좌표와
        함께 회전한다. 따라서 가중 합산은 E(3)-등변이다.

        기저:
            0. x: 등변 블록 이후 쿼리 중심 좌표 기저.
            1. reference delta: 레퍼런스 중심 - 쿼리 원자.
            2. pocket delta: 포켓 중심 - 쿼리 원자.
            3. neighbor aggregate: sum_j (x_j - x_i).
        """
        coeff = self.out_basis_coeff(h)  # [N, 4]

        basis_query = x_in
        basis_reference = self._reference_delta_basis(batch)
        basis_pocket = self._pocket_delta_basis(batch)
        basis_neighbor = self._neighbor_vector_basis(x_in, edge_index)

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
        """그래프별 강체 속도 필드를 예측한다 (direction A).

        단계:
            1. 원자별 등변 벡터 기저를 구성한다 (basis head와 동일한 집합).
            2. 불변 스칼라 가중 기저 합산으로 원자별 omega/v_lin을 구성한다.
            3. 그래프당 하나의 omega_g와 v_lin_g로 풀링한다 (평균이 등변성을 보존).
            4. 중심화된 입력 좌표에 대한 강체 필드로 확장한다:
                   v_i = omega_g x (x_in_i - c_g) + v_lin_g

        ``x_in``은 중심화된 *입력* 쿼리 좌표이다 (블록 업데이트된 ``x``가 아님).
        따라서 반환된 필드는 실제 샘플러 상태의 정확한 강체 운동이다. 등변 기저도
        ``x_in`` (및 레퍼런스/포켓 기하)로부터 구성된다: 블록 업데이트된 ``x``는
        정규화되지 않아 블록을 거치며 커질 수 있고, 이로 인해 ``omega``가 폭발하고
        손실이 불안정해진다. ``h``는 여전히 불변 계수로서 학습된 메시지 패싱
        컨텍스트를 담고 있다.
        """
        num_graphs = int(batch.query_batch.max().item()) + 1

        basis_list = [
            x_in,                                       # 쿼리 중심 좌표
            self._reference_delta_basis(batch),         # 레퍼런스 무게중심 -> 쿼리
            self._pocket_delta_basis(batch),            # 포켓 무게중심 -> 쿼리
            self._neighbor_vector_basis(x_in, edge_index),  # 쿼리 이웃 집계
        ]
        if self.use_pocket_conditioning:
            # 포켓 형태 벡터 (단순 중심이 아닌 방향성 표면 정보)
            basis_list.append(pocket_basis if pocket_basis is not None else torch.zeros_like(x_in))
        bases = torch.stack(basis_list, dim=1)  # [N, n_bases, 3]

        omega_atom = (self.out_rigid_omega(h).unsqueeze(-1) * bases).sum(dim=1)  # [N, 3]
        vlin_atom = (self.out_rigid_vlin(h).unsqueeze(-1) * bases).sum(dim=1)  # [N, 3]

        omega = _segment_mean(omega_atom, batch.query_batch, num_graphs)  # [G, 3]
        vlin = _segment_mean(vlin_atom, batch.query_batch, num_graphs)  # [G, 3]

        com = _segment_mean(x_in, batch.query_batch, num_graphs)  # [G, 3]
        rel = x_in - com[batch.query_batch]  # [N, 3]
        omega_i = omega[batch.query_batch]  # [N, 3]
        vlin_i = vlin[batch.query_batch]  # [N, 3]

        return torch.cross(omega_i, rel, dim=-1) + vlin_i
    
    def forward(self, batch: AlignmentBatch, t_graph: Tensor) -> Tensor:
        """쿼리 속도 필드 ``v_theta(x_t, t, cond)``를 예측한다.

        Args:
            batch: 정렬 미니배치.
            t_graph: 그래프당 시간 ``[B]``.

        Returns:
            쿼리 원자의 속도 벡터 ``[Nq, 3]``.
        """
        if batch.query_pos.numel() == 0:
            return torch.zeros_like(batch.query_pos)

        num_graphs = int(batch.query_batch.max().item()) + 1
        com = _segment_mean(batch.query_pos, batch.query_batch, num_graphs)
        x = batch.query_pos - com[batch.query_batch]
        x_in = x.clone()  # 중심화된 입력 좌표, rigid head에서 사용됨
        
        h_atom = self.atom_embed(batch.query_atom_type)

        if self.use_atom_index_embed:
            local_index = self._local_atom_index(batch.query_batch)
            if int(local_index.max().item()) >= self.max_atoms:
                raise ValueError(f"local atom index exceeds max_atoms={self.max_atoms}; increase --max-atoms.")
            h_atom = h_atom + self.atom_index_embed(local_index)

        h_t = self.time_embed(t_graph)[batch.query_batch]
        ref_dir, ref_dist = self._reference_context(batch)

        h = self.in_proj(torch.cat([h_atom, h_t, ref_dir, ref_dist], dim=-1))

        # 포켓 형태 컨디셔닝: 리간드 피처를 보강하고 고정된 결합 부위 원자로부터
        # 방향성 포켓 형태 기저를 구성한다 (리간드 무게중심 기준 좌표계에서).
        pocket_basis = None
        if (
            self.use_pocket_conditioning
            and batch.pocket_pos is not None
            and batch.pocket_batch is not None
            and batch.pocket_batch.numel() > 0
        ):
            xp = batch.pocket_pos - com[batch.pocket_batch]
            if batch.pocket_atom_type is not None:
                h_pkt = self.atom_embed(batch.pocket_atom_type)  # 포켓 화학 정보 (공유 원자 임베딩)
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

        # 최우선: 디버그용 직접 헤드.
        # 의도적으로 비등변이며 정상 동작 확인 용도로만 사용해야 한다.
        if self.use_direct_vector_head:
            return self.out_vec(h)

        # Direction A: 강체 속도 필드 (분자 내 기하 구조 정확히 보존).
        if self.use_rigid_head:
            return self._rigid_head(h, x, x_in, batch, edge_index, pocket_basis)

        # 프로덕션 호환 표현력 있는 등변 헤드.
        if self.use_equivariant_basis_head:
            return self._equivariant_basis_head(h, x, batch, edge_index)

        # 레거시 제약적 등변 헤드.
        gate = self.out_gate(h)
        v = gate * x
        return v