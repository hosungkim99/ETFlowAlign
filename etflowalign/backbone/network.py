"""등변 트랜스포머 백본: 블록 스택 -> 원자당 속도장 v_θ [N,3].

flow matching 의 벡터장을 예측한다. 조건/태스크를 모르는 순수 등변 네트워크
(SPEC §5). 등변성은 tests/test_equivariance.py 로 강제한다.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from etflowalign.backbone.embedding import AtomTimeEmbedding
from etflowalign.backbone.layers import EquivariantAttention, GatedEquivariantBlock
from etflowalign.backbone.rbf import CosineCutoff, ExpNormalSmearing
from etflowalign.backbone.utils import build_radius_graph, soft_normalize


class EquivariantTransformer(nn.Module):
    """ET 스타일 등변 백본.

    forward(z, pos, t, batch) -> v_θ [N, 3]
      - z:     [N] long   원자종
      - pos:   [N, 3]     현재 좌표 x_t
      - t:     [B]        flow-matching 시간 (분자별)
      - batch: [N] long   원자->분자 매핑
    """

    def __init__(
        self,
        hidden_channels: int = 256,
        num_layers: int = 8,
        num_rbf: int = 64,
        num_heads: int = 8,
        cutoff: float = 10.0,
        max_z: int = 100,
        soft_norm: bool = True,
    ):
        super().__init__()
        self.hidden = hidden_channels
        self.cutoff = float(cutoff)
        self.soft_norm = bool(soft_norm)

        self.embedding = AtomTimeEmbedding(hidden_channels, max_z=max_z)
        self.distance_expansion = ExpNormalSmearing(cutoff, num_rbf)
        self.cutoff_fn = CosineCutoff(cutoff)

        self.layers = nn.ModuleList(
            [EquivariantAttention(hidden_channels, num_rbf, num_heads) for _ in range(num_layers)]
        )
        self.out_norm = nn.LayerNorm(hidden_channels)
        # 속도장 헤드: 벡터피처 -> 1개 3D 벡터, zero-init 으로 초기 출력 0
        self.velocity_head = GatedEquivariantBlock(hidden_channels, out_channels=1, zero_init=True)

    def forward(self, z: Tensor, pos: Tensor, t: Tensor, batch: Tensor,
                ref_z: Tensor = None, ref_pos: Tensor = None,
                ref_batch: Tensor = None) -> Tensor:
        """ref_* 가 주어지면 query+reference 를 한 그래프로 처리(조건화).

        reference 노드는 위치(상대 기하)로만 조건에 들어가고, 속도장 출력은
        query 노드만 반환한다. joint(query+ref) 회전 등변성이 유지된다.
        """
        Nq = z.size(0)
        conditioned = ref_z is not None
        if conditioned:
            is_ref = torch.cat([
                torch.zeros(Nq, dtype=torch.long, device=z.device),
                torch.ones(ref_z.size(0), dtype=torch.long, device=z.device),
            ])
            z = torch.cat([z, ref_z])
            pos = torch.cat([pos, ref_pos], dim=0)
            batch = torch.cat([batch, ref_batch])
        else:
            is_ref = None

        N = z.size(0)
        h = self.embedding(z, t, batch, is_ref)               # [N, F]
        vec = pos.new_zeros(N, 3, self.hidden)                # 벡터피처 0 초기화

        edge_index, r_ij, u_ij = build_radius_graph(pos, batch, self.cutoff)
        f_ij = self.distance_expansion(r_ij)                  # [E, num_rbf]
        cutoff_val = self.cutoff_fn(r_ij)                     # [E]

        for layer in self.layers:
            dx, dvec = layer(h, vec, edge_index, r_ij, u_ij, f_ij, cutoff_val)
            h = h + dx
            vec = vec + dvec
            if self.soft_norm:
                vec = soft_normalize(vec)                     # 안정화(SPEC §5.5)

        h = self.out_norm(h)
        v_out = self.velocity_head(h, vec).squeeze(-1)        # [N, 3]
        if conditioned:
            return v_out[:Nq]                                 # query 노드만 반환
        return v_out
