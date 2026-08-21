"""등변 어텐션 블록 + 게이트드 등변 출력 헤드.

TorchMD-Net(ET) 의 EquivariantMultiHeadAttention 을 self-contained 로 재구현.
등변 규칙(SPEC §5.2):
  - 불변 스칼라끼리만 MLP/비선형에 자유롭게 넣는다.
  - 벡터는 (a) 불변 스칼라로 스케일, (b) 등변 방향 û_ij 와 결합,
    (c) 두 벡터 내적으로 불변 생성 — 이 3가지로만 다룬다.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from etflowalign.backbone.utils import safe_norm, scatter_add


class EquivariantAttention(nn.Module):
    """스칼라 h [N,F] 와 벡터 vec [N,3,F] 를 함께 갱신하는 등변 어텐션 레이어."""

    def __init__(self, hidden_channels: int = 256, num_rbf: int = 64, num_heads: int = 8):
        super().__init__()
        assert hidden_channels % num_heads == 0, "hidden 은 head 수로 나눠떨어져야 함"
        self.hidden = hidden_channels
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads

        self.layernorm = nn.LayerNorm(hidden_channels)  # 안정화: 스칼라 LayerNorm
        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels * 3)
        self.o_proj = nn.Linear(hidden_channels, hidden_channels * 3)
        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 3, bias=False)

        # 거리(RBF) -> 키/값 게이트 (distance_influence = "both")
        self.dk_proj = nn.Linear(num_rbf, hidden_channels)
        self.dv_proj = nn.Linear(num_rbf, hidden_channels * 3)

        self.act = nn.SiLU()
        self.reset_parameters()

    def reset_parameters(self):
        for m in [self.q_proj, self.k_proj, self.v_proj, self.o_proj,
                  self.dk_proj, self.dv_proj]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.vec_proj.weight)

    def forward(
        self,
        x: Tensor,          # [N, F]  스칼라 (불변)
        vec: Tensor,        # [N, 3, F]  벡터 (등변)
        edge_index: Tensor, # [2, E]
        r_ij: Tensor,       # [E]
        u_ij: Tensor,       # [E, 3]  단위 방향 (등변)
        f_ij: Tensor,       # [E, num_rbf]  거리 특징 (불변)
        cutoff_val: Tensor, # [E]  cosine cutoff 포락선
    ):
        N = x.size(0)
        E = edge_index.size(1)
        H, D = self.num_heads, self.head_dim

        xn = self.layernorm(x)
        q = self.q_proj(xn).reshape(N, H, D)
        k = self.k_proj(xn).reshape(N, H, D)
        v = self.v_proj(xn).reshape(N, H, 3 * D)

        # 벡터 투영 -> 3분할. vec1·vec2 내적은 불변(스칼라 업데이트에 사용)
        vec1, vec2, vec3 = torch.split(self.vec_proj(vec), self.hidden, dim=-1)  # each [N,3,F]
        vec_dot = (vec1 * vec2).sum(dim=1)  # [N, F]  (등변벡터 내적 -> 불변)

        dk = self.act(self.dk_proj(f_ij)).reshape(E, H, D)
        dv = self.act(self.dv_proj(f_ij)).reshape(E, H, 3 * D)

        i, j = edge_index[0], edge_index[1]  # 메시지 j -> i

        # 어텐션 가중치 (불변): (q_i · k_j ⊙ dk) -> SiLU -> cutoff 포락
        attn = (q[i] * k[j] * dk).sum(dim=-1)          # [E, H]
        attn = self.act(attn) * cutoff_val[:, None]     # [E, H]

        # 값 메시지: 스칼라부 s, 벡터 게이트 g1(피처 게이트), g2(방향 주입)
        vj = v[j] * dv                                  # [E, H, 3D]
        s, g1, g2 = torch.split(vj, D, dim=2)           # each [E, H, D]

        s = (s * attn.unsqueeze(-1)).reshape(E, self.hidden)  # 스칼라 메시지 [E, F]

        vecj = vec[j].reshape(E, 3, H, D)               # 이웃 벡터피처
        # Δvec 메시지 = (이웃벡터 ⊙ 불변게이트) + (등변방향 ⊙ 불변게이트)
        vec_msg = vecj * g1.unsqueeze(1) + u_ij.reshape(E, 3, 1, 1) * g2.unsqueeze(1)
        vec_msg = vec_msg.reshape(E, 3, self.hidden)    # [E, 3, F]

        # i 로 집계
        dx_agg = scatter_add(s, i, N)                   # [N, F]
        dvec_agg = scatter_add(vec_msg, i, N)           # [N, 3, F]

        # 출력 투영으로 스칼라/벡터 갱신 결합
        o1, o2, o3 = torch.split(self.o_proj(dx_agg), self.hidden, dim=1)  # each [N, F]
        dx = vec_dot * o2 + o3                           # [N, F]  (불변)
        dvec = vec3 * o1.unsqueeze(1) + dvec_agg         # [N, 3, F]  (등변)
        return dx, dvec


class GatedEquivariantBlock(nn.Module):
    """벡터 피처를 등변 규칙 하에 out_channels 개의 3D 벡터로 사영.

    PaiNN/TorchMD-Net 출력 블록. 속도장 헤드로 out_channels=1 사용.
    zero_init=True 이면 초기 벡터 출력이 0 이 되어 학습이 항등 근처에서 시작한다
    (SPEC §5.5 안정화: 출력 zero-init).
    """

    def __init__(self, hidden_channels: int, out_channels: int, zero_init: bool = False):
        super().__init__()
        self.out_channels = out_channels
        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)
        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_channels),
        )
        self.reset_parameters()
        if zero_init:
            # 벡터 경로를 0 으로 시작 -> 초기 출력 벡터 = 0 (게이트는 학습 가능하게 유지)
            nn.init.zeros_(self.vec2_proj.weight)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        for m in self.update_net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor, vec: Tensor):
        v1 = self.vec1_proj(vec)             # [N, 3, F]
        v2 = self.vec2_proj(vec)             # [N, 3, out]
        v1_norm = safe_norm(v1, dim=1)       # [N, F]  (불변)
        gate = self.update_net(torch.cat([x, v1_norm], dim=-1))  # [N, out]  (불변)
        out_vec = v2 * gate.unsqueeze(1)     # [N, 3, out]  (등변)
        return out_vec
