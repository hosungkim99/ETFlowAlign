"""백본 공용 헬퍼: 반경 그래프 구성, scatter 집계, 안전 노름/소프트정규화.

등변성을 지키기 위해 좌표는 절대값이 아니라
상대 기하(거리 r_ij, 단위 방향 û_ij)로만 네트워크에 들어간다.
"""

from __future__ import annotations

import torch
from torch import Tensor


def safe_norm(x: Tensor, dim: int = -1, eps: float = 1e-8, keepdim: bool = False) -> Tensor:
    """거리 0에서도 유한한 그래디언트를 갖는 노름.

    torch.norm 은 크기가 0인 지점에서 그래디언트가 NaN 이 되므로,
    제곱합에 eps 를 더한 뒤 sqrt 를 취한다. (과거 NaN 크래시 방지 교훈)
    """
    return torch.sqrt(torch.sum(x * x, dim=dim, keepdim=keepdim) + eps)


def soft_normalize(vec: Tensor, eps: float = 1e-8) -> Tensor:
    """벡터 피처 소프트정규화: vec / (1 + ||vec||).

    채널별 3D 벡터의 크기로 나눠 폭발을 막는다(SPEC §5.5 안정화 3종 중 하나).
    스칼라(불변)로 나누는 연산이라 등변성은 유지된다.

    vec: [N, 3, F]  ->  같은 모양
    """
    norm = safe_norm(vec, dim=1, eps=eps, keepdim=True)  # [N, 1, F] (불변)
    return vec / (1.0 + norm)


def scatter_add(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """dim=0 기준 index 별 합산. torch_scatter 의존성 없이 native 로 구현.

    src:   [E, ...]
    index: [E]  (0 <= index < dim_size)
    반환:  [dim_size, ...]
    """
    out_shape = (dim_size,) + tuple(src.shape[1:])
    out = src.new_zeros(out_shape)
    idx = index.view((-1,) + (1,) * (src.dim() - 1)).expand_as(src)
    out.scatter_add_(0, idx, src)
    return out


def scatter_mean(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """dim=0 기준 index 별 평균 (COM 계산 등)."""
    s = scatter_add(src, index, dim_size)
    ones = torch.ones(src.size(0), device=src.device, dtype=src.dtype)
    c = scatter_add(ones, index, dim_size).clamp(min=1.0)
    return s / c.view((-1,) + (1,) * (src.dim() - 1))


def build_radius_graph(
    pos: Tensor, batch: Tensor, cutoff: float, eps: float = 1e-8
):
    """같은 분자(batch) 안에서 cutoff 이내 원자쌍으로 방향 그래프를 만든다.

    작은 분자(drug-sized) 기준 O(N^2) 이면 충분. 필요 시 radius_graph 로 교체.

    반환:
      edge_index: [2, E]  (row0 = i 수신자, row1 = j 송신자, 메시지는 j->i)
      r_ij:       [E]     원자쌍 거리
      u_ij:       [E, 3]  단위 방향벡터 (pos_j - pos_i)/r_ij  (등변량)
    """
    diff = pos[:, None, :] - pos[None, :, :]          # [N, N, 3]
    dist = torch.sqrt((diff * diff).sum(-1) + eps)     # [N, N]
    same = batch[:, None] == batch[None, :]            # 같은 분자만
    mask = same & (dist < cutoff) & (dist > 1e-4)      # self-loop 제외
    idx = mask.nonzero(as_tuple=False)                 # [E, 2]
    i, j = idx[:, 0], idx[:, 1]
    edge_index = torch.stack([i, j], dim=0)            # [2, E]
    r_ij = dist[i, j]                                  # [E]
    u_ij = (pos[j] - pos[i]) / r_ij[:, None]           # [E, 3]
    return edge_index, r_ij, u_ij
