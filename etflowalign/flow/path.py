"""보간 경로 + 기하 헬퍼 (Kabsch RMSD).

linear flow-matching path:
  x_t = (1-t)·x0 + t·x1,   목표 속도 u = x1 - x0
시간 t 는 분자별 스칼라이며 원자에 브로드캐스트된다.
"""

from __future__ import annotations

import torch
from torch import Tensor


def center_pos(pos: Tensor, batch: Tensor) -> Tensor:
    """분자별 COM 을 빼서 중심화 (평행이동 자유도 제거)."""
    num_graphs = int(batch.max().item()) + 1
    out = pos.clone()
    for g in range(num_graphs):
        idx = (batch == g).nonzero(as_tuple=True)[0]
        out[idx] = out[idx] - out[idx].mean(dim=0, keepdim=True)
    return out


def center_by_reference(x1: Tensor, batch: Tensor, ref_pos: Tensor,
                        ref_batch: Tensor, num_graphs: int):
    """조건화 시: 전체 복합체를 reference COM 기준으로 중심화.

    query 의 절대 위치(=reference 대비 정렬 위치)를 보존하므로,
    x1 은 COM 이 0 이 아니라 정렬 오프셋을 유지한다(placement 학습).
    반환: (중심화된 x1, 중심화된 ref_pos)
    """
    from etflowalign.backbone.utils import scatter_mean
    ref_com = scatter_mean(ref_pos, ref_batch, num_graphs)   # [B, 3]
    return x1 - ref_com[batch], ref_pos - ref_com[ref_batch]


def sample_time(num_graphs: int, device, dtype=torch.float32) -> Tensor:
    """분자별 t ~ U(0,1)."""
    return torch.rand(num_graphs, device=device, dtype=dtype)


def interpolate(x0: Tensor, x1: Tensor, t: Tensor, batch: Tensor) -> Tensor:
    """x_t = (1-t)·x0 + t·x1.  t: [B] -> 원자별 브로드캐스트."""
    t_atom = t[batch][:, None]  # [N, 1]
    return (1.0 - t_atom) * x0 + t_atom * x1


def target_velocity(x0: Tensor, x1: Tensor) -> Tensor:
    """linear path 의 목표 속도 u = x1 - x0 (t 에 무관)."""
    return x1 - x0


# --------------------------------------------------------------------------
# Kabsch 정렬 / RMSD  ── 과거 transpose 버그 이력 있음, 규약 주의
# --------------------------------------------------------------------------
def kabsch_rotation(P: Tensor, Q: Tensor) -> Tensor:
    """중심화된 P 를 Q 에 최적 정렬하는 회전 R 반환 (P @ R.T ≈ Q).

    H = P^T Q, SVD H=U S V^T, R = V·diag(1,1,d)·U^T, d=sign(det(V U^T)).
    """
    H = P.T @ Q
    U, S, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.det(Vt.T @ U.T))
    D = torch.diag(torch.tensor([1.0, 1.0, d], device=P.device, dtype=P.dtype))
    R = Vt.T @ D @ U.T
    return R


def kabsch_aligned_rmsd(pred: Tensor, target: Tensor) -> float:
    """단일 분자: 중심화 + 최적 회전 후 RMSD (형상만 비교)."""
    Pc = pred - pred.mean(dim=0, keepdim=True)
    Qc = target - target.mean(dim=0, keepdim=True)
    R = kabsch_rotation(Pc, Qc)
    aligned = Pc @ R.T
    return torch.sqrt(((aligned - Qc) ** 2).sum(dim=-1).mean()).item()


def raw_rmsd(pred: Tensor, target: Tensor) -> float:
    """정렬 없는 원시 RMSD (중심화만)."""
    Pc = pred - pred.mean(dim=0, keepdim=True)
    Qc = target - target.mean(dim=0, keepdim=True)
    return torch.sqrt(((Pc - Qc) ** 2).sum(dim=-1).mean()).item()


def kabsch_align_source_to_target(x0: Tensor, x1: Tensor, batch: Tensor,
                                  num_graphs: int) -> Tensor:
    """각 분자의 x0(source)를 x1(target)에 최적 회전으로 정렬(중심화 포함).

    등변 백본은 '무작위 방향 x0 -> 고정 방향 x1' 매핑이 원리상 불가능하다
    (등변성 위반). source 를 target 방향에 맞춰두면 flow 는 전역 회전이 아니라
    '모양 형성'만 학습하고, 생성 시 방향은 등변성이 처리한다(ET-Flow rmsd_align).

    x1 은 이미 중심화됐다고 가정. 반환: 정렬된 x0 [N,3].
    """
    out = x0.clone()
    for g in range(num_graphs):
        idx = (batch == g).nonzero(as_tuple=True)[0]
        P = x0[idx] - x0[idx].mean(dim=0, keepdim=True)
        Q = x1[idx] - x1[idx].mean(dim=0, keepdim=True)
        R = kabsch_rotation(P, Q)          # P @ R.T ≈ Q
        out[idx] = P @ R.T
    return out
