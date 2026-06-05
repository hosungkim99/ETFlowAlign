"""ETFlowAlign 공용 소형 헬퍼 모음.

이 모듈은 경량으로 유지한다. 핵심 알고리즘 로직은 여기에 두지 않는다.
"""

from __future__ import annotations

import torch
from torch import Tensor


def safe_norm(x: Tensor, dim: int = -1, keepdim: bool = False, eps: float = 1e-8) -> Tensor:
    """수치적으로 안정적인 벡터 노름."""
    return torch.sqrt((x * x).sum(dim=dim, keepdim=keepdim).clamp_min(eps))


def segment_mean(x: Tensor, batch: Tensor, num_graphs: int | None = None) -> Tensor:
    """그래프별 평균 텐서를 계산한다."""
    if num_graphs is None:
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
    out = torch.zeros(num_graphs, x.size(-1), device=x.device, dtype=x.dtype)
    cnt = torch.zeros(num_graphs, 1, device=x.device, dtype=x.dtype)
    out.index_add_(0, batch, x)
    cnt.index_add_(0, batch, torch.ones_like(batch, dtype=x.dtype).unsqueeze(-1))
    return out / cnt.clamp_min(1.0)


def center_by_batch(x: Tensor, batch: Tensor) -> Tensor:
    """그래프별 평균 좌표를 뺀다.

    Args:
        x: 좌표/피처 텐서 ``[N, D]``.
        batch: 행별 그래프 ID ``[N]``.

    Returns:
        그래프별 평균이 제거된 중심화된 텐서.
    """
    num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
    if num_graphs == 0:
        return x
    means = torch.zeros(num_graphs, x.size(-1), device=x.device, dtype=x.dtype)
    counts = torch.zeros(num_graphs, 1, device=x.device, dtype=x.dtype)
    means.index_add_(0, batch, x)
    counts.index_add_(0, batch, torch.ones_like(batch, dtype=x.dtype).unsqueeze(-1))
    means = means / counts.clamp_min(1.0)
    return x - means[batch]
