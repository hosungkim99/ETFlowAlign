"""Small shared helpers for ETFlowAlign.

Keep this module lightweight; do not move core algorithmic logic here.
"""

from __future__ import annotations

import torch
from torch import Tensor


def safe_norm(x: Tensor, dim: int = -1, keepdim: bool = False, eps: float = 1e-8) -> Tensor:
    """Numerically stable vector norm."""
    return torch.sqrt((x * x).sum(dim=dim, keepdim=keepdim).clamp_min(eps))


def segment_mean(x: Tensor, batch: Tensor, num_graphs: int | None = None) -> Tensor:
    """Compute per-graph mean tensor."""
    if num_graphs is None:
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
    out = torch.zeros(num_graphs, x.size(-1), device=x.device, dtype=x.dtype)
    cnt = torch.zeros(num_graphs, 1, device=x.device, dtype=x.dtype)
    out.index_add_(0, batch, x)
    cnt.index_add_(0, batch, torch.ones_like(batch, dtype=x.dtype).unsqueeze(-1))
    return out / cnt.clamp_min(1.0)


def center_by_batch(x: Tensor, batch: Tensor) -> Tensor:
    """Subtract per-graph mean coordinates.

    Args:
        x: Coordinate/feature tensor ``[N, D]``.
        batch: Graph id per row ``[N]``.

    Returns:
        Centered tensor with per-graph mean removed.
    """
    if batch.numel() == 0:
        return x
    means = segment_mean(x, batch)
    return x - means[batch]
