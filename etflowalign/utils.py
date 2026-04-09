"""Small shared helpers for ETFlowAlign.

Keep this module lightweight; do not move core algorithmic logic here.
"""

from __future__ import annotations

import torch
from torch import Tensor


def center_by_batch(x: Tensor, batch: Tensor) -> Tensor:
    """Subtract per-graph mean coordinates.

    Args:
        x: Coordinate/feature tensor ``[N, D]``.
        batch: Graph id per row ``[N]``.

    Returns:
        Centered tensor with per-graph mean removed.
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
