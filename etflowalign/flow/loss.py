"""flow-matching 목적함수.

batchwise_l2: 원자별 l2 노름 -> 분자별 평균 -> 배치 평균 (ET-Flow 동일).
보조손실 없음 (bond/clash penalty 금지, SPEC §4.3).
"""

from __future__ import annotations

import torch
from torch import Tensor

from etflowalign.backbone.utils import scatter_add


def batchwise_l2_loss(pred: Tensor, target: Tensor, batch: Tensor) -> Tensor:
    """pred, target: [N, 3], batch: [N] -> 스칼라 손실."""
    per_atom = torch.norm(pred - target, p=2, dim=-1)          # [N]
    num_graphs = int(batch.max().item()) + 1
    sums = scatter_add(per_atom, batch, num_graphs)            # [B]
    counts = scatter_add(torch.ones_like(per_atom), batch, num_graphs)
    per_mol = sums / counts.clamp(min=1.0)                     # 분자별 평균
    return per_mol.mean()                                      # 배치 평균
