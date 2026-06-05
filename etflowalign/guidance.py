"""배치 안전 인터페이스 검사를 갖춘 포켓 인식 가이던스 유틸리티."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor

from .model import AlignmentBatch

UFFEnergyFn = Callable[[Tensor, AlignmentBatch, "UFFGuidanceConfig"], Tensor]


@dataclass
class UFFGuidanceConfig:
    """가이던스 하이퍼파라미터."""

    scale: float = 1.0


class UFFPocketGuidance:
    """그래프 간 결합을 방지하는 간단한 미분 가능 가이던스."""

    def __init__(self, config: Optional[UFFGuidanceConfig] = None, backend_energy_fn: Optional[UFFEnergyFn] = None) -> None:
        self.config = config or UFFGuidanceConfig()
        self._backend_energy_fn = backend_energy_fn

    def energy(self, x: Tensor, batch: AlignmentBatch) -> Tensor:
        """그래프별 가산 에너지를 계산한다."""
        if self._backend_energy_fn is not None:
            out = self._backend_energy_fn(x, batch, self.config)
            if not torch.is_tensor(out):
                raise TypeError("Backend energy function must return a torch.Tensor scalar.")
            return out

        total_energy = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        for graph_id in batch.query_batch.unique(sorted=True):
            graph_mask = batch.query_batch == graph_id
            xg = x[graph_mask]
            if xg.size(0) <= 1:
                continue
            d = torch.cdist(xg, xg)
            d = d + torch.eye(xg.size(0), device=x.device, dtype=x.dtype) * 1e6
            total_energy = total_energy + (1.0 / d.clamp_min(1e-3)).mean()
        return total_energy

    def __call__(self, batch: AlignmentBatch, t_graph: Tensor, v: Tensor) -> Tensor:
        """정확히 `[Nq, 3]` 형상의 가이던스 텐서를 반환한다."""
        del t_graph
        x = batch.query_pos.detach().clone().requires_grad_(True)
        grad = torch.autograd.grad(self.energy(x=x, batch=batch), x, create_graph=False, retain_graph=False)[0]
        guidance = -self.config.scale * grad

        if guidance.shape != v.shape:
            raise ValueError(
                f"GuidanceFn must return Tensor[Nq,3]; expected {tuple(v.shape)} got {tuple(guidance.shape)}."
            )
        return guidance