"""Pocket-aware guidance utilities with strict batch-safe interface checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor

from .model import AlignmentBatch

UFFEnergyFn = Callable[[Tensor, AlignmentBatch, "UFFGuidanceConfig"], Tensor]


@dataclass
class UFFGuidanceConfig:
    """Guidance hyperparameters."""

    scale: float = 1.0


class UFFPocketGuidance:
    """Simple differentiable guidance that avoids cross-graph coupling."""

    def __init__(self, config: Optional[UFFGuidanceConfig] = None, backend_energy_fn: Optional[UFFEnergyFn] = None) -> None:
        self.config = config or UFFGuidanceConfig()
        self._backend_energy_fn = backend_energy_fn

    def energy(self, x: Tensor, batch: AlignmentBatch) -> Tensor:
        """Compute per-graph additive energy."""
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
        """Return guidance tensor with exact shape `[Nq, 3]`."""
        del t_graph
        x = batch.query_pos.detach().clone().requires_grad_(True)
        grad = torch.autograd.grad(self.energy(x=x, batch=batch), x, create_graph=False, retain_graph=False)[0]
        guidance = -self.config.scale * grad

        if guidance.shape != v.shape:
            raise ValueError(
                f"GuidanceFn must return Tensor[Nq,3]; expected {tuple(v.shape)} got {tuple(guidance.shape)}."
            )
        return guidance