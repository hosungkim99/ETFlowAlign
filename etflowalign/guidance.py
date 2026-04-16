"""Pocket-aware guidance utilities, including UFF-gradient guidance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from .model import AlignmentBatch


@dataclass
class UFFGuidanceConfig:
    """Hyperparameters for pocket-aware UFF gradient guidance."""

    scale: float = 1.0
    query_repulsion_weight: float = 0.2
    pocket_attraction_weight: float = 1.0
    pocket_repulsion_weight: float = 0.5
    epsilon: float = 0.1
    sigma: float = 1.5


class UFFPocketGuidance:
    """Differentiable pocket-aware UFF-style guidance.

    This class computes an energy and returns ``-dE/dx`` for query coordinates.
    If a dedicated UFF backend is available in the runtime, users can swap this
    implementation with that backend while preserving the same call contract.
    """

    def __init__(self, config: Optional[UFFGuidanceConfig] = None) -> None:
        self.config = config or UFFGuidanceConfig()

    def _lj(self, dist: Tensor, sigma: float, epsilon: float) -> Tensor:
        inv = sigma / dist.clamp_min(1e-6)
        inv6 = inv**6
        inv12 = inv6 * inv6
        return 4.0 * epsilon * (inv12 - inv6)

    def energy(self, x: Tensor, batch: AlignmentBatch) -> Tensor:
        """Compute UFF-like pocket-aware energy for a single alignment state."""
        e_total = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # Query self term: weak steric repulsion to prevent collapse.
        d_qq = torch.cdist(x, x)
        eye = torch.eye(x.size(0), device=x.device, dtype=torch.bool)
        d_qq = d_qq.masked_fill(eye, 1e6)
        e_qq = self._lj(d_qq, sigma=self.config.sigma, epsilon=self.config.epsilon).mean()
        e_total = e_total + self.config.query_repulsion_weight * e_qq

        # Pocket interaction terms.
        if batch.pocket_pos is not None and batch.pocket_pos.numel() > 0:
            d_qp = torch.cdist(x, batch.pocket_pos)
            lj_qp = self._lj(d_qp, sigma=self.config.sigma, epsilon=self.config.epsilon)
            # Balance attraction and repulsion for pocket-aware steering.
            attract = -torch.exp(-d_qp / self.config.sigma).mean()
            repel = lj_qp.clamp_min(0.0).mean()
            e_total = e_total + self.config.pocket_attraction_weight * attract + self.config.pocket_repulsion_weight * repel

        return e_total

    def __call__(self, batch: AlignmentBatch, t_graph: Tensor, v: Tensor) -> Tensor:
        """Return guidance vector field g(x,t) = -∇_x E(x)."""
        del t_graph, v
        x = batch.query_pos.detach().clone().requires_grad_(True)
        energy = self.energy(x=x, batch=batch)
        grad = torch.autograd.grad(energy, x, create_graph=False, retain_graph=False)[0]
        return -self.config.scale * grad
