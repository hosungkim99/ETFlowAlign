"""ODE sampling for ETFlowAlign with optional guidance hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor

from .model import AlignmentBatch, ETFlowAlignModel

GuidanceFn = Callable[[AlignmentBatch, Tensor, Tensor], Tensor]


@dataclass
class ODESamplerConfig:
    n_steps: int = 50
    t_start: float = 0.0
    t_end: float = 1.0
    guidance_scale: float = 0.0
    max_guidance_norm: float = 5.0


class ETFlowAlignSampler:
    """Euler ODE sampler for alignment.

    The sampler integrates:
        dx / dt = v_theta(x, t, cond) + guidance
    with explicit clipping on guidance to improve stability.
    """

    def __init__(self, model: ETFlowAlignModel, config: ODESamplerConfig) -> None:
        self.model = model
        self.config = config

    def _clip_guidance(self, g: Tensor) -> Tensor:
        norm = torch.norm(g, dim=-1, keepdim=True).clamp_min(1e-8)
        scale = (self.config.max_guidance_norm / norm).clamp(max=1.0)
        return g * scale

    @torch.no_grad()
    def sample(
        self,
        batch: AlignmentBatch,
        x0: Tensor,
        guidance_fn: Optional[GuidanceFn] = None,
    ) -> Tensor:
        num_graphs = int(batch.query_batch.max().item()) + 1
        t_grid = torch.linspace(self.config.t_start, self.config.t_end, self.config.n_steps + 1, device=x0.device)

        x = x0.clone()
        for i in range(self.config.n_steps):
            t = t_grid[i]
            dt = t_grid[i + 1] - t
            t_graph = torch.full((num_graphs,), float(t), device=x.device)

            cur_batch = AlignmentBatch(
                query_pos=x,
                query_atom_type=batch.query_atom_type,
                query_batch=batch.query_batch,
                reference_pos=batch.reference_pos,
                reference_atom_type=batch.reference_atom_type,
                reference_batch=batch.reference_batch,
                pocket_pos=batch.pocket_pos,
            )
            v = self.model(cur_batch, t_graph=t_graph)

            if guidance_fn is not None and self.config.guidance_scale > 0.0:
                g = guidance_fn(cur_batch, t_graph, v)
                g = self._clip_guidance(g)
                v = v + self.config.guidance_scale * g

            x = x + dt * v

            # NaN/Inf safety net for stiff guidance.
            if not torch.isfinite(x).all():
                raise FloatingPointError("Non-finite coordinates during ODE integration. Reduce guidance strength.")

        return x
