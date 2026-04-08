"""ODE sampling for ETFlowAlign with safe guidance injection policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import torch
from torch import Tensor

from .model import AlignmentBatch, ETFlowAlignModel

GuidanceFn = Callable[[AlignmentBatch, Tensor, Tensor], Tensor]


@dataclass
class ODESamplerConfig:
    n_steps: int = 50
    t_start: float = 0.0
    t_end: float = 1.0
    solver: Literal["euler", "heun"] = "heun"
    guidance_scale: float = 0.0
    guidance_mode: Literal["vector_field", "predictor_corrector"] = "vector_field"
    max_guidance_norm: float = 5.0


class ETFlowAlignSampler:
    """ODE sampler for alignment flow matching.

    Guidance mode:
        - vector_field: add guidance directly to velocity before state update.
        - predictor_corrector: apply guidance as a separate correction step.
    """

    def __init__(self, model: ETFlowAlignModel, config: ODESamplerConfig) -> None:
        self.model = model
        self.config = config

    def _clip_guidance(self, g: Tensor) -> Tensor:
        norm = torch.norm(g, dim=-1, keepdim=True).clamp_min(1e-8)
        return g * (self.config.max_guidance_norm / norm).clamp(max=1.0)

    def _model_v(self, batch: AlignmentBatch, x: Tensor, t_graph: Tensor) -> Tensor:
        cur = AlignmentBatch(
            query_pos=x,
            query_atom_type=batch.query_atom_type,
            query_batch=batch.query_batch,
            reference_pos=batch.reference_pos,
            reference_atom_type=batch.reference_atom_type,
            reference_batch=batch.reference_batch,
            pocket_pos=batch.pocket_pos,
        )
        return self.model(cur, t_graph=t_graph)

    def _apply_guidance(self, batch: AlignmentBatch, x: Tensor, t_graph: Tensor, v: Tensor, guidance_fn: Optional[GuidanceFn]) -> Tensor:
        if guidance_fn is None or self.config.guidance_scale <= 0.0:
            return x, v

        cur = AlignmentBatch(
            query_pos=x,
            query_atom_type=batch.query_atom_type,
            query_batch=batch.query_batch,
            reference_pos=batch.reference_pos,
            reference_atom_type=batch.reference_atom_type,
            reference_batch=batch.reference_batch,
            pocket_pos=batch.pocket_pos,
        )
        g = self._clip_guidance(guidance_fn(cur, t_graph, v))

        if self.config.guidance_mode == "vector_field":
            return x, v + self.config.guidance_scale * g

        # predictor-corrector style: update state separately by guidance.
        x = x + (self.config.guidance_scale / max(1, self.config.n_steps)) * g
        return x, v

    @torch.no_grad()
    def sample(self, batch: AlignmentBatch, x0: Tensor, guidance_fn: Optional[GuidanceFn] = None) -> Tensor:
        num_graphs = int(batch.query_batch.max().item()) + 1
        t_grid = torch.linspace(self.config.t_start, self.config.t_end, self.config.n_steps + 1, device=x0.device)

        x = x0.clone()
        for i in range(self.config.n_steps):
            t = t_grid[i]
            dt = t_grid[i + 1] - t
            t_graph = torch.full((num_graphs,), float(t), device=x.device)

            v = self._model_v(batch, x, t_graph)
            x, v = self._apply_guidance(batch, x, t_graph, v, guidance_fn)

            if self.config.solver == "euler":
                x = x + dt * v
            else:
                x_pred = x + dt * v
                t_next = torch.full((num_graphs,), float(t_grid[i + 1]), device=x.device)
                v_next = self._model_v(batch, x_pred, t_next)
                x = x + 0.5 * dt * (v + v_next)

            if not torch.isfinite(x).all():
                raise FloatingPointError("Non-finite coordinates during ODE integration. Reduce guidance scale.")

        return x
