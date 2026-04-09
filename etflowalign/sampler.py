"""ODE sampling for ETFlowAlign with safe guidance injection policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import torch
from torch import Tensor

from .model import AlignmentBatch, ETFlowAlignModel

GuidanceFn = Callable[[AlignmentBatch, Tensor, Tensor], Tensor]
RefineFn = Callable[[AlignmentBatch, Tensor, Tensor], Tensor]


@dataclass
class ODESamplerConfig:
    """Configuration for ODE inference.

    Attributes:
        n_steps: Number of integration steps from t_start to t_end.
        t_start: Initial integration time.
        t_end: Final integration time.
        solver: Numerical solver (Euler or Heun).
        guidance_scale: Global multiplier for external guidance.
        guidance_mode: How guidance is injected.
        max_guidance_norm: Per-atom norm cap for guidance vectors.
        pc_corrector_step_scale: Step scale for predictor-corrector correction.
        adaptive_dt: Enable simple adaptive step scaling by velocity norm.
        adaptive_dt_max_scale: Maximum multiplier for adaptive dt.
    """
    n_steps: int = 50
    t_start: float = 0.0
    t_end: float = 1.0
    solver: Literal["euler", "heun"] = "heun"
    guidance_scale: float = 0.0
    guidance_mode: Literal["vector_field", "predictor_corrector"] = "vector_field"
    max_guidance_norm: float = 5.0
    pc_corrector_step_scale: float = 1.0
    adaptive_dt: bool = False
    adaptive_dt_max_scale: float = 2.0
    log_trajectory: bool = False


class ETFlowAlignSampler:
    """ODE sampler for alignment flow matching.

    Guidance mode:
        - vector_field: add guidance directly to velocity before state update.
        - predictor_corrector: apply guidance as a separate correction step.
    """

    def __init__(self, model: ETFlowAlignModel, config: ODESamplerConfig) -> None:
        """Bind trained model and sampling hyperparameters."""
        self.model = model
        self.config = config

    def _clip_guidance(self, g: Tensor) -> Tensor:
        """Clip guidance vectors by L2 norm for numerical stability."""
        norm = torch.norm(g, dim=-1, keepdim=True).clamp_min(1e-8)
        return g * (self.config.max_guidance_norm / norm).clamp(max=1.0)

    def _model_v(self, batch: AlignmentBatch, x: Tensor, t_graph: Tensor) -> Tensor:
        """Evaluate model velocity at state x and time t."""
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

    def _apply_guidance(
        self, batch: AlignmentBatch, x: Tensor, t_graph: Tensor, v: Tensor, guidance_fn: Optional[GuidanceFn]
    ) -> tuple[Tensor, Tensor]:
        """Apply optional external guidance based on configured mode.

        Returns:
            Tuple ``(x, v)`` after guidance adjustment.
        """
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
        x = x + (
            self.config.pc_corrector_step_scale
            * self.config.guidance_scale
            / max(1, self.config.n_steps)
        ) * g
        return x, v

    @torch.no_grad()
    def sample(
        self,
        batch: AlignmentBatch,
        x0: Tensor,
        guidance_fn: Optional[GuidanceFn] = None,
        refine_fn: Optional[RefineFn] = None,
    ) -> Tensor:
        """Integrate ODE from source state to final aligned state.

        Args:
            batch: Conditioning batch.
            x0: Initial query coordinates.
            guidance_fn: Optional external guidance callback.

        Returns:
            Final query coordinates after ODE integration.
        """
        num_graphs = int(batch.query_batch.max().item()) + 1
        t_grid = torch.linspace(self.config.t_start, self.config.t_end, self.config.n_steps + 1, device=x0.device)

        x = x0.clone()
        self.last_trace: list[dict[str, float]] = []
        for i in range(self.config.n_steps):
            t = t_grid[i]
            dt = t_grid[i + 1] - t
            t_graph = torch.full((num_graphs,), float(t), device=x.device)

            v = self._model_v(batch, x, t_graph)
            x, v = self._apply_guidance(batch, x, t_graph, v, guidance_fn)

            dt_eff = dt
            if self.config.adaptive_dt:
                v_norm = torch.norm(v, dim=-1).mean().clamp_min(1e-8)
                dt_eff = dt / v_norm
                dt_eff = dt_eff.clamp(min=0.0, max=float(dt) * self.config.adaptive_dt_max_scale)

            if self.config.solver == "euler":
                x = x + dt_eff * v
            else:
                x_pred = x + dt_eff * v
                t_next = torch.full((num_graphs,), float(t_grid[i + 1]), device=x.device)
                v_next = self._model_v(batch, x_pred, t_next)
                # Guidance should also affect corrected slope in Heun mode.
                _, v_next = self._apply_guidance(batch, x_pred, t_next, v_next, guidance_fn)
                x = x + 0.5 * dt_eff * (v + v_next)

            # Optional clean-state refinement hook (for physics/UFF style correctors).
            if refine_fn is not None:
                x = refine_fn(batch, t_graph, x)

            if not torch.isfinite(x).all():
                raise FloatingPointError("Non-finite coordinates during ODE integration. Reduce guidance scale.")

            if self.config.log_trajectory:
                self.last_trace.append(
                    {
                        "step": float(i),
                        "t": float(t.item()),
                        "x_norm": float(torch.norm(x, dim=-1).mean().item()),
                        "v_norm": float(torch.norm(v, dim=-1).mean().item()),
                    }
                )

        return x
