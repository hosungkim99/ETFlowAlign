"""Flow-matching objective and alignment-specific probability path for ETFlowAlign."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from .model import AlignmentBatch, ETFlowAlignModel
from .utils import segment_mean


@dataclass
class FlowMatchingConfig:
    """Configuration for alignment-specific flow matching.

    Attributes:
        sigma: Noise scale used in the interpolating probability path.
        source_type: Strategy for source state ``x0``.
        source_noise_scale: Noise amount for source samplers.
        time_eps: Numerical margin to avoid exactly 0/1 time values.
    """

    sigma: float = 0.05
    source_type: Literal["gaussian", "reference_anchored", "query_perturbed", "rigid_reference_perturbed"] = "reference_anchored"
    source_noise_scale: float = 0.5
    time_eps: float = 1e-4
    time_weighting: Literal["uniform", "mid"] = "uniform"
    allow_query_perturbed_for_inference: bool = False


class AlignmentFlowMatcher:
    """Construct x_t and vector-field target u_t for alignment flow matching.

    Path:
        x_t = (1 - t) * x0 + t * x1 + sigma_t * eps
        u_t = (x1 - x0) + sigma_dot_t * eps
    where x1 is target aligned query coordinates.
    """

    def __init__(self, config: FlowMatchingConfig) -> None:
        """Store flow-matching hyperparameters."""
        self.config = config

    def sample_time(self, num_graphs: int, device: torch.device) -> Tensor:
        """Uniformly sample training times for each graph in a batch."""
        return torch.empty(num_graphs, device=device).uniform_(self.config.time_eps, 1.0 - self.config.time_eps)

    def sigma_t(self, t_node: Tensor) -> Tensor:
        """Compute path noise scale ``sigma(t)`` at node-expanded times."""
        return self.config.sigma * torch.sqrt((t_node * (1.0 - t_node)).clamp_min(1e-8))

    def sigma_dot_t(self, t_node: Tensor) -> Tensor:
        """Compute derivative ``d sigma(t) / dt`` used in target vector field."""
        denom = torch.sqrt((t_node * (1.0 - t_node)).clamp_min(1e-8))
        return self.config.sigma * 0.5 * (1.0 - 2.0 * t_node) / denom

    def sample_source(self, batch: AlignmentBatch, target_query_pos: Tensor | None = None) -> Tensor:
        """Sample source state ``x0`` for flow matching.

        Modes:
            - gaussian: isotropic Gaussian source.
            - query_perturbed: perturb target with Gaussian noise.
            - reference_anchored: sample around reference center of mass.
            - rigid_reference_perturbed: rigid perturb reference pose (alignment-aware prior).
        """
        stype = self.config.source_type
        if stype == "gaussian":
            return torch.randn_like(batch.query_pos)

        if stype == "query_perturbed" and target_query_pos is not None:
            # NOTE: primarily for training ablation; not a pure generative prior.
            return target_query_pos + self.config.source_noise_scale * torch.randn_like(target_query_pos)

        if stype == "rigid_reference_perturbed":
            rigid = self._rigid_reference_prior(batch)
            if rigid is not None:
                return rigid + self.config.source_noise_scale * torch.randn_like(batch.query_pos)

        # reference_anchored fallback
        if batch.reference_pos is None or batch.reference_batch is None or batch.reference_batch.numel() == 0:
            return torch.randn_like(batch.query_pos)

        num_graphs = int(batch.reference_batch.max().item()) + 1
        center = segment_mean(batch.reference_pos, batch.reference_batch, num_graphs)
        return center[batch.query_batch] + self.config.source_noise_scale * torch.randn_like(batch.query_pos)

    def sample_inference_source(self, batch: AlignmentBatch) -> Tensor:
        """Sample source for inference-time generation.

        Prevents accidentally using query_perturbed prior during true generation.
        """
        if self.config.source_type == "query_perturbed" and not self.config.allow_query_perturbed_for_inference:
            backup = FlowMatchingConfig(
                sigma=self.config.sigma,
                source_type="reference_anchored",
                source_noise_scale=self.config.source_noise_scale,
                time_eps=self.config.time_eps,
                time_weighting=self.config.time_weighting,
                allow_query_perturbed_for_inference=self.config.allow_query_perturbed_for_inference,
            )
            return AlignmentFlowMatcher(backup).sample_source(batch=batch, target_query_pos=None)
        return self.sample_source(batch=batch, target_query_pos=None)

    def _rigid_reference_prior(self, batch: AlignmentBatch) -> Tensor | None:
        """Alignment-aware prior via per-graph rigid perturbation of reference coordinates."""
        if batch.reference_pos is None or batch.reference_batch is None:
            return None
        if batch.query_pos.size(0) != batch.reference_pos.size(0):
            return None
        if not torch.equal(batch.query_batch, batch.reference_batch):
            return None

        x0 = torch.zeros_like(batch.query_pos)
        num_graphs = int(batch.query_batch.max().item()) + 1
        for g in range(num_graphs):
            idx = torch.where(batch.query_batch == g)[0]
            ref = batch.reference_pos[idx]
            com = ref.mean(dim=0, keepdim=True)
            centered = ref - com
            q, _ = torch.linalg.qr(torch.randn(3, 3, device=ref.device, dtype=ref.dtype))
            if torch.det(q) < 0:
                q[:, -1] *= -1
            trans = torch.randn(1, 3, device=ref.device, dtype=ref.dtype) * self.config.source_noise_scale
            x0[idx] = centered @ q.T + com + trans
        return x0

    def build_training_state(self, batch: AlignmentBatch, target_query_pos: Tensor, t_graph: Tensor) -> tuple[Tensor, Tensor]:
        """Construct noisy path state and vector-field regression target.

        Args:
            batch: Input batch containing conditioning information.
            target_query_pos: Ground-truth aligned query positions ``x1`` (already aligned target).
            t_graph: Time per graph ``[B]``.

        Returns:
            x_t: Interpolated noisy states at sampled time.
            u_t: Target velocity vectors for flow matching.
        """
        t_node = t_graph[batch.query_batch]
        x0 = self.sample_source(batch=batch, target_query_pos=target_query_pos)
        eps = torch.randn_like(target_query_pos)

        sigma = self.sigma_t(t_node).unsqueeze(-1)
        sigma_dot = self.sigma_dot_t(t_node).unsqueeze(-1)
        x_t = (1.0 - t_node).unsqueeze(-1) * x0 + t_node.unsqueeze(-1) * target_query_pos + sigma * eps
        u_t = (target_query_pos - x0) + sigma_dot * eps
        return x_t, u_t

    def loss(self, pred_v: Tensor, target_u: Tensor, batch_index: Tensor, t_graph: Tensor | None = None) -> Tensor:
        """Compute per-graph averaged MSE over vector fields.

        If ``time_weighting == 'mid'``, emphasize mid-time samples (near t=0.5).
        """
        per_atom = ((pred_v - target_u) ** 2).sum(dim=-1)
        num_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
        if num_graphs == 0:
            return per_atom.mean() * 0.0

        graph_sum = torch.zeros(num_graphs, device=per_atom.device, dtype=per_atom.dtype)
        graph_cnt = torch.zeros(num_graphs, device=per_atom.device, dtype=per_atom.dtype)
        graph_sum.index_add_(0, batch_index, per_atom)
        graph_cnt.index_add_(0, batch_index, torch.ones_like(per_atom))
        graph_loss = graph_sum / graph_cnt.clamp_min(1.0)

        if t_graph is None or self.config.time_weighting == "uniform":
            return graph_loss.mean()

        # Mid-time emphasis curriculum.
        w = 1.0 - (2.0 * t_graph - 1.0).abs()  # max at t=0.5
        w = w / w.sum().clamp_min(1e-8)
        return (w * graph_loss).sum()


def flow_matching_step(
    model: ETFlowAlignModel,
    matcher: AlignmentFlowMatcher,
    batch: AlignmentBatch,
    target_query_pos: Tensor,
) -> Tensor:
    """Single ETFlowAlign flow-matching training step.

    This helper performs:
        1) time sampling,
        2) path/target construction,
        3) model forward on ``x_t``,
        4) vector-field regression loss.
    """
    num_graphs = int(batch.query_batch.max().item()) + 1
    t_graph = matcher.sample_time(num_graphs=num_graphs, device=batch.query_pos.device)
    x_t, u_t = matcher.build_training_state(batch=batch, target_query_pos=target_query_pos, t_graph=t_graph)

    step_batch = AlignmentBatch(
        query_pos=x_t,
        query_atom_type=batch.query_atom_type,
        query_batch=batch.query_batch,
        reference_pos=batch.reference_pos,
        reference_atom_type=batch.reference_atom_type,
        reference_batch=batch.reference_batch,
        pocket_pos=batch.pocket_pos,
    )
    pred_v = model(step_batch, t_graph=t_graph)
    return matcher.loss(pred_v=pred_v, target_u=u_t, batch_index=batch.query_batch, t_graph=t_graph)
