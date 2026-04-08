
"""Flow-matching objective and probability path design for ETFlowAlign."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from .model import AlignmentBatch, ETFlowAlignModel


@dataclass
class FlowMatchingConfig:
    """Configuration for alignment-specific flow matching."""

    sigma: float = 0.05
    source_type: Literal["gaussian", "reference_anchored"] = "gaussian"
    time_eps: float = 1e-4


class AlignmentFlowMatcher:
    """Constructs x_t and target vector field u_t for training.

    Path design (scaffold):
        x_t = (1 - t) * x0 + t * x1 + sigma_t * eps
        u_t = d/dt x_t = (x1 - x0) + sigma_dot_t * eps
    where x1 is ground-truth aligned query coordinates.
    """

    def __init__(self, config: FlowMatchingConfig) -> None:
        self.config = config

    def sample_time(self, num_graphs: int, device: torch.device) -> Tensor:
        return torch.empty(num_graphs, device=device).uniform_(self.config.time_eps, 1.0 - self.config.time_eps)

    def sigma_t(self, t_node: Tensor) -> Tensor:
        return self.config.sigma * torch.sqrt((t_node * (1.0 - t_node)).clamp_min(1e-8))

    def sigma_dot_t(self, t_node: Tensor) -> Tensor:
        denom = torch.sqrt((t_node * (1.0 - t_node)).clamp_min(1e-8))
        return self.config.sigma * 0.5 * (1.0 - 2.0 * t_node) / denom

    def sample_source(self, batch: AlignmentBatch) -> Tensor:
        if self.config.source_type == "gaussian":
            return torch.randn_like(batch.query_pos)

        # reference_anchored: initialize around reference COM per graph.
        if batch.reference_pos is None or batch.reference_batch is None:
            return torch.randn_like(batch.query_pos)

        ref = batch.reference_pos
        ref_batch = batch.reference_batch
        num_graphs = int(ref_batch.max().item()) + 1
        centers = torch.zeros(num_graphs, 3, device=ref.device, dtype=ref.dtype)
        counts = torch.zeros(num_graphs, 1, device=ref.device, dtype=ref.dtype)
        centers.index_add_(0, ref_batch, ref)
        counts.index_add_(0, ref_batch, torch.ones_like(ref_batch, dtype=ref.dtype).unsqueeze(-1))
        centers = centers / counts.clamp_min(1.0)
        return centers[batch.query_batch] + 0.5 * torch.randn_like(batch.query_pos)

    def build_training_state(self, batch: AlignmentBatch, target_query_pos: Tensor, t_graph: Tensor) -> tuple[Tensor, Tensor]:
        t_node = t_graph[batch.query_batch]
        x0 = self.sample_source(batch)
        eps = torch.randn_like(target_query_pos)
        sigma = self.sigma_t(t_node).unsqueeze(-1)
        sigma_dot = self.sigma_dot_t(t_node).unsqueeze(-1)

        x_t = (1.0 - t_node).unsqueeze(-1) * x0 + t_node.unsqueeze(-1) * target_query_pos + sigma * eps
        u_t = (target_query_pos - x0) + sigma_dot * eps
        return x_t, u_t

    def loss(self, pred_v: Tensor, target_u: Tensor, batch_index: Tensor) -> Tensor:
        per_atom = ((pred_v - target_u) ** 2).sum(dim=-1)
        num_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
        if num_graphs == 0:
            return per_atom.mean() * 0.0

        graph_sum = torch.zeros(num_graphs, device=per_atom.device, dtype=per_atom.dtype)
        graph_cnt = torch.zeros(num_graphs, device=per_atom.device, dtype=per_atom.dtype)
        graph_sum.index_add_(0, batch_index, per_atom)
        graph_cnt.index_add_(0, batch_index, torch.ones_like(per_atom))
        return (graph_sum / graph_cnt.clamp_min(1.0)).mean()


def flow_matching_step(
    model: ETFlowAlignModel,
    matcher: AlignmentFlowMatcher,
    batch: AlignmentBatch,
    target_query_pos: Tensor,
) -> Tensor:
    """Single ETFlowAlign flow-matching training step."""
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
    return matcher.loss(pred_v=pred_v, target_u=u_t, batch_index=batch.query_batch)
