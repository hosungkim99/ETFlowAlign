"""Flow-matching objective and alignment-specific probability path for ETFlowAlign."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from .model import AlignmentBatch, ETFlowAlignModel
from .validation import validate_alignment_batch


@dataclass
class FlowMatchingConfig:
    sigma: float = 0.05
    source_type: Literal["gaussian", "reference_anchored", "query_perturbed"] = "reference_anchored"
    source_noise_scale: float = 0.5
    time_eps: float = 1e-4
    use_kabsch_alignment: bool = True
    harmonic_prior_strength: float = 0.0
    center_source: bool = True
    center_target: bool = True


class AlignmentFlowMatcher:
    def __init__(self, config: FlowMatchingConfig) -> None:
        self.config = config

    def sample_time(self, num_graphs: int, device: torch.device) -> Tensor:
        return torch.empty(num_graphs, device=device).uniform_(self.config.time_eps, 1.0 - self.config.time_eps)

    def sigma_t(self, t_node: Tensor) -> Tensor:
        return self.config.sigma * torch.sqrt((t_node * (1.0 - t_node)).clamp_min(1e-8))

    def sigma_dot_t(self, t_node: Tensor) -> Tensor:
        denom = torch.sqrt((t_node * (1.0 - t_node)).clamp_min(1e-8))
        return self.config.sigma * 0.5 * (1.0 - 2.0 * t_node) / denom

    def sample_source(self, batch: AlignmentBatch, target_query_pos: Tensor | None = None) -> Tensor:
        stype = self.config.source_type
        if stype == "gaussian":
            return torch.randn_like(batch.query_pos)
        if stype == "query_perturbed":
            if target_query_pos is None:
                raise ValueError("source_type='query_perturbed' requires target_query_pos.")
            return target_query_pos + self.config.source_noise_scale * torch.randn_like(target_query_pos)
        if batch.reference_pos is None or batch.reference_batch is None or batch.reference_batch.numel() == 0:
            return torch.randn_like(batch.query_pos)
        num_graphs = int(batch.reference_batch.max().item()) + 1
        center = torch.zeros(num_graphs, 3, device=batch.reference_pos.device, dtype=batch.reference_pos.dtype)
        count = torch.zeros(num_graphs, 1, device=batch.reference_pos.device, dtype=batch.reference_pos.dtype)
        center.index_add_(0, batch.reference_batch, batch.reference_pos)
        count.index_add_(0, batch.reference_batch, torch.ones_like(batch.reference_batch, dtype=batch.reference_pos.dtype).unsqueeze(-1))
        center = center / count.clamp_min(1.0)
        return center[batch.query_batch] + self.config.source_noise_scale * torch.randn_like(batch.query_pos)

    def _kabsch_align_source_to_target(self, x0: Tensor, target_query_pos: Tensor, batch_index: Tensor) -> Tensor:
        out = x0.clone()
        for g in batch_index.unique(sorted=True):
            mask = batch_index == g
            p = x0[mask]
            q = target_query_pos[mask]
            if p.size(0) < 2:
                out[mask] = q
                continue
            pc = p - p.mean(0, keepdim=True)
            qc = q - q.mean(0, keepdim=True)
            h = pc.transpose(0, 1) @ qc
            u, _, vT = torch.linalg.svd(h)
            r = vT.transpose(0, 1) @ u.transpose(0, 1)
            if torch.det(r) < 0:
                vT[-1, :] *= -1
                r = vT.transpose(0, 1) @ u.transpose(0, 1)
            aligned = pc @ r + q.mean(0, keepdim=True)
            out[mask] = aligned
        return out

    def _apply_harmonic_prior_if_needed(self, x0: Tensor, batch: AlignmentBatch) -> Tensor:
        if self.config.harmonic_prior_strength <= 0:
            return x0
        center = torch.zeros(int(batch.query_batch.max().item()) + 1, 3, device=x0.device, dtype=x0.dtype)
        count = torch.zeros(center.size(0), 1, device=x0.device, dtype=x0.dtype)
        center.index_add_(0, batch.query_batch, x0)
        count.index_add_(0, batch.query_batch, torch.ones(x0.size(0), 1, device=x0.device, dtype=x0.dtype))
        center = center / count.clamp_min(1.0)
        return x0 - self.config.harmonic_prior_strength * (x0 - center[batch.query_batch])

    def _center_by_graph(self, x: Tensor, batch_index: Tensor) -> Tensor:
        num_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
        if num_graphs == 0:
            return x
        mean = torch.zeros(num_graphs, x.size(-1), device=x.device, dtype=x.dtype)
        count = torch.zeros(num_graphs, 1, device=x.device, dtype=x.dtype)
        mean.index_add_(0, batch_index, x)
        count.index_add_(0, batch_index, torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype))
        mean = mean / count.clamp_min(1.0)
        return x - mean[batch_index]

    def build_training_state(self, batch: AlignmentBatch, target_query_pos: Tensor, t_graph: Tensor) -> tuple[Tensor, Tensor]:
        validate_alignment_batch(batch, target_query_pos=target_query_pos, require_target=True)
        t_node = t_graph[batch.query_batch]
        x0 = self.sample_source(batch=batch, target_query_pos=target_query_pos)
        if self.config.center_source:
            x0 = self._center_by_graph(x0, batch.query_batch)
        x1 = target_query_pos
        if self.config.center_target:
            x1 = self._center_by_graph(x1, batch.query_batch)
        x0 = self._apply_harmonic_prior_if_needed(x0=x0, batch=batch)
        if self.config.use_kabsch_alignment:
            x0 = self._kabsch_align_source_to_target(x0=x0, target_query_pos=x1, batch_index=batch.query_batch)
        eps = torch.randn_like(x1)
        sigma = self.sigma_t(t_node).unsqueeze(-1)
        sigma_dot = self.sigma_dot_t(t_node).unsqueeze(-1)
        x_t = (1.0 - t_node).unsqueeze(-1) * x0 + t_node.unsqueeze(-1) * x1 + sigma * eps
        u_t = (x1 - x0) + sigma_dot * eps
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


def flow_matching_step(model: ETFlowAlignModel, matcher: AlignmentFlowMatcher, batch: AlignmentBatch, target_query_pos: Tensor) -> Tensor:
    validate_alignment_batch(batch, target_query_pos=target_query_pos, require_target=True)
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
        pocket_batch=batch.pocket_batch,
    )
    pred_v = model(step_batch, t_graph=t_graph)
    return matcher.loss(pred_v=pred_v, target_u=u_t, batch_index=batch.query_batch)
