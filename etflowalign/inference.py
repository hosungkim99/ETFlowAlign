
"""Inference entrypoints for ETFlowAlign multi-sample generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor

from .model import AlignmentBatch
from .sampler import ETFlowAlignSampler, GuidanceFn

RankFn = Callable[[Tensor], Tensor]


@dataclass
class InferenceConfig:
    num_samples: int = 8


def generate_candidates(
    sampler: ETFlowAlignSampler,
    batch: AlignmentBatch,
    source_sampler: Callable[[AlignmentBatch], Tensor],
    guidance_fn: Optional[GuidanceFn] = None,
    num_samples: int = 8,
) -> Tensor:
    """Generate multiple aligned query candidates.

    Returns:
        Tensor of shape [num_samples, num_query_atoms, 3]
    """
    outputs = []
    for _ in range(num_samples):
        x0 = source_sampler(batch)
        xT = sampler.sample(batch=batch, x0=x0, guidance_fn=guidance_fn)
        outputs.append(xT)
    return torch.stack(outputs, dim=0)


def rank_candidates(candidates: Tensor, rank_fn: Optional[RankFn] = None) -> tuple[Tensor, Tensor]:
    """Rank candidates with pluggable scoring logic.

    TODO:
        Connect task-specific ranking metrics (e.g., TanimotoCombo) in downstream code.
    """
    if rank_fn is None:
        scores = torch.zeros(candidates.size(0), device=candidates.device)
    else:
        scores = rank_fn(candidates)

    order = torch.argsort(scores, descending=True)
    return candidates[order], scores[order]
