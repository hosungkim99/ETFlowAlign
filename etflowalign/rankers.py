"""Ranking utilities for ETFlowAlign inference.

This module provides a plugin-friendly ranking interface to combine
TanimotoCombo-like shape/color similarity with docking/physics-inspired
scores. The default implementation is dependency-light and can be
replaced by external plugins for production use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor

from .model import AlignmentBatch

# Each plugin receives all candidates at once and returns one score per candidate.
ScoreFn = Callable[[Tensor, AlignmentBatch], Tensor]


@dataclass
class RankerConfig:
    """Configuration for plugin-based candidate ranking."""

    tanimoto_weight: float = 1.0
    physics_weight: float = 1.0
    shape_sigma: float = 1.2
    clash_cutoff: float = 1.0
    clash_penalty_weight: float = 2.0


@dataclass
class RankBreakdown:
    """Ranking score decomposition per candidate."""

    total: Tensor
    tanimoto: Tensor
    physics: Tensor


class PluginRanker:
    """Composable ranker: TanimotoCombo-like score + docking/physics score.

    Production users can inject true OpenEye ROCS TanimotoCombo and docking
    callbacks through ``tanimoto_plugin`` / ``physics_plugin`` while keeping
    the same external API.
    """

    def __init__(
        self,
        config: Optional[RankerConfig] = None,
        tanimoto_plugin: Optional[ScoreFn] = None,
        physics_plugin: Optional[ScoreFn] = None,
    ) -> None:
        self.config = config or RankerConfig()
        self.tanimoto_plugin = tanimoto_plugin
        self.physics_plugin = physics_plugin

    def score_tanimoto(self, candidates: Tensor, batch: AlignmentBatch) -> Tensor:
        """Compute TanimotoCombo-like score.

        Default fallback: Gaussian-overlap shape Tanimoto + atom-type matched
        "color" overlap (proxy for pharmacophore/color contribution).
        """
        if self.tanimoto_plugin is not None:
            return self.tanimoto_plugin(candidates, batch)
        return proxy_tanimoto_combo(
            candidates=candidates,
            reference_pos=batch.reference_pos,
            query_atom_type=batch.query_atom_type,
            reference_atom_type=batch.reference_atom_type,
            sigma=self.config.shape_sigma,
        )

    def score_physics(self, candidates: Tensor, batch: AlignmentBatch) -> Tensor:
        """Compute docking/physics-inspired score."""
        if self.physics_plugin is not None:
            return self.physics_plugin(candidates, batch)
        return proxy_physics_score(
            candidates=candidates,
            pocket_pos=batch.pocket_pos,
            reference_pos=batch.reference_pos,
            clash_cutoff=self.config.clash_cutoff,
            clash_penalty_weight=self.config.clash_penalty_weight,
        )

    def score(self, candidates: Tensor, batch: AlignmentBatch) -> RankBreakdown:
        """Return combined ranking score and component breakdown."""
        tanimoto = self.score_tanimoto(candidates, batch)
        physics = self.score_physics(candidates, batch)
        total = self.config.tanimoto_weight * tanimoto + self.config.physics_weight * physics
        return RankBreakdown(total=total, tanimoto=tanimoto, physics=physics)


class LegacyReferenceMseRanker:
    """Backward-compatible ranker from initial scaffold (reference MSE)."""

    def score(self, candidates: Tensor, batch: AlignmentBatch) -> RankBreakdown:
        if batch.reference_pos is None:
            score = torch.zeros(candidates.size(0), device=candidates.device)
        else:
            diff = candidates - batch.reference_pos.unsqueeze(0)
            mse = (diff * diff).mean(dim=(1, 2))
            score = -mse
        zeros = torch.zeros_like(score)
        return RankBreakdown(total=score, tanimoto=zeros, physics=zeros)


def _pairwise_min_dist(a: Tensor, b: Tensor) -> Tensor:
    """Return minimum distance from each point in ``a`` to set ``b``."""
    if b is None or b.numel() == 0:
        return torch.full((a.size(0),), 5.0, device=a.device, dtype=a.dtype)
    return torch.cdist(a, b).min(dim=-1).values


def proxy_physics_score(
    candidates: Tensor,
    pocket_pos: Optional[Tensor],
    reference_pos: Optional[Tensor],
    clash_cutoff: float,
    clash_penalty_weight: float,
) -> Tensor:
    """Cheap docking/physics proxy.

    Encourages proximity to pocket points (or reference fallback) while
    penalizing severe clashes.
    """
    receptor = pocket_pos if pocket_pos is not None else reference_pos
    scores = []
    for cand in candidates:
        min_dist = _pairwise_min_dist(cand, receptor)
        affinity = -min_dist.mean()
        clash = torch.relu(torch.tensor(clash_cutoff, device=cand.device, dtype=cand.dtype) - min_dist).mean()
        scores.append(affinity - clash_penalty_weight * clash)
    return torch.stack(scores)


def _gaussian_overlap(query: Tensor, ref: Tensor, sigma: float) -> Tensor:
    """Compute symmetric Gaussian overlap between two point sets."""
    if ref is None or ref.numel() == 0:
        return torch.tensor(0.0, device=query.device, dtype=query.dtype)

    gamma = 1.0 / (2.0 * (sigma * sigma))
    d2_qr = torch.cdist(query, ref) ** 2
    d2_qq = torch.cdist(query, query) ** 2
    d2_rr = torch.cdist(ref, ref) ** 2

    ov_qr = torch.exp(-gamma * d2_qr).sum()
    ov_qq = torch.exp(-gamma * d2_qq).sum()
    ov_rr = torch.exp(-gamma * d2_rr).sum()
    return (2.0 * ov_qr) / (ov_qq + ov_rr + 1e-8)


def _atom_match_mask(query_atom_type: Optional[Tensor], reference_atom_type: Optional[Tensor], device: torch.device) -> Optional[Tensor]:
    if query_atom_type is None or reference_atom_type is None:
        return None
    return (query_atom_type[:, None] == reference_atom_type[None, :]).to(device=device)


def proxy_tanimoto_combo(
    candidates: Tensor,
    reference_pos: Optional[Tensor],
    query_atom_type: Optional[Tensor],
    reference_atom_type: Optional[Tensor],
    sigma: float,
) -> Tensor:
    """Approximate TanimotoCombo score from shape + atom-type matched color terms."""
    if reference_pos is None or reference_pos.numel() == 0:
        return torch.zeros(candidates.size(0), device=candidates.device)

    mask = _atom_match_mask(query_atom_type, reference_atom_type, device=candidates.device)
    scores = []
    for cand in candidates:
        shape = _gaussian_overlap(cand, reference_pos, sigma=sigma)
        if mask is None:
            color = torch.tensor(0.0, device=cand.device, dtype=cand.dtype)
        else:
            d2 = torch.cdist(cand, reference_pos) ** 2
            gamma = 1.0 / (2.0 * (sigma * sigma))
            weighted = torch.exp(-gamma * d2) * mask
            # Normalize to [0,1]-ish range by number of atoms.
            color = weighted.sum() / (mask.sum().clamp_min(1.0) + 1e-8)
        scores.append(shape + color)
    return torch.stack(scores)
