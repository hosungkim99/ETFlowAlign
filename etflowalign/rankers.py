"""ETFlowAlign 추론을 위한 랭킹 유틸리티.

이 모듈은 TanimotoCombo 방식의 형상/색상 유사도와 도킹/물리 기반 점수를
결합하는 플러그인 친화적 랭킹 인터페이스를 제공한다. 기본 구현은 의존성이
최소화되어 있으며, 프로덕션 환경에서 외부 플러그인으로 교체할 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor

from .model import AlignmentBatch

# 각 플러그인은 모든 후보를 한 번에 받아 후보당 점수 하나를 반환한다.
ScoreFn = Callable[[Tensor, AlignmentBatch], Tensor]


@dataclass
class RankerConfig:
    """플러그인 기반 후보 랭킹을 위한 설정."""

    tanimoto_weight: float = 1.0
    physics_weight: float = 1.0
    shape_sigma: float = 1.2
    clash_cutoff: float = 1.0
    clash_penalty_weight: float = 2.0


@dataclass
class RankBreakdown:
    """후보별 랭킹 점수 분해 결과."""

    total: Tensor
    tanimoto: Tensor
    physics: Tensor


class PluginRanker:
    """조합 가능한 랭커: TanimotoCombo 방식 점수 + 도킹/물리 점수.

    프로덕션 사용자는 ``tanimoto_plugin`` / ``physics_plugin``을 통해
    실제 OpenEye ROCS TanimotoCombo 및 도킹 콜백을 주입하면서도
    동일한 외부 API를 유지할 수 있다.
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
        """TanimotoCombo 방식의 점수를 계산한다.

        기본 폴백: 가우시안 오버랩 기반 형상 Tanimoto + 원자 타입 일치 기반
        "색상" 오버랩 (파마코포어/색상 기여도의 대리값).
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
        """도킹/물리 기반 점수를 계산한다."""
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
        """결합된 랭킹 점수와 구성 요소 분해 결과를 반환한다."""
        tanimoto = self.score_tanimoto(candidates, batch)
        physics = self.score_physics(candidates, batch)
        total = self.config.tanimoto_weight * tanimoto + self.config.physics_weight * physics
        return RankBreakdown(total=total, tanimoto=tanimoto, physics=physics)


class LegacyReferenceMseRanker:
    """초기 스캐폴드로부터의 하위 호환 랭커 (참조 MSE 기반)."""

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
    """``a``의 각 점에서 집합 ``b``까지의 최소 거리를 반환한다."""
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
    """간단한 도킹/물리 대리 점수.

    포켓 포인트(또는 참조 폴백)와의 근접성을 장려하면서
    심각한 충돌에 패널티를 부과한다.
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
    """두 점 집합 간의 대칭적 가우시안 오버랩을 계산한다."""
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
    """형상 + 원자 타입 매칭 색상 항으로부터 근사 TanimotoCombo 점수를 계산한다."""
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
            # 원자 수를 기준으로 [0,1] 범위로 정규화한다.
            color = weighted.sum() / (mask.sum().clamp_min(1.0) + 1e-8)
        scores.append(shape + color)
    return torch.stack(scores)
