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
    """플러그인 기반 후보 랭킹을 위한 가중치 및 임계값 설정 컨테이너.

    생성이유:
        Tanimoto/물리 항의 상대 가중치와 형상/충돌 관련 임계값을 한 곳에 모아
        랭커 호출 간 일관된 점수 정책을 유지하기 위해 dataclass로 분리한다.

    역할:
        PluginRanker와 프록시 점수 함수들이 참조하는 하이퍼파라미터를 보관한다.

    필드:
        tanimoto_weight (float): 전체 점수에서 Tanimoto(형상+색상) 항의 가중치.
        physics_weight (float): 전체 점수에서 도킹/물리 항의 가중치.
        shape_sigma (float): 가우시안 오버랩의 너비(σ). 형상/색상 점수의
            거리 민감도를 결정한다.
        clash_cutoff (float): 충돌로 간주하는 최소 거리 임계값(Å). 이보다 가까운
            접촉에 패널티가 부과된다.
        clash_penalty_weight (float): 물리 점수에서 충돌 패널티의 가중치.

    파이프라인 단계:
        10단계(랭킹) - 설정 보조.
    """

    tanimoto_weight: float = 1.0
    physics_weight: float = 1.0
    shape_sigma: float = 1.2
    clash_cutoff: float = 1.0
    clash_penalty_weight: float = 2.0


@dataclass
class RankBreakdown:
    """후보별 랭킹 점수와 그 구성 요소 분해 결과를 담는 컨테이너.

    생성이유:
        최종 점수만 반환하면 어떤 요인(형상/색상 vs 물리)이 점수를 좌우했는지
        알 수 없다. 디버깅/분석을 위해 구성 요소별 점수를 함께 보존한다.

    역할:
        PluginRanker.score 등 랭커의 점수 산출 결과를 표준 형태로 전달한다.

    필드:
        total (Tensor): 가중 합산된 최종 후보 점수 ``[num_samples]``.
        tanimoto (Tensor): TanimotoCombo(형상+색상) 항 점수 ``[num_samples]``.
        physics (Tensor): 도킹/물리 항 점수 ``[num_samples]``.

    파이프라인 단계:
        10단계(랭킹) - 결과 전달.
    """

    total: Tensor
    tanimoto: Tensor
    physics: Tensor


class PluginRanker:
    """조합 가능한 랭커: TanimotoCombo 방식 점수 + 도킹/물리 점수.

    프로덕션 사용자는 ``tanimoto_plugin`` / ``physics_plugin``을 통해
    실제 OpenEye ROCS TanimotoCombo 및 도킹 콜백을 주입하면서도
    동일한 외부 API를 유지할 수 있다.

    생성이유:
        랭킹 점수 구성 요소(형상 유사도, 물리/도킹)를 플러그인으로 교체 가능한
        형태로 추상화하여, 의존성이 가벼운 기본 구현으로도 동작하고 프로덕션에서
        실제 도구로 대체할 수 있게 하기 위함이다.

    역할:
        랭킹(10단계)의 기본 랭커. Tanimoto 항과 물리 항을 각각 계산하고
        가중 합산하여 RankBreakdown으로 반환한다.

    정의되는 주요 변수와 의미:
        config (RankerConfig): 가중치/임계값 등 점수 정책 설정.
        tanimoto_plugin (Optional[ScoreFn]): 주입 시 기본 형상 점수를 대체하는
            외부 Tanimoto 점수 콜백.
        physics_plugin (Optional[ScoreFn]): 주입 시 기본 물리 점수를 대체하는
            외부 도킹/물리 점수 콜백.

    파이프라인 단계:
        10단계(랭킹).
    """

    def __init__(
        self,
        config: Optional[RankerConfig] = None,
        tanimoto_plugin: Optional[ScoreFn] = None,
        physics_plugin: Optional[ScoreFn] = None,
    ) -> None:
        """랭커의 설정과 선택적 점수 플러그인을 초기화한다.

        생성 속성:
            self.config (RankerConfig): 전달된 config가 없으면 기본값으로 생성.
            self.tanimoto_plugin (Optional[ScoreFn]): 외부 Tanimoto 점수 콜백.
            self.physics_plugin (Optional[ScoreFn]): 외부 물리/도킹 점수 콜백.

        파라미터:
            config (Optional[RankerConfig]): 점수 정책 설정. None이면 기본 생성.
            tanimoto_plugin (Optional[ScoreFn]): 후보와 배치를 받아 후보당 점수를
                반환하는 형상 유사도 콜백.
            physics_plugin (Optional[ScoreFn]): 후보와 배치를 받아 후보당 점수를
                반환하는 물리/도킹 콜백.

        파이프라인 단계:
            10단계(랭킹) - 객체 초기화.
        """
        self.config = config or RankerConfig()
        self.tanimoto_plugin = tanimoto_plugin
        self.physics_plugin = physics_plugin

    def score_tanimoto(self, candidates: Tensor, batch: AlignmentBatch) -> Tensor:
        """TanimotoCombo 방식의 형상+색상 유사도 점수를 후보별로 계산한다.

        기본 폴백: 가우시안 오버랩 기반 형상 Tanimoto + 원자 타입 일치 기반
        "색상" 오버랩 (파마코포어/색상 기여도의 대리값).

        생성이유:
            reference 분자와의 3D 형상/화학적 유사도는 정렬 품질의 핵심 지표다.
            외부 ROCS 등을 쓸 수 없는 환경에서도 동작하도록 경량 프록시를 둔다.

        역할:
            tanimoto_plugin이 주입되어 있으면 그것에 위임하고, 없으면
            proxy_tanimoto_combo로 형상+색상 점수를 계산한다.

        메커니즘:
            tanimoto_plugin이 None이 아니면 (candidates, batch)로 호출해 반환한다.
            None이면 batch의 reference 좌표/원자타입과 config.shape_sigma를 넘겨
            proxy_tanimoto_combo를 호출한다.

        파라미터:
            candidates (Tensor): 후보 포즈 ``[num_samples, Nq, 3]``.
            batch (AlignmentBatch): reference 좌표/원자타입을 제공하는 배치.

        반환:
            Tensor: 후보당 Tanimoto 유사도 점수 ``[num_samples]``.

        파이프라인 단계:
            10단계(랭킹) - 형상/색상 점수 계산.
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
        """도킹/물리 기반 점수를 후보별로 계산한다.

        생성이유:
            형상 유사도만으로는 포켓 내 결합 타당성을 평가할 수 없다. 포켓과의
            근접성과 충돌 패널티를 반영하는 물리 항이 필요하다.

        역할:
            physics_plugin이 주입되어 있으면 그것에 위임하고, 없으면
            proxy_physics_score로 포켓 근접성 - 충돌 패널티 점수를 계산한다.

        메커니즘:
            physics_plugin이 None이 아니면 (candidates, batch)로 호출해 반환한다.
            None이면 batch.pocket_pos(없으면 reference 폴백)와 config의
            clash_cutoff/clash_penalty_weight를 넘겨 proxy_physics_score를 호출한다.

        파라미터:
            candidates (Tensor): 후보 포즈 ``[num_samples, Nq, 3]``.
            batch (AlignmentBatch): pocket_pos/reference_pos를 제공하는 배치.

        반환:
            Tensor: 후보당 물리/도킹 점수 ``[num_samples]``.

        파이프라인 단계:
            10단계(랭킹) - 물리/도킹 점수 계산.
        """
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
        """Tanimoto 항과 물리 항을 가중 합산한 최종 점수와 분해 결과를 반환한다.

        생성이유:
            rank_candidates가 사용할 최종 점수를 산출하면서, 디버깅을 위해
            구성 요소별 기여를 함께 보존할 단일 진입점이 필요하다.

        역할:
            score_tanimoto와 score_physics를 호출해 두 항을 구하고,
            config의 가중치로 합산하여 RankBreakdown으로 묶어 반환한다.

        메커니즘:
            tanimoto = score_tanimoto(...), physics = score_physics(...)를 구하고
            total = tanimoto_weight*tanimoto + physics_weight*physics를 계산한다.
            세 텐서를 RankBreakdown으로 감싸 반환한다.

        파라미터:
            candidates (Tensor): 후보 포즈 ``[num_samples, Nq, 3]``.
            batch (AlignmentBatch): 점수 계산에 필요한 참조/포켓 정보를 가진 배치.

        반환:
            RankBreakdown: total/tanimoto/physics 점수(각 ``[num_samples]``).

        파이프라인 단계:
            10단계(랭킹) - 최종 점수 합산.
        """
        tanimoto = self.score_tanimoto(candidates, batch)
        physics = self.score_physics(candidates, batch)
        total = self.config.tanimoto_weight * tanimoto + self.config.physics_weight * physics
        return RankBreakdown(total=total, tanimoto=tanimoto, physics=physics)


class LegacyReferenceMseRanker:
    """참조 좌표와의 MSE를 기준으로 점수화하는 하위 호환용 레거시 랭커.

    생성이유:
        초기 스캐폴드 시절의 단순 평가 방식(참조 좌표와의 거리)을 유지해야 하는
        기존 코드/실험과의 하위 호환을 위해 보존한다.

    역할:
        후보와 reference_pos 간 평균제곱오차의 음수를 점수로 사용하여
        PluginRanker와 동일한 RankBreakdown 인터페이스로 반환한다.

    정의되는 주요 변수와 의미:
        별도 인스턴스 상태 없음(설정/플러그인 미사용). score 메서드만 제공한다.

    파이프라인 단계:
        10단계(랭킹) - 레거시 점수화.
    """

    def score(self, candidates: Tensor, batch: AlignmentBatch) -> RankBreakdown:
        """후보와 참조 좌표 간 MSE의 음수를 점수로 산출한다.

        생성이유:
            참조 포즈에 가까울수록 좋은 정렬이라는 단순 가정 하의 점수가
            필요한 레거시 경로를 지원한다.

        역할:
            reference_pos가 있으면 후보별 MSE를 계산해 음수를 점수로 쓰고,
            없으면 0점을 반환한다. tanimoto/physics 항은 0으로 채운다.

        메커니즘:
            reference_pos가 None이면 후보 수 길이의 0 텐서를 점수로 둔다.
            아니면 (candidates - reference_pos.unsqueeze(0))의 제곱을 (1,2)축
            평균해 후보별 MSE를 얻고 score = -mse로 둔다. 구성 요소(tanimoto,
            physics)는 score와 같은 형상의 0 텐서로 채워 RankBreakdown 반환.

        파라미터:
            candidates (Tensor): 후보 포즈 ``[num_samples, Nq, 3]``.
            batch (AlignmentBatch): reference_pos를 제공하는 배치.

        반환:
            RankBreakdown: total=-MSE(또는 0), tanimoto=0, physics=0.

        파이프라인 단계:
            10단계(랭킹) - 레거시 MSE 점수화.
        """
        if batch.reference_pos is None:
            score = torch.zeros(candidates.size(0), device=candidates.device)
        else:
            diff = candidates - batch.reference_pos.unsqueeze(0)
            mse = (diff * diff).mean(dim=(1, 2))
            score = -mse
        zeros = torch.zeros_like(score)
        return RankBreakdown(total=score, tanimoto=zeros, physics=zeros)


def _pairwise_min_dist(a: Tensor, b: Tensor) -> Tensor:
    """``a``의 각 점에서 집합 ``b``까지의 최소 거리를 반환한다.

    생성이유:
        물리 점수에서 리간드 원자 각각이 수용체(포켓/참조)에 얼마나 가까운지를
        측정해야 한다. 이를 위한 점-집합 최소 거리 계산을 재사용 가능하게 분리.

    역할:
        proxy_physics_score의 보조 함수로, 근접성/충돌 판정의 기반 거리를 제공.

    메커니즘:
        b가 None이거나 비어 있으면 모든 점에 대해 5.0의 기본 거리를 반환한다.
        그렇지 않으면 torch.cdist(a, b)를 계산해 마지막 축 최소값을 취한다.

    파라미터:
        a (Tensor): 기준 점 집합 ``[Na, 3]`` (예: 리간드 후보 좌표).
        b (Tensor): 대상 점 집합 ``[Nb, 3]`` (예: 포켓 좌표). None/빈 텐서 허용.

    반환:
        Tensor: a의 각 점에서 b까지의 최소 거리 ``[Na]``.

    파이프라인 단계:
        10단계(랭킹) - 물리 점수 보조.
    """
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
    """간단한 도킹/물리 대리 점수를 후보별로 계산한다.

    포켓 포인트(또는 참조 폴백)와의 근접성을 장려하면서
    심각한 충돌에 패널티를 부과한다.

    생성이유:
        실제 도킹 엔진 없이도 포켓 적합성을 대략적으로 평가할 경량 점수가
        필요하다. 근접 보상과 충돌 패널티의 조합으로 이를 근사한다.

    역할:
        score_physics의 기본 폴백 구현. 후보 각각에 대해 수용체와의 최소 거리
        통계로 친화도와 충돌을 계산해 점수를 산출한다.

    메커니즘:
        수용체 receptor = pocket_pos(없으면 reference_pos)를 정한다. 각 후보에
        대해 _pairwise_min_dist로 후보 원자별 최소 거리를 구하고,
        affinity = -평균최소거리(가까울수록 높음), clash =
        relu(clash_cutoff - 최소거리)의 평균(임계보다 가까운 접촉의 침범량)을
        계산한다. 점수 = affinity - clash_penalty_weight*clash 를 누적해 stack.

    파라미터:
        candidates (Tensor): 후보 포즈 ``[num_samples, Nq, 3]``.
        pocket_pos (Optional[Tensor]): 포켓 좌표 ``[Np, 3]``. None이면 참조로 폴백.
        reference_pos (Optional[Tensor]): 포켓이 없을 때 사용할 참조 좌표.
        clash_cutoff (float): 충돌로 간주하는 최소 거리 임계값.
        clash_penalty_weight (float): 충돌 패널티 가중치.

    반환:
        Tensor: 후보당 물리 점수 ``[num_samples]``.

    파이프라인 단계:
        10단계(랭킹) - 물리 점수 프록시.
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
    """두 점 집합 간의 대칭적(Tanimoto 형태) 가우시안 형상 오버랩을 계산한다.

    생성이유:
        분자 형상 유사도를 ROCS 없이 근사하기 위해, 원자 좌표를 가우시안으로
        보고 겹침 적분을 Tanimoto 형태로 정규화한 척도를 제공한다.

    역할:
        proxy_tanimoto_combo의 형상 항을 계산하는 보조 함수.

    메커니즘:
        ref가 None/빈 텐서면 0을 반환한다. gamma=1/(2σ²)로 두고 query-ref,
        query-query, ref-ref 쌍거리 제곱에 exp(-gamma·d²)를 적용해 각각 합산
        (ov_qr, ov_qq, ov_rr). 결과는 2·ov_qr / (ov_qq + ov_rr + 1e-8) 형태의
        대칭 Tanimoto 오버랩이다.

    파라미터:
        query (Tensor): 첫 점 집합(후보 좌표) ``[Nq, 3]``.
        ref (Tensor): 둘째 점 집합(참조 좌표) ``[Nr, 3]``. None/빈 텐서 허용.
        sigma (float): 가우시안 너비 σ.

    반환:
        Tensor: 0~1 부근의 스칼라 형상 오버랩 점수.

    파이프라인 단계:
        10단계(랭킹) - 형상 점수 보조.
    """
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
    """쿼리-참조 원자 타입이 일치하는 쌍을 표시하는 불리언 마스크를 만든다.

    생성이유:
        Tanimoto의 "색상"(화학적 유사도) 항은 같은 원자 타입끼리의 오버랩만
        세야 한다. 이를 위해 타입 일치 여부 행렬이 필요하다.

    역할:
        proxy_tanimoto_combo의 색상 항 계산에 쓰일 ``[Nq, Nr]`` 일치 마스크 생성.

    메커니즘:
        둘 중 하나라도 None이면 None을 반환한다. 아니면 브로드캐스트 비교
        (query[:,None] == reference[None,:])로 일치 행렬을 만들어 device로 옮긴다.

    파라미터:
        query_atom_type (Optional[Tensor]): 쿼리 원자 타입 ``[Nq]``.
        reference_atom_type (Optional[Tensor]): 참조 원자 타입 ``[Nr]``.
        device (torch.device): 결과 텐서를 둘 디바이스.

    반환:
        Optional[Tensor]: 타입 일치 불리언 마스크 ``[Nq, Nr]``. 입력이 없으면 None.

    파이프라인 단계:
        10단계(랭킹) - 색상 점수 보조.
    """
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
    """형상 + 원자 타입 매칭 색상 항으로부터 근사 TanimotoCombo 점수를 계산한다.

    생성이유:
        OpenEye ROCS의 TanimotoCombo(형상+색상)를 외부 의존 없이 근사해 후보
        랭킹을 가능하게 하기 위한 경량 프록시가 필요하다.

    역할:
        score_tanimoto의 기본 폴백 구현. 각 후보의 형상 오버랩과 색상 오버랩을
        합해 후보당 점수를 만든다.

    메커니즘:
        reference_pos가 None/빈 텐서면 0 점수를 반환한다. _atom_match_mask로
        타입 일치 마스크를 구한 뒤, 각 후보 cand에 대해:
        - shape = _gaussian_overlap(cand, reference_pos, sigma)
        - mask가 None이면 color=0, 아니면 gamma=1/(2σ²)로 exp(-gamma·d²)에
          마스크를 곱한 합을 mask.sum()(최소 1.0 보장)으로 정규화해 [0,1] 색상.
        shape+color를 후보 점수로 stack 하여 반환한다.

    파라미터:
        candidates (Tensor): 후보 포즈 ``[num_samples, Nq, 3]``.
        reference_pos (Optional[Tensor]): 참조 좌표 ``[Nr, 3]``. None/빈 텐서면 0점.
        query_atom_type (Optional[Tensor]): 쿼리 원자 타입 ``[Nq]``.
        reference_atom_type (Optional[Tensor]): 참조 원자 타입 ``[Nr]``.
        sigma (float): 가우시안 오버랩 너비 σ.

    반환:
        Tensor: 후보당 근사 TanimotoCombo 점수 ``[num_samples]``.

    파이프라인 단계:
        10단계(랭킹) - 형상+색상 점수 프록시.
    """
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
