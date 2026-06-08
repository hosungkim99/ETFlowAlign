"""평가 스크립트와 학습 중 모니터링에서 공용으로 사용하는 포즈-RMSD 평가 모듈.

모델과 복합체별 ``.pt`` 경로 목록을 받아, 저장된 소스 각각에서 포즈를 샘플링하고
타겟과의 원자 단위 RMSD를 측정한다(동일 프레임, 대응 관계가 알려진 경우).
"""

from __future__ import annotations

import statistics

import torch

from .data import load_alignment_batch_from_pt
from .dataset import Conditioning, apply_conditioning
from .sampler import ETFlowAlignSampler, ODESamplerConfig


def _rmsd(a: torch.Tensor, b: torch.Tensor) -> float:
    """두 좌표 집합 간 원자 단위 RMSD를 계산한다.

    한 줄 요약:
        동일 프레임·동일 대응 관계를 가진 두 좌표 텐서 사이의 평균 제곱근 편차(RMSD)를 반환.

    생성 이유:
        평가 모듈 곳곳에서 소스/예측 좌표를 타겟과 비교할 때 반복되는 RMSD 계산을 단일
        헬퍼로 통일하기 위해 만들어졌다.

    역할:
        파이프라인 6단계(평가)의 기본 거리 측정 단위. 정렬 품질의 핵심 지표를 산출한다.

    메커니즘:
        원자별 좌표 차이를 제곱해 마지막 축(좌표 x/y/z)으로 합산하여 원자별 제곱거리를 얻고,
        그 평균에 제곱근을 취해 스칼라 RMSD를 만든다. (사전 정렬/대응 매칭은 가정하지 않음.)

    파라미터:
        a (torch.Tensor): 좌표 텐서, 형상 (N, 3).
        b (torch.Tensor): 비교 대상 좌표 텐서, 형상 (N, 3). a와 원자 순서가 대응되어야 한다.

    반환:
        float: 두 좌표 집합 간의 RMSD 스칼라 값.

    파이프라인 단계:
        6단계(평가)의 거리 계산 보조.
    """
    return float(((a - b) ** 2).sum(dim=-1).mean().sqrt())


def summarize(rows: list[dict]) -> dict:
    """복합체별 행을 집계하여 요약 통계를 반환한다.

    한 줄 요약:
        복합체별 평가 결과(행) 목록을 받아 소스/예측 RMSD의 평균·중앙값 및 성공률 지표로
        집계한다.

    생성 이유:
        evaluate_paths가 만든 복합체별 행들을 사람이 한눈에 볼 수 있는 집계 지표로 변환하기
        위해 만들어졌다. 학습 중 검증 모니터링과 사후 평가 양쪽에서 공통으로 쓰인다.

    역할:
        파이프라인 6단계(평가)의 집계기. 데이터셋 전체 성능을 요약 통계로 환원한다.

    메커니즘:
        - rows가 비어 있으면 {"n": 0}을 반환한다.
        - 각 행에서 pred_rmsd/source_rmsd를 추출해 평균(mean)과 중앙값(median)을 계산한다.
        - 예측 RMSD가 2A/5A 미만인 비율(frac_lt2/frac_lt5)과, 예측이 소스보다 개선된 비율
          (improved_frac, pred<source인 복합체 비율)을 산출한다.

    파라미터:
        rows (list[dict]): 복합체별 평가 행 목록. 각 행은 "pred_rmsd"와 "source_rmsd" 키를
            포함해야 한다(evaluate_paths가 생성).

    반환:
        dict: 집계 통계 사전. 비어 있으면 {"n": 0}, 아니면 n, source_rmsd_mean/median,
            pred_rmsd_mean/median, frac_lt2, frac_lt5, improved_frac 키를 포함한다.

    파이프라인 단계:
        6단계(평가)의 결과 집계.
    """
    if not rows:
        return {"n": 0}
    pred = [r["pred_rmsd"] for r in rows]
    src = [r["source_rmsd"] for r in rows]
    n = len(pred)
    return {
        "n": n,
        "source_rmsd_mean": statistics.mean(src),
        "source_rmsd_median": statistics.median(src),
        "pred_rmsd_mean": statistics.mean(pred),
        "pred_rmsd_median": statistics.median(pred),
        "frac_lt2": sum(p < 2.0 for p in pred) / n,
        "frac_lt5": sum(p < 5.0 for p in pred) / n,
        "improved_frac": sum(p < s for p, s in zip(pred, src)) / n,
    }


def format_summary(s: dict) -> str:
    """요약 통계 사전을 사람이 읽기 쉬운 한 줄 문자열로 변환한다.

    한 줄 요약:
        summarize가 반환한 집계 사전을 로그 출력용 단일 라인 문자열로 포매팅한다.

    생성 이유:
        학습 중 검증 로그([val])와 사후 평가 출력에서 동일한 형식으로 요약을 표시하기 위해
        만들어졌다. 지표 표기 방식을 한 곳에 모아 일관성을 유지한다.

    역할:
        파이프라인 6단계(평가)의 표시(presentation) 계층. 집계 결과를 출력 문자열로 변환.

    메커니즘:
        - n이 0(또는 키 부재)이면 "no complexes evaluated."를 반환한다.
        - 그 외에는 복합체 수, 소스 RMSD 평균/중앙값, 예측 RMSD 평균/중앙값, <2A·<5A 비율과
          개선 비율을 백분율로 환산해 한 줄 문자열로 구성한다.

    파라미터:
        s (dict): summarize가 반환한 요약 통계 사전.

    반환:
        str: 사람이 읽기 좋은 한 줄 요약 문자열.

    파이프라인 단계:
        6단계(평가)의 요약 출력 포매팅.
    """
    if s.get("n", 0) == 0:
        return "no complexes evaluated."
    return (
        f"n={s['n']} | source mean={s['source_rmsd_mean']:.3f} med={s['source_rmsd_median']:.3f}"
        f" | pred mean={s['pred_rmsd_mean']:.3f} med={s['pred_rmsd_median']:.3f}"
        f" | <2A={s['frac_lt2'] * 100:.1f}% <5A={s['frac_lt5'] * 100:.1f}% improved={s['improved_frac'] * 100:.1f}%"
    )


@torch.no_grad()
def evaluate_paths(
    model,
    paths: list[str],
    *,
    conditioning: Conditioning = "pocket",
    n_steps: int = 50,
    solver: str = "heun",
    device: str | torch.device = "cpu",
    limit: int = 0,
) -> tuple[dict, list[dict]]:
    """각 복합체에 대해 포즈 샘플링을 실행하고 (요약, 복합체별 행) 쌍을 반환한다.

    모델의 학습/평가 모드 전환은 호출자가 책임진다. 샘플링 자체는
    ``no_grad`` 컨텍스트에서 실행된다.

    한 줄 요약:
        주어진 복합체 .pt 경로들 각각에서 포즈를 샘플링하고 타겟과의 원자 단위 RMSD를
        측정하여, 전체 요약 통계와 복합체별 상세 행을 함께 반환한다.

    생성 이유:
        학습 중 검증 모니터링(train.run_training)과 사후 평가에서 동일한 포즈-RMSD 평가
        절차를 공유하기 위해 만들어졌다. @torch.no_grad로 감싸 그래디언트 없이 추론만
        수행한다.

    역할:
        파이프라인 6단계(평가)의 핵심 루프. 각 복합체를 로드→조건 적용→ODE 샘플링→RMSD
        측정으로 처리하고, 그 결과를 집계기(summarize)에 넘긴다.

    메커니즘:
        - limit>0이면 평가할 경로를 앞에서부터 limit개로 제한한다.
        - ODESamplerConfig(n_steps, solver)로 ETFlowAlignSampler를 생성한다.
        - 각 경로에서 load_alignment_batch_from_pt로 배치/타겟/메타를 로드한다(읽기 실패한
          복합체는 예외를 삼키고 건너뛴다).
        - apply_conditioning으로 conditioning(pocket/reference/both) 신호를 적용한다.
        - 초기 좌표 x0(=query_pos)에서 sampler.sample로 예측 포즈를 생성한다.
        - 소스 RMSD(x0 vs target), 예측 RMSD(pred vs target), 질량중심 거리(com_dist)를
          계산하고, query 결합 인덱스가 있으면 예측 포즈의 결합 길이 최소/평균/최대를
          행에 추가한다.

    파라미터:
        model: 포즈 샘플링에 사용할 ETFlowAlign 모델. 학습/평가 모드 전환은 호출자 책임.
        paths (list[str]): 복합체별 .pt 페이로드 경로 목록.
        conditioning (Conditioning): 적용할 조건 신호("pocket"/"reference"/"both").
            키워드 전용. 기본값 "pocket".
        n_steps (int): ODE 적분 스텝 수. 키워드 전용. 기본값 50.
        solver (str): ODE 솔버 종류("euler"/"heun"). 키워드 전용. 기본값 "heun".
        device (str | torch.device): 텐서 연산을 수행할 디바이스. 키워드 전용. 기본값 "cpu".
        limit (int): 평가할 최대 복합체 수(0이면 제한 없음). 키워드 전용. 기본값 0.

    반환:
        tuple[dict, list[dict]]:
            (summary, rows). summary는 summarize의 집계 통계 사전, rows는 복합체별 평가
            상세(pdb_id, n_atoms, source_rmsd, pred_rmsd, com_dist 및 선택적 bond_* 키) 목록.

    파이프라인 단계:
        6단계(평가)의 메인 평가 루프(학습 중 모니터링·사후 평가 공용).
    """
    if limit and limit > 0:
        paths = paths[:limit]
    sampler = ETFlowAlignSampler(model, ODESamplerConfig(n_steps=n_steps, solver=solver))

    rows: list[dict] = []
    for path in paths:
        try:
            batch, target, meta = load_alignment_batch_from_pt(path, require_target=True, device=device)
        except Exception:  # noqa: BLE001 - 읽을 수 없는 복합체는 평가에서 건너뜀
            continue
        batch = apply_conditioning(batch, conditioning)
        x0 = batch.query_pos
        pred = sampler.sample(batch=batch, x0=x0.clone())

        row = {
            "pdb_id": meta.get("pdb_id"),
            "n_atoms": int(x0.size(0)),
            "source_rmsd": _rmsd(x0, target),
            "pred_rmsd": _rmsd(pred, target),
            "com_dist": float((pred.mean(0) - target.mean(0)).norm()),
        }
        if batch.query_bond_index is not None and batch.query_bond_index.numel() > 0:
            bi = batch.query_bond_index
            bl = torch.linalg.norm(pred[bi[0]] - pred[bi[1]], dim=-1)
            row.update(bond_min=float(bl.min()), bond_mean=float(bl.mean()), bond_max=float(bl.max()))
        rows.append(row)

    return summarize(rows), rows
