"""ETFlowAlign 공용 소형 헬퍼 모음.

이 모듈은 경량으로 유지한다. 핵심 알고리즘 로직은 여기에 두지 않는다.
"""

from __future__ import annotations

import torch
from torch import Tensor


def safe_norm(x: Tensor, dim: int = -1, keepdim: bool = False, eps: float = 1e-8) -> Tensor:
    """수치적으로 안정적인 벡터 노름.

    한 줄 요약:
        제곱합에 하한(eps)을 적용한 뒤 제곱근을 취해 0 근처에서도 안전한 L2
        노름을 계산한다.

    생성 이유:
        ``torch.norm``은 노름이 0에 가까울 때 역전파에서 NaN/Inf 기울기를 낼 수
        있다. 제곱합을 eps로 클램프해 미분 안정성을 확보하기 위함이다.

    역할:
        파이프라인 11단계(보조)의 공용 유틸로, 모델/손실 계산 전반에서 안정적인
        거리·노름 계산에 사용된다.

    메커니즘:
        ``(x*x).sum(dim, keepdim)``으로 제곱합을 구하고 ``clamp_min(eps)``로
        하한을 강제한 뒤 ``torch.sqrt``를 적용한다.

    파라미터:
        x (Tensor): 노름을 계산할 입력 텐서.
        dim (int): 축약할 차원(기본 -1, 마지막 차원).
        keepdim (bool): 축약 차원을 유지할지 여부(기본 False).
        eps (float): 제곱합에 적용할 하한값(기본 1e-8).

    반환:
        Tensor: 지정 차원에 대해 계산된 안정적 L2 노름.

    파이프라인 단계:
        11단계(보조)의 공용 유틸.
    """
    return torch.sqrt((x * x).sum(dim=dim, keepdim=keepdim).clamp_min(eps))


def segment_mean(x: Tensor, batch: Tensor, num_graphs: int | None = None) -> Tensor:
    """그래프별 평균 텐서를 계산한다.

    한 줄 요약:
        노드별 ``batch`` 인덱스를 기준으로 같은 그래프에 속한 행들의 평균을
        그래프 단위로 산출한다.

    생성 이유:
        멀티-그래프 배치에서 그래프별 집계(예: 무게중심, 풀링)가 필요하며,
        루프 없이 ``index_add_`` 기반으로 효율적으로 segment 평균을 구하기 위함이다.

    역할:
        파이프라인 11단계(보조)의 공용 유틸로, 그래프 단위 평균 집계를 제공한다.

    메커니즘:
        ``num_graphs``가 None이면 ``batch.max()+1``로 그래프 수를 추정한다(빈
        배치는 0). 합산용 출력 텐서와 카운트 텐서를 0으로 만든 뒤 ``index_add_``로
        그래프별 합과 개수를 누적하고, 합을 ``clamp_min(1.0)``한 개수로 나눠
        그래프별 평균을 반환한다.

    파라미터:
        x (Tensor): 집계할 노드 피처 텐서 ``[N, D]``.
        batch (Tensor): 각 행이 속한 그래프 ID ``[N]``.
        num_graphs (int | None): 그래프 수. None이면 ``batch``로부터 추정.

    반환:
        Tensor: 그래프별 평균 텐서 ``[num_graphs, D]``.

    파이프라인 단계:
        11단계(보조)의 공용 유틸.
    """
    if num_graphs is None:
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
    out = torch.zeros(num_graphs, x.size(-1), device=x.device, dtype=x.dtype)
    cnt = torch.zeros(num_graphs, 1, device=x.device, dtype=x.dtype)
    out.index_add_(0, batch, x)
    cnt.index_add_(0, batch, torch.ones_like(batch, dtype=x.dtype).unsqueeze(-1))
    return out / cnt.clamp_min(1.0)


def center_by_batch(x: Tensor, batch: Tensor) -> Tensor:
    """그래프별 평균 좌표를 뺀다.

    한 줄 요약:
        각 노드에서 자신이 속한 그래프의 평균 좌표를 빼서 그래프 단위로
        중심화(centering)한다.

    생성 이유:
        SE(3) flow-matching에서 좌표는 평행이동에 불변해야 하므로, 그래프별로
        무게중심을 원점으로 옮기는 전처리가 필요하다.

    역할:
        파이프라인 11단계(보조)의 공용 유틸로, 좌표의 그래프별 평행이동 정규화를
        제공한다.

    메커니즘:
        ``batch.max()+1``로 그래프 수를 구하고(빈 배치는 입력을 그대로 반환),
        ``index_add_``로 그래프별 좌표 합과 노드 수를 누적해 평균을 계산한 뒤,
        각 노드에서 ``means[batch]``를 빼서 중심화한다.

    Args:
        x: 좌표/피처 텐서 ``[N, D]``.
        batch: 행별 그래프 ID ``[N]``.

    Returns:
        그래프별 평균이 제거된 중심화된 텐서.

    파이프라인 단계:
        11단계(보조)의 공용 유틸.
    """
    num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
    if num_graphs == 0:
        return x
    means = torch.zeros(num_graphs, x.size(-1), device=x.device, dtype=x.dtype)
    counts = torch.zeros(num_graphs, 1, device=x.device, dtype=x.dtype)
    means.index_add_(0, batch, x)
    counts.index_add_(0, batch, torch.ones_like(batch, dtype=x.dtype).unsqueeze(-1))
    means = means / counts.clamp_min(1.0)
    return x - means[batch]
