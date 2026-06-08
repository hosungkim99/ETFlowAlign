"""ETFlowAlign 배치 유효성 검사 유틸리티."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .model import AlignmentBatch


def _require_tensor(name: str, value: Optional[Tensor]) -> Tensor:
    """값이 존재하는 torch 텐서인지 보장하고 그대로 반환한다.

    한 줄 요약:
        주어진 값이 None이 아니고 torch 텐서임을 확인한 뒤 반환한다.

    생성 이유:
        배치의 필수 필드가 누락(None)되었거나 텐서가 아닌 경우를 검사 초기에
        명확한 메시지로 거부하기 위한 가드가 필요하다.

    역할:
        파이프라인 11단계(보조)의 배치 검증에서 필수 텐서 필드의 존재/타입을
        보장하는 공통 헬퍼.

    메커니즘:
        값이 None이면 누락 오류, ``torch.is_tensor``가 False이면 타입 오류로
        ``ValueError``를 던지고, 통과하면 입력 값을 그대로 반환한다.

    파라미터:
        name (str): 오류 메시지에 사용할 필드 이름.
        value (Optional[Tensor]): 검사할 값.

    반환:
        Tensor: 검증을 통과한 입력 텐서.

    파이프라인 단계:
        11단계(보조)의 배치 검증.
    """
    if value is None:
        raise ValueError(f"Missing required field '{name}'.")
    if not torch.is_tensor(value):
        raise ValueError(f"Field '{name}' must be a torch.Tensor.")
    return value


def _check_shape(name: str, x: Tensor, dim1: Optional[int] = None, rank: Optional[int] = None) -> None:
    """텐서의 차원 수(rank)와 두 번째 차원 크기를 검증한다.

    한 줄 요약:
        텐서가 기대하는 rank 및 ``[N, dim1]`` 형상 제약을 만족하는지 확인한다.

    생성 이유:
        좌표(``[N,3]``)나 인덱스(``[N]``) 등 모델이 요구하는 텐서 형상을 통일된
        방식으로 검사해, 형상 불일치를 모델 실행 전에 잡아내기 위함이다.

    역할:
        파이프라인 11단계(보조)의 배치 검증에서 개별 텐서의 형상 제약을 확인한다.

    메커니즘:
        ``rank``가 주어지면 ``x.dim() != rank``일 때, ``dim1``이 주어지면 텐서
        rank가 2 미만이거나 ``x.size(1) != dim1``일 때 각각 ``ValueError``를
        던진다. 두 조건 모두 None이면 아무 검사도 하지 않는다.

    파라미터:
        name (str): 오류 메시지에 사용할 필드 이름.
        x (Tensor): 형상을 검사할 텐서.
        dim1 (Optional[int]): 기대하는 두 번째 차원 크기(예: 좌표면 3). None이면 미검사.
        rank (Optional[int]): 기대하는 차원 수. None이면 미검사.

    반환:
        None: 검증을 통과하면 아무것도 반환하지 않고, 실패 시 예외를 던진다.

    파이프라인 단계:
        11단계(보조)의 배치 검증.
    """
    if rank is not None and x.dim() != rank:
        raise ValueError(f"Field '{name}' must have rank {rank}, got shape {tuple(x.shape)}.")
    if dim1 is not None and (x.dim() < 2 or x.size(1) != dim1):
        raise ValueError(f"Field '{name}' must have shape [N,{dim1}], got shape {tuple(x.shape)}.")


def validate_alignment_batch(
    batch: AlignmentBatch,
    target_query_pos: Tensor | None = None,
    require_reference: bool = False,
    require_pocket: bool = False,
    require_target: bool = False,
) -> None:
    """모델/샘플러 실행 전에 배치 구조와 형상을 검증한다.

    한 줄 요약:
        ``AlignmentBatch``의 필수/선택 텐서 필드들에 대해 존재 여부, dtype,
        형상, 차원 간 정합성을 일괄 검사한다.

    생성 이유:
        잘못된 형상이나 인덱스 불일치가 모델/샘플러 내부에서 모호한 오류로
        터지는 것을 막고, 데이터 로딩 시점에 명확한 메시지로 조기 실패시키기
        위함이다.

    역할:
        파이프라인 11단계(보조)에서 배치 무결성(텐서 shape/인덱스 정합성)을
        검증하며, 데이터 로딩 및 추론 진입점에서 호출된다.

    메커니즘:
        query 필드(pos/atom_type/batch)는 항상 필수로 보고 텐서 존재, 형상
        (``query_pos``는 ``[N,3]``, 나머지는 rank 1), 공통 첫 차원 Nq, dtype
        (좌표는 부동소수점, 타입/배치는 int32/int64)을 검사한다. reference와
        pocket 필드는 ``require_*`` 플래그가 켜져 있거나 해당 필드가 하나라도
        존재하면 검사하며, pos와 batch의 형상 및 첫 차원 일치를 확인한다.
        정답(``target_query_pos``)은 ``require_target``이면 존재를 강제하고,
        존재하면 ``[Nq,3]`` 형상과 query와의 첫 차원 일치를 확인한다.

    파라미터:
        batch (AlignmentBatch): 검증할 배치 객체.
        target_query_pos (Tensor | None): 정답 좌표 텐서(없으면 None).
        require_reference (bool): reference 필드 존재를 강제할지 여부(기본 False).
        require_pocket (bool): pocket 필드 존재를 강제할지 여부(기본 False).
        require_target (bool): 정답 좌표 존재를 강제할지 여부(기본 False).

    반환:
        None: 모든 검사를 통과하면 아무것도 반환하지 않고, 위반 시 ``ValueError``.

    파이프라인 단계:
        11단계(보조)의 배치 무결성 검사.
    """
    qpos = _require_tensor("query_pos", batch.query_pos)
    qtype = _require_tensor("query_atom_type", batch.query_atom_type)
    qbatch = _require_tensor("query_batch", batch.query_batch)

    _check_shape("query_pos", qpos, dim1=3, rank=2)
    _check_shape("query_atom_type", qtype, rank=1)
    _check_shape("query_batch", qbatch, rank=1)
    if qpos.size(0) != qtype.size(0) or qpos.size(0) != qbatch.size(0):
        raise ValueError("query_pos, query_atom_type, query_batch must share first dimension Nq.")
    if not torch.is_floating_point(qpos):
        raise ValueError("query_pos must be floating point.")
    if qtype.dtype not in (torch.int32, torch.int64):
        raise ValueError("query_atom_type must be int32 or int64.")
    if qbatch.dtype not in (torch.int32, torch.int64):
        raise ValueError("query_batch must be int32 or int64.")

    if require_reference or batch.reference_pos is not None or batch.reference_batch is not None:
        rpos = _require_tensor("reference_pos", batch.reference_pos)
        rbatch = _require_tensor("reference_batch", batch.reference_batch)
        _check_shape("reference_pos", rpos, dim1=3, rank=2)
        _check_shape("reference_batch", rbatch, rank=1)
        if rpos.size(0) != rbatch.size(0):
            raise ValueError("reference_pos and reference_batch must share first dimension Nr.")

    pocket_pos = batch.pocket_pos
    pocket_batch = getattr(batch, "pocket_batch", None)
    if require_pocket or pocket_pos is not None or pocket_batch is not None:
        ppos = _require_tensor("pocket_pos", pocket_pos)
        pbatch = _require_tensor("pocket_batch", pocket_batch)
        _check_shape("pocket_pos", ppos, dim1=3, rank=2)
        _check_shape("pocket_batch", pbatch, rank=1)
        if ppos.size(0) != pbatch.size(0):
            raise ValueError("pocket_pos and pocket_batch must share first dimension Np.")

    if require_target:
        target_query_pos = _require_tensor("target_query_pos", target_query_pos)
    if target_query_pos is not None:
        _check_shape("target_query_pos", target_query_pos, dim1=3, rank=2)
        if target_query_pos.size(0) != qpos.size(0):
            raise ValueError("target_query_pos must have same first dimension as query_pos (Nq).")
