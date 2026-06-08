""".pt 파일에서 ETFlowAlign 배치를 불러오는 I/O 헬퍼 모음."""
from __future__ import annotations

from typing import Any

import torch

from .model import AlignmentBatch
from .validation import validate_alignment_batch


def load_alignment_batch_from_pt(path: str, require_target: bool, device: str | torch.device | None = None) -> tuple[AlignmentBatch, torch.Tensor | None, dict[str, Any]]:
    """단일 복합체 ``.pt`` 파일을 읽어 ``AlignmentBatch``/정답/메타데이터로 변환한다.

    한 줄 요약:
        디스크의 dict 페이로드를 로드해 ``AlignmentBatch`` 객체, 정답 좌표,
        그리고 나머지 메타데이터로 분해해 반환한다.

    생성 이유:
        준비 스크립트가 생성한 dict 형태의 ``.pt`` 파일을 모델이 이해하는
        ``AlignmentBatch``로 옮기는 표준 I/O 진입점이 필요하다. 데이터셋과
        추론 코드가 동일한 로딩/검증 경로를 공유하도록 한다.

    역할:
        파이프라인 2단계(데이터 로딩)에서 ``.pt`` -> ``AlignmentBatch`` 변환과
        구조 검증을 담당한다.

    메커니즘:
        ``torch.load(..., weights_only=False)``로 dict 페이로드를 읽고, dict가
        아니면 ``ValueError``를 던진다. dict의 알려진 키들을 ``AlignmentBatch``의
        대응 필드로 매핑하고, ``target_query_pos``를 정답으로 분리한다.
        ``validate_alignment_batch``로 구조/형상을 검증한 뒤, 배치/정답 키로
        소비된 키를 제외한 나머지 항목을 메타데이터 dict로 모아 반환한다.

    파라미터:
        path (str): 로드할 복합체 ``.pt`` 파일 경로.
        require_target (bool): 정답 좌표(``target_query_pos``) 존재를 강제할지 여부.
        device (str | torch.device | None): ``torch.load``의 ``map_location``으로
            전달되는 디바이스(None이면 저장된 위치/CPU).

    반환:
        tuple[AlignmentBatch, torch.Tensor | None, dict[str, Any]]:
            (구성된 배치, 정답 좌표 텐서 또는 None, 배치/정답 외 메타데이터 dict).

    파이프라인 단계:
        2단계(데이터 로딩/배치 구성)의 단일 파일 로딩.
    """
    raw = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(raw, dict):
        raise ValueError("Expected .pt file to contain a dict payload.")

    batch = AlignmentBatch(
        query_pos=raw.get("query_pos"),
        query_atom_type=raw.get("query_atom_type"),
        query_batch=raw.get("query_batch"),
        reference_pos=raw.get("reference_pos"),
        reference_atom_type=raw.get("reference_atom_type"),
        reference_batch=raw.get("reference_batch"),
        pocket_pos=raw.get("pocket_pos"),
        pocket_batch=raw.get("pocket_batch"),
        pocket_atom_type=raw.get("pocket_atom_type"),
        query_node_attr=raw.get("query_node_attr"),
        reference_node_attr=raw.get("reference_node_attr"),
        query_bond_index=raw.get("query_bond_index"),
        query_bond_length=raw.get("query_bond_length"),
    )
    target = raw.get("target_query_pos")

    validate_alignment_batch(batch, target_query_pos=target, require_target=require_target)
    metadata = {k: v for k, v in raw.items() if k not in {
        "query_pos", "query_atom_type", "query_batch", "reference_pos", "reference_atom_type", "reference_batch",
        "pocket_pos", "pocket_batch", "pocket_atom_type", "query_node_attr", "reference_node_attr", "target_query_pos",
        "query_bond_index","query_bond_length"
    }}
    return batch, target, metadata