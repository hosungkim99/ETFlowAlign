# etflowalign/scripts/prepare_diffalign_example_batch.py
"""번들된 DiffAlign 예제에서 ETFlowAlign .pt 추론 배치를 생성한다.

Example:
    python -m etflowalign.scripts.prepare_diffalign_example_batch \
      --repo-root /home/deepfold/users/hosung/work/ETFlowAlign \
      --out /home/deepfold/users/hosung/work/ETFlowAlign/etflowalign/smoke_tests/diffalign_example_infer.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from etflowalign.diffalign_adapter import (
    build_diffalign_example_inference_payload,
    validate_diffalign_example_inference_payload,
)


def build_argparser() -> argparse.ArgumentParser:
    """추론 배치 준비 CLI의 argparse 파서를 구성한다.

    한 줄 요약:
        DiffAlign 예제를 ETFlowAlign 추론 .pt로 변환하는 데 필요한 인자를 정의한 파서를 만든다.
    생성이유:
        저장소 루트, 입력 파일(query/reference/pocket) 경로, 출력 경로, 시드, 쿼리 컨포머 유지
        옵션을 CLI로 지정하기 위한 인자 사양을 한 곳에 모으기 위함.
    역할:
        --repo-root, --query-sdf, --reference-sdf, --pocket-pdb, --out, --seed,
        --keep-query-conformer 인자를 등록한다.
    메커니즘:
        argparse.ArgumentParser를 생성하고 add_argument로 각 옵션을 정의해 반환한다.
    파라미터:
        없음.
    반환:
        argparse.ArgumentParser - 위 인자들이 등록된 파서 객체.
    파이프라인 단계:
        1단계(데이터 준비) - 추론 배치 CLI 인자 정의.
    """
    parser = argparse.ArgumentParser(
        description="Convert DiffAlign example query/reference/pocket into ETFlowAlign .pt input."
    )

    parser.add_argument(
        "--repo-root",
        type=str,
        required=True,
        help="ETFlowAlign repository root.",
    )
    parser.add_argument(
        "--query-sdf",
        type=str,
        default="",
        help="Path to query.sdf. Defaults to external/diffalign/diffalign/example/query.sdf.",
    )
    parser.add_argument(
        "--reference-sdf",
        type=str,
        default="",
        help="Path to reference.sdf. Defaults to external/diffalign/diffalign/example/reference.sdf.",
    )
    parser.add_argument(
        "--pocket-pdb",
        type=str,
        default="",
        help="Path to pocket.pdb. Defaults to external/diffalign/diffalign/example/pocket.pdb.",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output .pt path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for query_pos initialization.",
    )
    parser.add_argument(
        "--keep-query-conformer",
        action="store_true",
        help="Use centered query SDF conformer as source instead of a rigid-randomized source.",
    )

    return parser


def _default_example_paths(repo_root: Path) -> tuple[Path, Path, Path]:
    """저장소 루트로부터 번들된 DiffAlign 예제의 기본 입력 경로를 만든다.

    한 줄 요약:
        repo_root 기준으로 example 디렉터리의 query/reference/pocket 기본 경로를 반환한다.
    생성이유:
        사용자가 입력 경로를 명시하지 않았을 때 사용할 번들 예제의 표준 위치를
        한 곳에서 계산하기 위함.
    역할:
        example 디렉터리 경로를 구성하고 query.sdf/reference.sdf/pocket.pdb 경로를 만든다.
    메커니즘:
        repo_root / "external" / "diffalign" / "diffalign" / "example" 아래 세 파일 경로를 조합한다.
    파라미터:
        repo_root (Path): ETFlowAlign 저장소 루트 경로.
    반환:
        tuple[Path, Path, Path]: (query.sdf, reference.sdf, pocket.pdb) 기본 경로.
    파이프라인 단계:
        1단계(데이터 준비) - 기본 예제 입력 경로 산출.
    """
    example_dir = repo_root / "external" / "diffalign" / "diffalign" / "example"
    return (
        example_dir / "query.sdf",
        example_dir / "reference.sdf",
        example_dir / "pocket.pdb",
    )


def _resolve_input_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """CLI 인자와 기본값을 종합해 실제 입력 파일 경로를 확정한다.

    한 줄 요약:
        사용자가 지정한 경로가 있으면 그것을, 없으면 기본 예제 경로를 절대 경로로 반환한다.
    생성이유:
        각 입력(query/reference/pocket)에 대해 "인자 우선, 없으면 기본값" 규칙을
        일관되게 적용하기 위함.
    역할:
        repo_root를 절대 경로화하고 기본 경로를 구한 뒤, 인자가 비어 있지 않으면 인자 경로를
        절대화하여 사용하고 비어 있으면 기본 경로를 사용한다.
    메커니즘:
        Path(...).resolve()로 repo_root와 각 인자 경로를 절대화하고, _default_example_paths로 기본값을 얻어
        삼항 선택으로 최종 경로를 결정한다.
    파라미터:
        args (argparse.Namespace): repo_root/query_sdf/reference_sdf/pocket_pdb 인자를 담은 파싱 결과.
    반환:
        tuple[Path, Path, Path]: (query, reference, pocket)의 확정된 절대 경로.
    파이프라인 단계:
        1단계(데이터 준비) - 입력 경로 확정.
    """
    repo_root = Path(args.repo_root).resolve()
    default_query_sdf, default_reference_sdf, default_pocket_pdb = _default_example_paths(repo_root)

    query_sdf = Path(args.query_sdf).resolve() if args.query_sdf else default_query_sdf
    reference_sdf = Path(args.reference_sdf).resolve() if args.reference_sdf else default_reference_sdf
    pocket_pdb = Path(args.pocket_pdb).resolve() if args.pocket_pdb else default_pocket_pdb

    return query_sdf, reference_sdf, pocket_pdb


def _print_tensor_shape(prefix: str, payload: dict[str, Any], key: str) -> None:
    """페이로드의 한 키 값을 형태/타입과 함께 한 줄로 출력한다.

    한 줄 요약:
        payload[key]가 텐서면 형태·dtype을, 아니면 값 자체를, 없으면 MISSING을 출력한다.
    생성이유:
        준비된 페이로드의 각 필드가 올바른 형태/타입을 갖는지 사람이 눈으로 확인할 수 있도록
        일관된 출력 포맷을 제공하기 위함.
    역할:
        키 존재 여부와 텐서 여부에 따라 분기하여 해당 정보를 print한다.
    메커니즘:
        payload.get(key)로 값을 꺼내 None이면 MISSING, torch.is_tensor면 shape/dtype, 그 외엔 값을 출력한다.
    파라미터:
        prefix (str): 각 출력 줄 앞에 붙일 접두사.
        payload (dict[str, Any]): 점검할 페이로드 딕셔너리.
        key (str): 출력할 페이로드 키.
    반환:
        None: 표준출력 부수효과만 갖는다.
    파이프라인 단계:
        1단계(데이터 준비) - 페이로드 점검 출력 보조.
    """
    value = payload.get(key)
    if value is None:
        print(f"{prefix} {key}=MISSING")
        return

    if torch.is_tensor(value):
        print(f"{prefix} {key}={tuple(value.shape)} {value.dtype}")
    else:
        print(f"{prefix} {key}={value}")


def print_payload_summary(prefix: str, payload: dict[str, Any]) -> None:
    """추론 페이로드의 핵심 필드 요약을 출력한다.

    한 줄 요약:
        미리 정한 핵심 키들에 대해 형태/값 요약을 차례로 출력한다.
    생성이유:
        준비된 추론 배치가 기대한 텐서들(query_pos, node_attr, reference/pocket 등)을
        제대로 담았는지 한눈에 검증하기 위함.
    역할:
        summary_keys 목록을 순회하며 각 키에 대해 _print_tensor_shape를 호출한다.
    메커니즘:
        고정된 summary_keys 리스트를 정의하고 각 키마다 _print_tensor_shape(prefix, payload, key)를 호출한다.
    파라미터:
        prefix (str): 각 출력 줄 앞에 붙일 접두사.
        payload (dict[str, Any]): 점검할 추론 페이로드.
    반환:
        None: 표준출력 부수효과만 갖는다.
    파이프라인 단계:
        1단계(데이터 준비) - 추론 페이로드 요약 출력.
    """
    summary_keys = [
        "query_pos",
        "query_node_attr",
        "reference_pos",
        "reference_node_attr",
        "pocket_pos",
        "pocket_batch",
        "reference_center_subtracted",
        "randomize_query_pos",
        "query_source",
        "seed",
    ]

    for key in summary_keys:
        _print_tensor_shape(prefix, payload, key)


def print_payload_keys(prefix: str, payload: dict[str, Any]) -> None:
    """페이로드의 모든 키를 정렬해 형태/값과 함께 출력한다.

    한 줄 요약:
        페이로드의 전체 키를 알파벳순으로 돌며 각 값의 형태/타입 또는 값을 출력한다.
    생성이유:
        요약 키 외에도 페이로드에 실제로 어떤 키들이 들어 있는지 전수 확인하여
        스키마 누락/오타를 잡기 위함.
    역할:
        키를 정렬해 순회하며 텐서면 형태·dtype을, 아니면 값을 들여쓰기와 함께 출력한다.
    메커니즘:
        sorted(payload.keys())로 순회하고 torch.is_tensor 여부에 따라 출력 형식을 분기한다.
    파라미터:
        prefix (str): 헤더 줄 앞에 붙일 접두사.
        payload (dict[str, Any]): 점검할 페이로드.
    반환:
        None: 표준출력 부수효과만 갖는다.
    파이프라인 단계:
        1단계(데이터 준비) - 페이로드 전체 키 점검 출력.
    """
    print(f"{prefix} keys:")
    for key in sorted(payload.keys()):
        value = payload[key]
        if torch.is_tensor(value):
            print(f"  - {key}: tensor shape={tuple(value.shape)} dtype={value.dtype}")
        else:
            print(f"  - {key}: {value}")


def main() -> None:
    """CLI 진입점: DiffAlign 예제를 ETFlowAlign 추론 .pt 배치로 변환·저장한다.

    한 줄 요약:
        예제 입력으로 추론 페이로드를 만들고 검증한 뒤 .pt로 저장하고 요약을 출력한다.
    생성이유:
        모델 추론에 바로 투입할 수 있는 단일 예제 입력 배치를 손쉽게 생성하는 진입점이 필요하다.
    역할:
        인자를 파싱해 입력 경로를 확정하고, build_diffalign_example_inference_payload로 페이로드를 생성,
        validate로 검증 후 출력 경로에 저장하고 요약/키를 출력한다.
    메커니즘:
        _resolve_input_paths로 경로 확정 → build_diffalign_example_inference_payload(randomize_query_pos는
        --keep-query-conformer의 부정) → validate_diffalign_example_inference_payload →
        부모 디렉터리 생성 후 torch.save → print_payload_summary/print_payload_keys 출력.
    파라미터:
        없음(인자는 build_argparser로 명령행에서 파싱).
    반환:
        None: 결과는 .pt 파일 저장과 표준출력 요약이다.
    파이프라인 단계:
        1단계(데이터 준비) - 추론 배치 생성 실행 진입점.
    """
    args = build_argparser().parse_args()

    repo_root = Path(args.repo_root).resolve()
    query_sdf, reference_sdf, pocket_pdb = _resolve_input_paths(args)

    payload = build_diffalign_example_inference_payload(
        repo_root=repo_root,
        query_sdf=query_sdf,
        reference_sdf=reference_sdf,
        pocket_pdb=pocket_pdb,
        randomize_query_pos=not args.keep_query_conformer,
        seed=args.seed,
    )

    validate_diffalign_example_inference_payload(payload)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)

    print(f"[prepare] saved: {out_path}")
    print_payload_summary("[prepare]", payload)
    print_payload_keys("[prepare]", payload)


if __name__ == "__main__":
    main()