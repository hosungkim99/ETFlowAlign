# etflowalign/scripts/prepare_diffalign_example_train_batch.py
"""DiffAlign 예제에서 ETFlowAlign 유사 학습용 .pt 배치를 생성한다.

이 스크립트는 레퍼런스 리간드 중심으로 센터링한 후 원본 쿼리 SDF 컨포머를
target_query_pos로 사용한다. 스모크 테스트 및 단일 예제 오버피팅을 위한 것이며,
실제 지도 학습에는 사용하지 않는다.

Example:
    python -m etflowalign.scripts.prepare_diffalign_example_train_batch \
      --repo-root /home/deepfold/users/hosung/work/ETFlowAlign \
      --out /home/deepfold/users/hosung/work/ETFlowAlign/etflowalign/smoke_tests/diffalign_example_train.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from etflowalign.diffalign_adapter import (
    build_diffalign_example_train_payload,
    validate_diffalign_example_train_payload,
)


def build_argparser() -> argparse.ArgumentParser:
    """유사 학습 배치 준비 CLI의 argparse 파서를 구성한다.

    한 줄 요약:
        DiffAlign 예제를 ETFlowAlign 유사 학습 .pt로 변환하는 데 필요한 인자를 정의한 파서를 만든다.
    생성이유:
        저장소 루트, 입력 파일 경로, 출력 경로, 시드, 쿼리 컨포머를 소스로 쓸지 여부를
        CLI로 지정하기 위한 인자 사양을 한 곳에 모으기 위함.
    역할:
        --repo-root, --query-sdf, --reference-sdf, --pocket-pdb, --out, --seed,
        --keep-query-conformer-as-source 인자를 등록한다.
    메커니즘:
        argparse.ArgumentParser를 생성하고 add_argument로 각 옵션을 정의해 반환한다.
    파라미터:
        없음.
    반환:
        argparse.ArgumentParser - 위 인자들이 등록된 파서 객체.
    파이프라인 단계:
        1단계(데이터 준비) - 유사 학습 배치 CLI 인자 정의.
    """
    parser = argparse.ArgumentParser(
        description="Convert DiffAlign example into an ETFlowAlign pseudo-training .pt batch."
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
    "--keep-query-conformer-as-source",
    action="store_true",
    help=(
        "Use centered query SDF conformer as query_pos instead of random N(0,I). "
        "Default uses random source coordinates."
    ),
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
    """유사 학습 페이로드의 핵심 필드 요약을 출력한다.

    한 줄 요약:
        미리 정한 핵심 키들(타겟 포함)에 대해 형태/값 요약을 차례로 출력한다.
    생성이유:
        준비된 학습 배치가 기대한 텐서들(query_pos, target_query_pos, node_attr 등)을
        제대로 담았는지 한눈에 검증하기 위함.
    역할:
        summary_keys 목록을 순회하며 각 키에 대해 _print_tensor_shape를 호출한다.
    메커니즘:
        고정된 summary_keys 리스트를 정의하고 각 키마다 _print_tensor_shape(prefix, payload, key)를 호출한다.
    파라미터:
        prefix (str): 각 출력 줄 앞에 붙일 접두사.
        payload (dict[str, Any]): 점검할 유사 학습 페이로드.
    반환:
        None: 표준출력 부수효과만 갖는다.
    파이프라인 단계:
        1단계(데이터 준비) - 유사 학습 페이로드 요약 출력.
    """
    summary_keys = [
        "query_pos",
        "target_query_pos",
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


def compute_source_target_rmsd(payload: dict[str, Any]) -> float:
    """소스(query_pos)와 타겟(target_query_pos) 사이의 RMSD를 계산한다.

    한 줄 요약:
        페이로드의 시작 포즈와 목표 포즈 간 원자 단위 RMSD를 반환한다.
    생성이유:
        유사 학습 배치에서 시작점이 목표로부터 얼마나 떨어져 있는지(난이도/강체 변환 크기)를
        수치로 확인하기 위함.
    역할:
        query_pos와 target_query_pos가 텐서인지 검증한 뒤 둘 사이의 RMSD를 계산한다.
    메커니즘:
        좌표 차의 제곱을 마지막 축으로 합산해 원자별 제곱거리를 만들고, 평균의 제곱근으로 RMSD를 구한다.
    파라미터:
        payload (dict[str, Any]): query_pos와 target_query_pos를 포함한 페이로드.
    반환:
        float: 소스-타겟 RMSD(Å).
    예외:
        TypeError: query_pos 또는 target_query_pos가 텐서가 아닐 때.
    파이프라인 단계:
        1단계(데이터 준비) - 준비된 배치의 시작-목표 거리 점검.
    """
    query_pos = payload["query_pos"]
    target_query_pos = payload["target_query_pos"]

    if not torch.is_tensor(query_pos) or not torch.is_tensor(target_query_pos):
        raise TypeError("query_pos and target_query_pos must be torch.Tensor values.")

    rmsd = torch.sqrt(((query_pos - target_query_pos) ** 2).sum(dim=-1).mean())
    return float(rmsd)


def main() -> None:
    """CLI 진입점: DiffAlign 예제를 ETFlowAlign 유사 학습 .pt 배치로 변환·저장한다.

    한 줄 요약:
        예제 입력으로 유사 학습 페이로드를 만들고 검증·저장한 뒤 요약과 소스-타겟 RMSD를 출력한다.
    생성이유:
        스모크 테스트/단일 예제 오버피팅용 학습 배치를 손쉽게 생성하는 진입점이 필요하다.
    역할:
        인자를 파싱해 입력 경로를 확정하고, build_diffalign_example_train_payload로 페이로드를 생성,
        validate로 검증 후 저장하고, 요약/키 및 compute_source_target_rmsd 결과를 출력한다.
    메커니즘:
        _resolve_input_paths로 경로 확정 → build_diffalign_example_train_payload(randomize_query_pos는
        --keep-query-conformer-as-source의 부정) → validate_diffalign_example_train_payload →
        torch.save → print_payload_summary/print_payload_keys → compute_source_target_rmsd 출력.
    파라미터:
        없음(인자는 build_argparser로 명령행에서 파싱).
    반환:
        None: 결과는 .pt 파일 저장과 표준출력 요약이다.
    파이프라인 단계:
        1단계(데이터 준비) - 유사 학습 배치 생성 실행 진입점.
    """
    args = build_argparser().parse_args()

    repo_root = Path(args.repo_root).resolve()
    query_sdf, reference_sdf, pocket_pdb = _resolve_input_paths(args)

    payload = build_diffalign_example_train_payload(
        repo_root=repo_root,
        query_sdf=query_sdf,
        reference_sdf=reference_sdf,
        pocket_pdb=pocket_pdb,
        randomize_query_pos=not args.keep_query_conformer_as_source,
        seed=args.seed,
    )

    validate_diffalign_example_train_payload(payload)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)

    print(f"[prepare-train] saved: {out_path}")
    print_payload_summary("[prepare-train]", payload)
    print_payload_keys("[prepare-train]", payload)

    source_target_rmsd = compute_source_target_rmsd(payload)
    print(f"[prepare-train] source_target_rmsd: {source_target_rmsd:.6f} Å")


if __name__ == "__main__":
    main()