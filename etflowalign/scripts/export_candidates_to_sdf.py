from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from rdkit import Chem
'''[역할]
1. diffalign_example_out.pt 로드
2. input_metadata.query_sdf 로드
3. candidates[k] 좌표를 query mol에 주입
4. reference_center_subtracted를 더해 global coordinate 복원
5. candidate_k.sdf 저장
'''
'''[목표]
diffalign_example_out.pt
        ↓
candidates tensor [K, Nq, 3]
        ↓
query_sdf RDKit Mol에 좌표 주입
        ↓
candidate_000.sdf, candidate_001.sdf 저장
'''

def load_pt(path: str | Path) -> dict[str, Any]:
    """ETFlowAlign .pt 출력 파일을 로드한다.

    이 파일은 자체 파이프라인에서 생성되므로, 향후 기본값 변경에 따른
    모호함을 피하기 위해 weights_only=False를 명시적으로 사용한다.

    한 줄 요약:
        추론 출력 .pt를 로드하고 필수 키와 candidates 텐서 형태를 검증해 반환한다.
    생성이유:
        SDF로 내보낼 후보 좌표/점수/메타데이터가 올바른 스키마를 갖는지 사전에 보장하여,
        이후 단계에서 형태 오류가 늦게 터지는 것을 막기 위함.
    역할:
        파일 존재 여부를 확인하고 torch.load로 로드한 뒤, dict 타입과 candidates/scores/metadata 키,
        candidates의 [K,N,3] 형태를 검증한다.
    메커니즘:
        Path로 존재 확인 → torch.load(weights_only=False) → dict/키/텐서/형태를 차례로 검사하고
        조건 위반 시 적절한 예외를 던진다.
    파라미터:
        path (str | Path): 추론 출력 .pt 파일 경로.
    반환:
        dict[str, Any]: candidates/scores/metadata 등을 담은 페이로드 딕셔너리.
    예외:
        FileNotFoundError, TypeError, KeyError, ValueError: 파일/스키마 검증 실패 시.
    파이프라인 단계:
        8단계(후처리) - 후보 포즈 SDF 내보내기 입력 로드.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input .pt file not found: {path}")

    obj = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict payload, got {type(obj)}")

    required = ["candidates", "scores", "metadata"]
    for key in required:
        if key not in obj:
            raise KeyError(f"Missing required key in output .pt: {key}")

    if not torch.is_tensor(obj["candidates"]):
        raise TypeError("obj['candidates'] must be a torch.Tensor")

    if obj["candidates"].ndim != 3 or obj["candidates"].shape[-1] != 3:
        raise ValueError(
            "obj['candidates'] must have shape [K, N, 3], "
            f"got {tuple(obj['candidates'].shape)}"
        )

    return obj


def load_query_mol(path: str | Path, *, sanitize: bool = True, remove_hs: bool = True) -> Chem.Mol:
    """SDF/MOL 파일에서 쿼리 리간드 템플릿 분자를 로드한다.

    한 줄 요약:
        SDF/MOL 파일을 읽어 컨포머를 가진 RDKit Mol 템플릿을 반환한다.
    생성이유:
        모델이 출력한 후보 좌표를 주입할 대상 분자(원자 종류·결합·수소 구성)가 필요한데,
        이는 원본 쿼리 분자에서 와야 하므로 이를 안전하게 로드하기 위함.
    역할:
        파일 존재와 확장자(.sdf/.mol)를 확인하고 분자를 로드한 뒤, 분자가 유효하고
        컨포머가 1개 이상 존재하는지 검증한다.
    메커니즘:
        Chem.SDMolSupplier로 첫 분자를 읽고(sanitize/remove_hs 옵션 적용), None이거나 컨포머가 없으면 예외를 던진다.
    파라미터:
        path (str | Path): 쿼리 분자 파일 경로(.sdf 또는 .mol).
        sanitize (bool, 키워드 전용): RDKit sanitize 적용 여부.
        remove_hs (bool, 키워드 전용): 로드 시 수소 제거 여부.
    반환:
        Chem.Mol: 컨포머를 가진 RDKit 분자 템플릿.
    예외:
        FileNotFoundError, ValueError: 파일 없음/지원하지 않는 확장자/로드 실패/컨포머 없음 시.
    파이프라인 단계:
        8단계(후처리) - 좌표 주입용 템플릿 분자 로드.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Query molecule file not found: {path}")

    suffix = path.suffix.lower()

    if suffix in {".sdf", ".mol"}:
        supplier = Chem.SDMolSupplier(str(path), sanitize=sanitize, removeHs=remove_hs)
        mol = supplier[0] if supplier is not None and len(supplier) > 0 else None
    else:
        raise ValueError(f"Unsupported query molecule extension: {path.suffix}")

    if mol is None:
        raise ValueError(f"Failed to load query molecule: {path}")

    if mol.GetNumConformers() == 0:
        raise ValueError(f"Query molecule has no conformer: {path}")

    return mol


def set_mol_positions(mol: Chem.Mol, coords: torch.Tensor) -> Chem.Mol:
    """coords[N,3]로 컨포머 좌표를 설정한 mol의 복사본을 반환한다.

    한 줄 요약:
        입력 분자의 복사본을 만들고 그 컨포머 좌표를 coords로 덮어써서 반환한다.
    생성이유:
        원본 템플릿 분자를 훼손하지 않으면서 각 후보 좌표를 분자에 주입한 새 분자를
        만들어야 하므로, 좌표 설정을 캡슐화한 함수가 필요하다.
    역할:
        coords 형태와 원자 수 일치를 검증한 뒤, mol을 복사하고 컨포머의 각 원자 위치를 coords로 설정한다.
    메커니즘:
        coords를 CPU float로 변환하고 [N,3]·원자수를 확인한 뒤 Chem.Mol(mol)로 복사,
        컨포머가 없으면 생성·추가하고, 모든 원자에 대해 SetAtomPosition으로 좌표를 설정한다.
    파라미터:
        mol (Chem.Mol): 좌표를 주입할 템플릿 분자.
        coords (torch.Tensor): 설정할 원자 좌표 [N,3].
    반환:
        Chem.Mol: 좌표가 갱신된 분자 복사본.
    예외:
        ValueError: coords 형태가 [N,3]이 아니거나 원자 수가 불일치할 때.
    파이프라인 단계:
        8단계(후처리) - 후보 좌표 주입.
    """
    coords = coords.detach().cpu().float()

    if coords.ndim != 2 or coords.shape[-1] != 3:
        raise ValueError(f"coords must have shape [N, 3], got {tuple(coords.shape)}")

    if mol.GetNumAtoms() != coords.shape[0]:
        raise ValueError(
            "Atom count mismatch between query molecule and candidate coordinates: "
            f"mol atoms={mol.GetNumAtoms()}, coords atoms={coords.shape[0]}"
        )

    out = Chem.Mol(mol)

    if out.GetNumConformers() == 0:
        conf = Chem.Conformer(out.GetNumAtoms())
        out.AddConformer(conf, assignId=True)

    conf = out.GetConformer()

    for atom_idx in range(out.GetNumAtoms()):
        x, y, z = coords[atom_idx].tolist()
        conf.SetAtomPosition(atom_idx, (float(x), float(y), float(z)))

    return out


def get_query_sdf_from_metadata(obj: dict[str, Any]) -> str | None:
    """ETFlowAlign 출력 메타데이터에서 query_sdf 경로를 추출한다.

    한 줄 요약:
        출력 페이로드의 metadata.input_metadata.query_sdf 경로를 안전하게 꺼내 반환한다.
    생성이유:
        사용자가 --query-sdf를 생략했을 때 추론 시 사용했던 원본 쿼리 SDF 경로를
        출력 메타데이터에서 자동으로 찾아 쓰기 위함.
    역할:
        metadata와 input_metadata가 dict인지 확인하며 중첩 키에서 query_sdf를 추출한다.
    메커니즘:
        obj.get("metadata") → input_metadata.get("query_sdf")를 단계적으로 꺼내고,
        타입이 어긋나거나 키가 없으면 None을, 있으면 문자열로 변환해 반환한다.
    파라미터:
        obj (dict[str, Any]): load_pt로 로드한 출력 페이로드.
    반환:
        str | None: query_sdf 경로 문자열 또는 찾지 못한 경우 None.
    파이프라인 단계:
        8단계(후처리) - 템플릿 경로 자동 추론.
    """
    metadata = obj.get("metadata", {})
    if not isinstance(metadata, dict):
        return None

    input_metadata = metadata.get("input_metadata", {})
    if not isinstance(input_metadata, dict):
        return None

    query_sdf = input_metadata.get("query_sdf")
    if query_sdf is None:
        return None

    return str(query_sdf)


def get_reference_center_from_metadata(obj: dict[str, Any]) -> torch.Tensor | None:
    """ETFlowAlign 출력 메타데이터에서 레퍼런스 중심 벡터를 추출한다.

    한 줄 요약:
        metadata.input_metadata.reference_center_subtracted를 길이-3 텐서로 추출한다.
    생성이유:
        모델은 중심을 뺀 센터링 좌표로 후보를 출력하므로, 전역 좌표로 복원하려면
        뺐던 중심 벡터를 메타데이터에서 가져와야 한다.
    역할:
        중첩 메타데이터에서 중심 벡터를 꺼내 [3] 형태의 float 텐서로 변환·검증한다.
    메커니즘:
        metadata/input_metadata가 dict인지 확인하고 center 값을 꺼내, 텐서면 CPU float로,
        아니면 torch.tensor로 변환한 뒤 형태가 (3,)인지 검증한다.
    파라미터:
        obj (dict[str, Any]): load_pt로 로드한 출력 페이로드.
    반환:
        torch.Tensor | None: 길이-3 중심 벡터 또는 찾지 못한 경우 None.
    예외:
        ValueError: 중심 벡터의 형태가 [3]이 아닐 때.
    파이프라인 단계:
        8단계(후처리) - 전역 좌표 복원용 중심 벡터 추출.
    """
    metadata = obj.get("metadata", {})
    if not isinstance(metadata, dict):
        return None

    input_metadata = metadata.get("input_metadata", {})
    if not isinstance(input_metadata, dict):
        return None

    center = input_metadata.get("reference_center_subtracted")
    if center is None:
        return None

    if torch.is_tensor(center):
        center_tensor = center.detach().cpu().float()
    else:
        center_tensor = torch.tensor(center, dtype=torch.float32)

    if center_tensor.shape != (3,):
        raise ValueError(
            "reference_center_subtracted must have shape [3], "
            f"got {tuple(center_tensor.shape)}"
        )

    return center_tensor


def write_candidate_sdfs(
    *,
    candidates: torch.Tensor,
    scores: torch.Tensor,
    query_mol: Chem.Mol,
    out_dir: str | Path,
    prefix: str = "candidate",
    restore_center: torch.Tensor | None = None,
    top_k: int | None = None,
    write_multisdf: bool = True,
) -> list[Path]:
    """후보 컨포메이션을 개별 SDF 파일로 저장한다.

    한 줄 요약:
        각 후보 좌표를 템플릿 분자에 주입해 개별 SDF로 저장하고, 선택적으로 통합 멀티 SDF도 작성한다.
    생성이유:
        모델이 출력한 [K,N,3] 후보 좌표들을 사람이/도킹 도구가 사용할 수 있는 SDF 포즈
        파일들로 변환하기 위한 핵심 내보내기 루틴이 필요하다.
    역할:
        출력 디렉터리를 만들고 점수 길이를 검증한 뒤, 선택된 후보들에 대해 좌표(필요 시 중심 복원)를
        분자에 주입하고 이름/인덱스/순위/점수 속성을 설정해 개별 SDF 및 통합 SDF로 기록한다.
    메커니즘:
        candidates/scores를 CPU float로 변환·검증하고, top_k에 따라 인덱스를 선택한다.
        write_multisdf면 멀티 SDWriter를 연다. 각 후보마다 restore_center가 있으면 좌표에 더하고
        set_mol_positions로 분자를 만든 뒤 SetProp으로 메타데이터를 붙여 개별 SDWriter로 저장하고
        멀티 writer에도 기록한다. finally에서 멀티 writer를 닫고 경로 목록에 추가한다.
    파라미터:
        candidates (torch.Tensor, 키워드 전용): 후보 좌표 [K,N,3].
        scores (torch.Tensor, 키워드 전용): 후보 점수 [K].
        query_mol (Chem.Mol, 키워드 전용): 좌표를 주입할 템플릿 분자.
        out_dir (str | Path, 키워드 전용): SDF 출력 디렉터리.
        prefix (str, 키워드 전용): 출력 파일 이름 접두사.
        restore_center (torch.Tensor | None, 키워드 전용): 전역 복원용 중심 벡터(있으면 좌표에 더함).
        top_k (int | None, 키워드 전용): 내보낼 후보 수(None/0 이하면 전체).
        write_multisdf (bool, 키워드 전용): 통합 멀티 분자 SDF 작성 여부.
    반환:
        list[Path]: 작성된 모든 SDF 파일 경로(통합 SDF 포함 시 마지막에 추가).
    예외:
        ValueError: scores 길이가 후보 수 K와 불일치할 때.
    파이프라인 단계:
        8단계(후처리) - 후보 포즈 SDF 내보내기 핵심 루틴.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = candidates.detach().cpu().float()

    if not torch.is_tensor(scores):
        scores = torch.tensor(scores, dtype=torch.float32)
    scores = scores.detach().cpu().float().view(-1)

    num_candidates = candidates.shape[0]

    if scores.numel() != num_candidates:
        raise ValueError(
            f"scores length must match candidates K. scores={scores.numel()}, K={num_candidates}"
        )

    if top_k is None or top_k <= 0:
        selected_indices = list(range(num_candidates))
    else:
        selected_indices = list(range(min(int(top_k), num_candidates)))

    written_paths: list[Path] = []

    multi_writer = None
    multi_path = out_dir / f"{prefix}_all.sdf"
    if write_multisdf:
        multi_writer = Chem.SDWriter(str(multi_path))

    try:
        for rank, cand_idx in enumerate(selected_indices):
            coords = candidates[cand_idx]

            if restore_center is not None:
                coords = coords + restore_center.view(1, 3)

            mol = set_mol_positions(query_mol, coords)

            score_value = float(scores[cand_idx].item())
            mol.SetProp("_Name", f"{prefix}_{cand_idx:03d}")
            mol.SetProp("candidate_index", str(cand_idx))
            mol.SetProp("rank", str(rank))
            mol.SetProp("score", f"{score_value:.8f}")

            out_path = out_dir / f"{prefix}_{cand_idx:03d}.sdf"
            writer = Chem.SDWriter(str(out_path))
            writer.write(mol)
            writer.close()

            written_paths.append(out_path)

            if multi_writer is not None:
                multi_writer.write(mol)

    finally:
        if multi_writer is not None:
            multi_writer.close()
            written_paths.append(multi_path)

    return written_paths


def build_argparser() -> argparse.ArgumentParser:
    """후보 SDF 내보내기 CLI의 argparse 파서를 구성한다.

    한 줄 요약:
        후보 좌표를 SDF로 내보내는 데 필요한 명령행 인자를 정의한 ArgumentParser를 만든다.
    생성이유:
        입력 .pt, 쿼리 SDF, 출력 디렉터리/접두사, top-k, 중심 복원·멀티 SDF·수소·sanitize 옵션을
        CLI로 지정하기 위한 인자 사양을 한 곳에 모으기 위함.
    역할:
        --input-pt, --query-sdf, --out-dir, --prefix, --top-k, --no-restore-center,
        --no-multisdf, --keep-hs, --no-sanitize 인자를 등록한다.
    메커니즘:
        argparse.ArgumentParser를 생성하고 add_argument로 각 옵션을 정의해 반환한다.
    파라미터:
        없음.
    반환:
        argparse.ArgumentParser - 위 인자들이 등록된 파서 객체.
    파이프라인 단계:
        8단계(후처리) - CLI 인자 정의.
    """
    parser = argparse.ArgumentParser(
        description="Export ETFlowAlign candidates from .pt to SDF files."
    )

    parser.add_argument(
        "--input-pt",
        type=str,
        required=True,
        help="ETFlowAlign inference output .pt path.",
    )
    parser.add_argument(
        "--query-sdf",
        type=str,
        default="",
        help=(
            "Query SDF template path. If omitted, the script uses "
            "metadata['input_metadata']['query_sdf'] from the .pt file."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for SDF files.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="candidate",
        help="Output SDF filename prefix.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Number of candidates to export. 0 means export all.",
    )
    parser.add_argument(
        "--no-restore-center",
        action="store_true",
        help=(
            "Do not add metadata['input_metadata']['reference_center_subtracted'] "
            "to candidate coordinates."
        ),
    )
    parser.add_argument(
        "--no-multisdf",
        action="store_true",
        help="Do not write a combined multi-molecule SDF.",
    )
    parser.add_argument(
        "--keep-hs",
        action="store_true",
        help="Keep hydrogens when loading query SDF. Default removes hydrogens.",
    )
    parser.add_argument(
        "--no-sanitize",
        action="store_true",
        help="Disable RDKit sanitization when loading query SDF.",
    )

    return parser


def main() -> None:
    """CLI 진입점: 추론 출력 .pt의 후보들을 SDF 파일로 내보낸다.

    한 줄 요약:
        출력 .pt와 쿼리 템플릿을 로드해 후보 좌표를 SDF로 저장하고 결과를 출력한다.
    생성이유:
        추론 결과(후보 포즈)를 표준 분자 파일 형식으로 떨어뜨려 시각화/도킹 후속 작업에
        바로 쓸 수 있게 하는 후처리 도구의 진입점이 필요하다.
    역할:
        인자를 파싱해 .pt를 로드하고, 쿼리 SDF 경로를 인자 또는 메타데이터에서 확정하며,
        템플릿 분자를 로드하고 옵션에 따라 중심 복원 벡터를 구한 뒤 write_candidate_sdfs로
        SDF를 작성하고 입력/출력 요약을 출력한다.
    메커니즘:
        load_pt → query_sdf 결정(인자 우선, 없으면 get_query_sdf_from_metadata) → load_query_mol →
        no_restore_center가 아니면 get_reference_center_from_metadata로 중심 확보 →
        write_candidate_sdfs 호출 → 작성된 경로 목록 출력.
    파라미터:
        없음(인자는 build_argparser로 명령행에서 파싱).
    반환:
        None: 결과는 SDF 파일 작성과 표준출력 요약이다.
    예외:
        ValueError: 쿼리 SDF 경로를 인자/메타데이터 어디서도 얻을 수 없을 때.
    파이프라인 단계:
        8단계(후처리) - 후보 SDF 내보내기 실행 진입점.
    """
    args = build_argparser().parse_args()

    obj = load_pt(args.input_pt)

    query_sdf = args.query_sdf.strip()
    if not query_sdf:
        query_sdf = get_query_sdf_from_metadata(obj) or ""

    if not query_sdf:
        raise ValueError(
            "Query SDF path was not provided and could not be inferred from metadata. "
            "Pass --query-sdf explicitly."
        )

    query_mol = load_query_mol(
        query_sdf,
        sanitize=not args.no_sanitize,
        remove_hs=not args.keep_hs,
    )

    candidates = obj["candidates"]
    scores = obj["scores"]

    restore_center = None
    if not args.no_restore_center:
        restore_center = get_reference_center_from_metadata(obj)
        if restore_center is None:
            print("[export] reference_center_subtracted not found; exporting centered coordinates.")
        else:
            print(f"[export] restoring global coordinates with center={restore_center.tolist()}")
    else:
        print("[export] exporting centered coordinates without global restoration.")

    written = write_candidate_sdfs(
        candidates=candidates,
        scores=scores,
        query_mol=query_mol,
        out_dir=args.out_dir,
        prefix=args.prefix,
        restore_center=restore_center,
        top_k=args.top_k,
        write_multisdf=not args.no_multisdf,
    )

    print(f"[export] input_pt: {args.input_pt}")
    print(f"[export] query_sdf: {query_sdf}")
    print(f"[export] candidates: {tuple(candidates.shape)}")
    print(f"[export] scores: {tuple(scores.shape)}")
    print(f"[export] out_dir: {args.out_dir}")
    print("[export] written files:")
    for path in written:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
    
'''[실행 예제 - global 복원 없이, 모델이 직접 낸 centered coordinate를 보고 싶다면]
python -m etflowalign.scripts.export_candidates_to_sdf \
  --input-pt /home/deepfold/users/hosung/work/ETFlowAlign/etflowalign/smoke_tests/diffalign_example_out.pt \
  --out-dir /home/deepfold/users/hosung/work/ETFlowAlign/etflowalign/smoke_tests/diffalign_example_sdf_centered \
  --prefix diffalign_example_candidate_centered \
  --no-restore-center
'''

'''[실행 예제]
 python -m etflowalign.scripts.export_candidates_to_sdf \
  --input-pt /home/deepfold/users/hosung/work/ETFlowAlign/etflowalign/smoke_tests/diffalign_example_out.pt \
  --out-dir /home/deepfold/users/hosung/work/ETFlowAlign/etflowalign/smoke_tests/diffalign_example_sdf \
  --prefix diffalign_example_candidate        
'''

