# etflowalign/scripts/prepare_pdbbind_dataset.py
"""PDBbind refined에서 ETFlowAlign 포켓 조건부 데이터셋을 구축한다.

각 PDBbind 복합체 디렉터리 ``<id>/``에는 ``<id>_ligand.sdf`` (결합 포즈)와
``<id>_pocket.pdb`` (결합 부위 원자)가 포함된다. 각 복합체에 대해
포켓 조건부로 복합체별 ``.pt`` 페이로드(diffalign_adapter와 동일한 스키마)를 생성한다:

    target_query_pos = 리간드 결합 컨포머 - pocket_center
    query_pos        = 타겟의 무작위 강체 변환 (direction A 소스)
    pocket_pos       = 포켓 원자 - pocket_center
    reference_*      = 없음 (포켓 조건부)

실패 사례(리간드 읽기 불가 등)는 건너뛰고 로그에 기록되며,
manifest.csv에는 AlignmentDataset / train_val_split에 사용할 성공한 복합체 목록이 저장된다.

Example:
    python -m etflowalign.scripts.prepare_pdbbind_dataset \
      --pdbbind-root /home/.../dataset/PDBBind20_refined/P-L \
      --out-dir /home/.../smoke_tests/00_inputs/pdbbind_refined_pt \
      --limit 50
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import traceback
from pathlib import Path

import torch
from rdkit import Chem

from etflowalign.diffalign_adapter import (
    _make_query_source_pos,
    mol_to_bond_index_and_length_local,
    mol_to_node_attr_local,
)


def build_argparser() -> argparse.ArgumentParser:
    """PDBbind 데이터셋 구축 CLI의 argparse 파서를 구성한다.

    한 줄 요약:
        PDBbind에서 포켓 조건부 데이터셋을 만드는 데 필요한 인자를 정의한 파서를 만든다.
    생성이유:
        PDBbind 루트/출력 디렉터리/매니페스트 경로, 시드, 소스 무작위화 여부, 개수 제한,
        포켓 수소 유지 옵션을 CLI로 지정하기 위한 인자 사양을 한 곳에 모으기 위함.
    역할:
        --pdbbind-root, --out-dir, --manifest, --seed, --no-randomize-source, --limit,
        --keep-pocket-hs 인자를 등록한다.
    메커니즘:
        argparse.ArgumentParser를 생성하고 add_argument로 각 옵션을 정의해 반환한다.
    파라미터:
        없음.
    반환:
        argparse.ArgumentParser - 위 인자들이 등록된 파서 객체.
    파이프라인 단계:
        1단계(데이터 준비) - PDBbind 데이터셋 CLI 인자 정의.
    """
    p = argparse.ArgumentParser(description="Build pocket-conditioned ETFlowAlign dataset from PDBbind.")
    p.add_argument("--pdbbind-root", required=True, help="PDBbind P-L root (contains year/<id>/ dirs).")
    p.add_argument("--out-dir", required=True, help="Output directory for per-complex .pt payloads.")
    p.add_argument("--manifest", default="", help="Manifest CSV path. Default: <out-dir>/manifest.csv")
    p.add_argument("--seed", type=int, default=0, help="Base seed for the rigid source transform.")
    p.add_argument("--no-randomize-source", action="store_true",
                   help="Use the centered ligand conformer as source instead of a rigid-randomized one.")
    p.add_argument("--limit", type=int, default=0, help="Process at most N complexes (0 = all).")
    p.add_argument("--keep-pocket-hs", action="store_true", help="Keep hydrogens in the pocket coordinates.")
    return p


def find_complexes(root: str) -> list[tuple[str, str]]:
    """root 하위에서 발견된 모든 ``*_pocket.pdb``에 대해 (pdb_id, complex_dir)를 반환한다.

    한 줄 요약:
        root 아래를 재귀 탐색해 포켓 PDB가 있는 복합체의 (pdb_id, 디렉터리) 쌍 목록을 만든다.
    생성이유:
        PDBbind 디렉터리 구조에서 처리 대상 복합체를 자동으로 열거하기 위함.
    역할:
        ``*_pocket.pdb`` 파일을 찾아 각 파일에서 pdb_id와 복합체 디렉터리를 추출한다.
    메커니즘:
        glob.glob(..., recursive=True)로 포켓 파일을 정렬 수집하고, 디렉터리명과
        파일명에서 "_pocket.pdb" 접미사를 제거해 pdb_id를 얻는다.
    파라미터:
        root (str): PDBbind P-L 루트 디렉터리 경로.
    반환:
        list[tuple[str, str]]: (pdb_id, complex_dir) 쌍의 목록.
    파이프라인 단계:
        1단계(데이터 준비) - 처리 대상 복합체 열거.
    """
    out = []
    for pocket in sorted(glob.glob(os.path.join(root, "**", "*_pocket.pdb"), recursive=True)):
        cdir = os.path.dirname(pocket)
        pdb_id = os.path.basename(pocket)[: -len("_pocket.pdb")]
        out.append((pdb_id, cdir))
    return out


def load_ligand(complex_dir: str, pdb_id: str) -> Chem.Mol:
    """결합 리간드를 로드한다. SDF(정제됨, 이후 비정제됨), 그 다음 MOL2 순으로 시도한다.

    한 줄 요약:
        복합체 디렉터리에서 리간드 분자를 여러 포맷/옵션으로 시도해 컨포머와 함께 로드한다.
    생성이유:
        PDBbind 리간드 파일은 sanitize 실패가 잦으므로, 여러 대체 경로를 순차 시도해
        가능한 한 많은 복합체를 성공적으로 로드하기 위함.
    역할:
        SDF를 sanitize 켜고/끄고 시도한 뒤, 실패하면 MOL2를 sanitize 켜고/끄고 시도하여
        컨포머가 있는 첫 유효 분자를 반환한다.
    메커니즘:
        Chem.SDMolSupplier로 SDF를 두 번(sanitize True/False) 시도하고, 그래도 없으면
        Chem.MolFromMol2File로 MOL2를 시도한다. 컨포머가 1개 이상인 분자만 유효로 본다.
    파라미터:
        complex_dir (str): 복합체 파일들이 있는 디렉터리.
        pdb_id (str): 복합체 식별자(파일명 접두사).
    반환:
        Chem.Mol: 컨포머를 가진 리간드 분자.
    예외:
        ValueError: 어떤 경로로도 컨포머를 로드하지 못했을 때.
    파이프라인 단계:
        1단계(데이터 준비) - 리간드 결합 포즈 로드.
    """
    sdf = os.path.join(complex_dir, f"{pdb_id}_ligand.sdf")
    mol2 = os.path.join(complex_dir, f"{pdb_id}_ligand.mol2")

    for sanitize in (True, False):
        if os.path.exists(sdf):
            supplier = Chem.SDMolSupplier(sdf, sanitize=sanitize, removeHs=True)
            mol = supplier[0] if supplier is not None and len(supplier) > 0 else None
            if mol is not None and mol.GetNumConformers() > 0:
                return mol
    if os.path.exists(mol2):
        mol = Chem.MolFromMol2File(mol2, sanitize=True, removeHs=True)
        if mol is None:
            mol = Chem.MolFromMol2File(mol2, sanitize=False, removeHs=True)
        if mol is not None and mol.GetNumConformers() > 0:
            return mol

    raise ValueError(f"Could not load a ligand conformer for {pdb_id}")


def parse_pocket_coords(pdb_path: str, drop_h: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """포켓 PDB에서 중원자 좌표와 원자 번호를 파싱한다 (정제 과정 없음).

    반환값: (coords[Np,3] float, atom_type[Np] long atomic numbers).

    한 줄 요약:
        포켓 PDB의 ATOM/HETATM 줄을 직접 파싱해 (선택적으로 수소 제외) 좌표를 추출한다.
    생성이유:
        RDKit 정제 없이도 포켓 좌표를 견고하게 읽기 위해, PDB 고정폭 필드를 직접 파싱한다.
    역할:
        ATOM/HETATM 라인에서 원소/원자명으로 수소 여부를 판단하고, 좌표 필드를 float로 읽어 모은다.
    메커니즘:
        파일을 줄 단위로 읽으며 ATOM/HETATM만 처리한다. 컬럼 76:78의 원소(또는 원자명 첫 글자)로
        수소를 식별해 drop_h이면 건너뛰고, 컬럼 30:54의 x/y/z를 float로 파싱해 누적한다. 동시에
        원소 심볼을 주기율표로 원자번호로 변환해 좌표와 1:1로 누적한다(인식 불가 심볼은 0).
        좌표를 하나도 못 읽으면 예외를 던진다.
    파라미터:
        pdb_path (str): 포켓 PDB 파일 경로.
        drop_h (bool): True이면 수소 원자를 제외한다.
    반환:
        tuple[torch.Tensor, torch.Tensor]: (포켓 좌표 [Np,3] float 텐서,
            원자번호 [Np] long 텐서).
    예외:
        ValueError: 좌표를 하나도 파싱하지 못했을 때.
    파이프라인 단계:
        1단계(데이터 준비) - 포켓 조건 좌표 파싱.
    """
    periodic = Chem.GetPeriodicTable()
    coords: list[list[float]] = []
    atom_types: list[int] = []
    with open(pdb_path, "r") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            element = line[76:78].strip()
            atom_name = line[12:16].strip()
            symbol = element if element else atom_name[:1]
            if drop_h and symbol.upper().startswith("H"):
                continue
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except ValueError:
                continue
            try:
                atomic_num = periodic.GetAtomicNumber(symbol.capitalize())
            except Exception:  # noqa: BLE001 - 인식 불가 심볼은 미지(0) 처리
                atomic_num = 0
            coords.append([x, y, z])
            atom_types.append(atomic_num)
    if not coords:
        raise ValueError(f"No atom coordinates parsed from pocket: {pdb_path}")
    return torch.tensor(coords, dtype=torch.float32), torch.tensor(atom_types, dtype=torch.long)


def ligand_atom_types_and_pos(mol: Chem.Mol) -> tuple[torch.Tensor, torch.Tensor]:
    """리간드 분자에서 원자 번호 텐서와 좌표 텐서를 추출한다.

    한 줄 요약:
        RDKit 분자에서 원자별 원자번호와 컨포머 좌표를 텐서로 반환한다.
    생성이유:
        페이로드 구성에 필요한 리간드의 원자 종류와 결합 포즈 좌표를 표준 텐서 형태로 얻기 위함.
    역할:
        모든 원자의 원자번호를 long 텐서로, 컨포머의 (x,y,z)를 float 텐서로 만든다.
    메커니즘:
        mol.GetAtoms()로 GetAtomicNum을 모으고, mol.GetConformer()에서 각 원자의 위치를 읽어 [N,3]로 만든다.
    파라미터:
        mol (Chem.Mol): 컨포머를 가진 리간드 분자.
    반환:
        tuple[torch.Tensor, torch.Tensor]: (원자번호 [N] long, 좌표 [N,3] float).
    파이프라인 단계:
        1단계(데이터 준비) - 리간드 원자 타입/좌표 추출.
    """
    atom_types = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long)
    conf = mol.GetConformer()
    pos = torch.tensor(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
         for i in range(mol.GetNumAtoms())],
        dtype=torch.float32,
    )
    return atom_types, pos


def build_payload(complex_dir: str, pdb_id: str, *, seed: int, randomize: bool, keep_pocket_hs: bool) -> dict:
    """단일 PDBbind 복합체에 대한 포켓 조건부 학습 페이로드를 구성한다.

    한 줄 요약:
        리간드/포켓을 로드해 포켓 중심으로 센터링하고, 타겟·소스 좌표 및 결합/노드 속성을 담은 dict를 만든다.
    생성이유:
        각 복합체를 AlignmentDataset가 소비할 수 있는 공통 스키마의 .pt 페이로드로 변환하기 위함.
    역할:
        타겟 포즈(센터링된 리간드), 무작위 강체 변환 소스, 센터링된 포켓 좌표, 결합 인덱스/길이,
        노드 속성, 메타데이터(중심·소스·시드 등)를 계산해 딕셔너리로 반환한다.
    메커니즘:
        load_ligand → ligand_atom_types_and_pos로 리간드 좌표를, parse_pocket_coords로 포켓 좌표를 얻고,
        포켓 중심(pocket_center)을 빼서 target_query_pos와 pocket_centered를 만든다.
        _make_query_source_pos로 (선택적 무작위) 소스 query_pos를 생성하고,
        mol_to_bond_index_and_length_local / mol_to_node_attr_local로 결합·노드 속성을 만들어
        규약 키들을 갖는 페이로드 dict를 구성한다.
    파라미터:
        complex_dir (str): 복합체 파일 디렉터리.
        pdb_id (str): 복합체 식별자.
        seed (int, 키워드 전용): 강체 소스 변환 시드.
        randomize (bool, 키워드 전용): True이면 소스를 무작위 강체 변환, False이면 센터링 컨포머 사용.
        keep_pocket_hs (bool, 키워드 전용): 포켓 좌표에서 수소 유지 여부.
    반환:
        dict: query/target/pocket 좌표와 원자타입, 결합, 노드 속성, 메타데이터를 담은 페이로드.
    파이프라인 단계:
        1단계(데이터 준비) - 복합체별 페이로드 구성.
    """
    mol = load_ligand(complex_dir, pdb_id)
    atom_types, ligand_pos = ligand_atom_types_and_pos(mol)
    pocket_pos, pocket_atom_type = parse_pocket_coords(
        os.path.join(complex_dir, f"{pdb_id}_pocket.pdb"), drop_h=not keep_pocket_hs
    )
    
    pocket_center = pocket_pos.mean(dim=0)
    target_query_pos = ligand_pos - pocket_center
    pocket_centered = pocket_pos - pocket_center

    query_pos = _make_query_source_pos(target_query_pos, randomize_query_pos=randomize, seed=seed)
    bond_index, bond_length = mol_to_bond_index_and_length_local(mol, target_query_pos)
    node_attr = mol_to_node_attr_local(mol)

    n = ligand_pos.size(0)
    return {
        "query_pos": query_pos.float(),
        "query_atom_type": atom_types.long(),
        "query_batch": torch.zeros(n, dtype=torch.long),
        "target_query_pos": target_query_pos.float(),
        "query_node_attr": node_attr.float(),
        "query_bond_index": bond_index.long(),
        "query_bond_length": bond_length.float(),
        "pocket_pos": pocket_centered.float(),
        "pocket_batch": torch.zeros(pocket_centered.size(0), dtype=torch.long),
        "pocket_atom_type": pocket_atom_type.long(),
        "reference_center_subtracted": pocket_center.float(),
        "source": "pdbbind_refined",
        "pdb_id": pdb_id,
        "query_source": "rigid_randomized_ligand_conformer" if randomize else "centered_ligand_conformer",
        "conditioning": "pocket",
        "seed": seed,
    }


def main() -> None:
    """CLI 진입점: PDBbind 복합체들을 순회하며 페이로드를 생성하고 매니페스트를 작성한다.

    한 줄 요약:
        PDBbind 복합체를 열거해 각각의 .pt 페이로드를 저장하고, 성공 목록 매니페스트와 실패 로그를 남긴다.
    생성이유:
        대규모 PDBbind 데이터를 불량 복합체에 강건하게 일괄 전처리하여 학습용 데이터셋을 구축하기 위함.
    역할:
        출력 디렉터리/매니페스트 경로를 준비하고, 복합체를 (옵션상 제한 적용해) 순회하며 build_payload로
        페이로드를 만들어 저장하고, 성공/실패를 집계해 매니페스트 CSV와 failures.log를 기록한다.
    메커니즘:
        find_complexes로 목록을 얻고 limit 적용. 각 복합체마다 try/except로 build_payload(시드는 base+i)를
        호출해 torch.save하고 rows에 메타를 누적, 예외 시 failures.log에 traceback을 기록한다.
        200개마다 진행 상황을 출력하고, 마지막에 manifest.csv를 쓰고 요약을 출력한다.
    파라미터:
        없음(인자는 build_argparser로 명령행에서 파싱).
    반환:
        None: 결과는 .pt 파일들, manifest.csv, failures.log 작성과 표준출력 요약이다.
    파이프라인 단계:
        1단계(데이터 준비) - PDBbind 데이터셋 일괄 구축 진입점.
    """
    args = build_argparser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else out_dir / "manifest.csv"

    complexes = find_complexes(args.pdbbind_root)
    if args.limit > 0:
        complexes = complexes[: args.limit]
    print(f"[pdbbind] found {len(complexes)} complexes under {args.pdbbind_root}")

    rows = []
    n_ok = n_fail = 0
    with open(out_dir / "failures.log", "w") as flog:
        for i, (pdb_id, cdir) in enumerate(complexes):
            try:
                payload = build_payload(
                    cdir, pdb_id,
                    seed=args.seed + i,
                    randomize=not args.no_randomize_source,
                    keep_pocket_hs=args.keep_pocket_hs,
                )
                out_path = out_dir / f"{pdb_id}.pt"
                torch.save(payload, out_path)
                rows.append({
                    "pdb_id": pdb_id,
                    "path": str(out_path),
                    "n_atoms": int(payload["query_pos"].size(0)),
                    "n_pocket": int(payload["pocket_pos"].size(0)),
                })
                n_ok += 1
            except Exception as exc:  # noqa: BLE001 - 데이터셋 전처리는 불량 복합체에 강건해야 함
                n_fail += 1
                flog.write(f"{pdb_id}\t{exc}\n{traceback.format_exc()}\n")
            if (i + 1) % 200 == 0:
                print(f"[pdbbind] {i + 1}/{len(complexes)}  ok={n_ok} fail={n_fail}")

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pdb_id", "path", "n_atoms", "n_pocket"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[pdbbind] done: ok={n_ok} fail={n_fail}")
    print(f"[pdbbind] manifest: {manifest_path}")
    print(f"[pdbbind] failures log: {out_dir / 'failures.log'}")


if __name__ == "__main__":
    main()
