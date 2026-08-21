"""정렬쌍 SDF -> 텐서 계약 변환기 (SPEC §3.1).

per-sample payload (dict):
  z         [Nq] long   query 원자종
  pos       [Nq,3] float query 좌표 (= 학습 타깃 x1, reference에 정렬돼 있음)
  bonds     [2,2Eq] long query 결합 (양방향)
  bond_type [2Eq] long   결합 차수 코드
  ref_z     [Nr] long    reference 원자종
  ref_pos   [Nr,3] float reference 좌표 (조건)
  mol_id    str          추적 메타

query 와 reference 는 '유사하지만 다른' 분자이므로 Nq != Nr 일 수 있다.
"""

from __future__ import annotations

import torch

from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator

# 결합 차수 코드 (harmonic prior 는 연결성만 쓰지만 향후 위해 보존)
_BOND_CODE = {
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
    Chem.BondType.AROMATIC: 4,
}


def load_mol(sdf_path: str) -> Chem.Mol:
    """SDF 에서 3D 좌표를 가진 첫 분자를 읽는다 (수소 보존)."""
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True)
    for mol in supplier:
        if mol is not None and mol.GetNumConformers() > 0:
            return mol
    raise ValueError(f"유효한 3D 분자를 찾지 못함: {sdf_path}")


def mol_to_graph(mol: Chem.Mol):
    """RDKit 분자 -> (z, pos, bonds, bond_type)."""
    conf = mol.GetConformer()
    z = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long)
    pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)  # [N,3]

    src, dst, bt = [], [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        code = _BOND_CODE.get(b.GetBondType(), 1)
        src += [i, j]
        dst += [j, i]         # 양방향
        bt += [code, code]
    if src:
        bonds = torch.tensor([src, dst], dtype=torch.long)
        bond_type = torch.tensor(bt, dtype=torch.long)
    else:
        bonds = torch.zeros(2, 0, dtype=torch.long)
        bond_type = torch.zeros(0, dtype=torch.long)
    return z, pos, bonds, bond_type


def tanimoto_2d(mol1: Chem.Mol, mol2: Chem.Mol, radius: int = 2, nbits: int = 2048) -> float:
    """2D Morgan Tanimoto 유사도 (데이터 필터 검증용)."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    fp1 = gen.GetFingerprint(mol1)
    fp2 = gen.GetFingerprint(mol2)
    return float(DataStructs.TanimotoSimilarity(fp1, fp2))


def build_pair_payload(query_sdf: str, reference_sdf: str, mol_id: str = "") -> dict:
    """정렬쌍 SDF 2개 -> payload dict (SPEC §3.1)."""
    q = load_mol(query_sdf)
    r = load_mol(reference_sdf)
    qz, qpos, qbonds, qbt = mol_to_graph(q)
    rz, rpos, _, _ = mol_to_graph(r)  # reference 결합은 현재 불필요
    return {
        "z": qz,
        "pos": qpos,               # = 타깃 x1
        "bonds": qbonds,
        "bond_type": qbt,
        "ref_z": rz,
        "ref_pos": rpos,
        "mol_id": mol_id,
    }


def validate_payload(p: dict) -> None:
    """텐서 계약 최소 검증 (모양/유한성)."""
    assert p["z"].dim() == 1 and p["z"].dtype == torch.long
    assert p["pos"].shape == (p["z"].numel(), 3)
    assert p["bonds"].dim() == 2 and p["bonds"].size(0) == 2
    assert p["bond_type"].numel() == p["bonds"].size(1)
    assert p["ref_pos"].shape == (p["ref_z"].numel(), 3)
    for k in ("pos", "ref_pos"):
        assert torch.isfinite(p[k]).all(), f"{k} 에 비유한 값"
    if p["bonds"].numel() > 0:
        assert int(p["bonds"].max()) < p["z"].numel(), "결합 인덱스 범위 초과"


def collate(payloads: list) -> dict:
    """payload 리스트 -> 배치 dict (PyG 스타일 batch 인덱스).

    query 와 reference 각각에 batch 인덱스를 부여한다.
    Phase 2(조건 없음)는 query 필드만 사용, Phase 4 에서 ref_* 사용.
    """
    z, pos, bonds, bt, batch = [], [], [], [], []
    ref_z, ref_pos, ref_batch = [], [], []
    q_off, r_off = 0, 0
    for g, p in enumerate(payloads):
        nq, nr = p["z"].numel(), p["ref_z"].numel()
        z.append(p["z"]); pos.append(p["pos"])
        bonds.append(p["bonds"] + q_off); bt.append(p["bond_type"])
        batch.append(torch.full((nq,), g, dtype=torch.long))
        ref_z.append(p["ref_z"]); ref_pos.append(p["ref_pos"])
        ref_batch.append(torch.full((nr,), g, dtype=torch.long))
        q_off += nq; r_off += nr
    return {
        "z": torch.cat(z),
        "pos": torch.cat(pos, dim=0),
        "bonds": torch.cat(bonds, dim=1),
        "bond_type": torch.cat(bt),
        "batch": torch.cat(batch),
        "ref_z": torch.cat(ref_z),
        "ref_pos": torch.cat(ref_pos, dim=0),
        "ref_batch": torch.cat(ref_batch),
        "num_graphs": len(payloads),
    }
