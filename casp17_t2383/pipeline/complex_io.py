#!/usr/bin/env python3
"""
complex_io.py - 범용 단백질-리간드 복합체 파서 (gemmi).

타겟 종류·체인 수·리간드 개수에 무관하게 동작하도록, 모든 스텝이 공유하는 공용 파서.
- 단백질: 모든 polymer(아미노산) 체인의 Cα
- 리간드: 아미노산/핵산/물이 아닌 모든 잔기를 'copy' 단위로 검출 (개수 무관)
- 리간드 좌표를 canonical 순서로 concat → 구조 간 RMSD/클러스터링 가능
- 리간드 ↔ SMILES를 원소 조성으로 매칭 (ligand.tsv에 SMILES가 여러 개여도)

주의(범용성의 한계): 동일 조성 리간드가 '여러 copy'일 때 copy 간 대응은
seqid 순서로 고정한다. 진짜 교환대칭(symmetry) 최적 배정은 하지 않음(보수적).
"""
from collections import Counter
import gemmi

WATER = {"HOH", "WAT", "DOD", "H2O"}


def parse_complex(path):
    """
    반환:
      prot_ca : dict[(chain, resnum)] -> [x,y,z]   (모든 단백질 체인의 Cα)
      ligands : list of dict {key:(chain,resname,seqid), resname, atoms:[(element,[x,y,z]),...]}
    """
    st = gemmi.read_structure(path)
    if len(st) == 0:
        return {}, []
    m = st[0]
    prot_ca, ligands = {}, []
    for ch in m:
        for res in ch:
            t = gemmi.find_tabulated_residue(res.name)
            is_aa = bool(t) and t.is_amino_acid()
            is_na = bool(t) and t.is_nucleic_acid()
            if is_aa:
                for a in res:
                    if a.name == "CA":
                        prot_ca[(ch.name, res.seqid.num)] = [a.pos.x, a.pos.y, a.pos.z]
                        break
            elif is_na or res.name in WATER:
                continue                       # 핵산·물은 리간드 아님
            else:
                atoms = [(a.element.name, [a.pos.x, a.pos.y, a.pos.z])
                         for a in res if a.element.name != "H"]
                if atoms:
                    ligands.append({"key": (ch.name, res.name, res.seqid.num),
                                    "resname": res.name, "atoms": atoms})
    return prot_ca, ligands


def _canon_order(ligands):
    """리간드 copy를 (resname, chain, seqid)로 정렬한 인덱스 → 구조 간 일관 순서."""
    return sorted(range(len(ligands)),
                  key=lambda i: (ligands[i]["resname"],
                                 ligands[i]["key"][0], ligands[i]["key"][2]))


def ligand_concat(ligands):
    """
    모든 리간드 heavy atom을 canonical 순서로 이어붙임.
    반환: (elements:list[str], coords:list[[x,y,z]])
    elements는 구조 간 '서명' — 같은 타겟이면 모든 구조에서 동일해야 함.
    """
    elems, coords = [], []
    for i in _canon_order(ligands):
        for el, xyz in ligands[i]["atoms"]:
            elems.append(el); coords.append(xyz)
    return elems, coords


def ligand_centroid(ligands):
    """모든 리간드 heavy atom의 무게중심 [x,y,z] (없으면 None)."""
    pts = [xyz for lg in ligands for _, xyz in lg["atoms"]]
    if not pts:
        return None
    n = len(pts)
    return [sum(p[k] for p in pts) / n for k in range(3)]


def composition(atoms_or_elems):
    """원소 multiset(Counter). 리간드↔SMILES 조성 매칭용."""
    if atoms_or_elems and isinstance(atoms_or_elems[0], tuple):
        elems = [el for el, _ in atoms_or_elems]
    else:
        elems = list(atoms_or_elems)
    return Counter(e.capitalize() for e in elems)


def smiles_composition(smiles):
    """SMILES의 heavy-atom 원소 조성(Counter). rdkit 필요."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.RemoveHs(mol)
    return Counter(a.GetSymbol() for a in mol.GetAtoms())


def match_ligand_to_smiles(ligand, smiles_list):
    """
    리간드 한 copy를 후보 SMILES들과 원소 조성으로 매칭.
    smiles_list: [(id, name, smiles), ...]
    반환: (id, name, smiles) 또는 None.
    - 후보 1개면 그대로 사용(단일 리간드 타겟).
    - 여러 개면 조성이 정확히 일치하는 것을 선택.
    """
    if len(smiles_list) == 1:
        return smiles_list[0]
    lig_comp = composition(ligand["atoms"])
    for sid, name, smi in smiles_list:
        if smiles_composition(smi) == lig_comp:
            return (sid, name, smi)
    return None


def read_ligand_tsv(path):
    """ligand.tsv → [(ID, Name, SMILES), ...] (행 여러 개 = 리간드 여러 종)."""
    import csv
    out = []
    with open(path, newline="") as h:
        for r in csv.DictReader(h, delimiter="\t"):
            if r.get("SMILES"):
                out.append((r.get("ID", ""), r.get("Name", ""), r["SMILES"]))
    return out


def write_ligands_pdb(ligands, pdb_out):
    """모든 리간드 copy를 한 PDB(HETATM, 체인 L)로 저장. gnina/도킹용."""
    lines = []
    serial = 1
    for ridx, lg in enumerate(ligands, 1):
        rn = lg["resname"][:3].upper()
        for el, (x, y, z) in lg["atoms"]:
            nm = f"{el:<2}"
            lines.append(
                f"HETATM{serial:5d} {nm:<4} {rn:>3} L{ridx:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {el:>2}")
            serial += 1
    lines.append("END")
    open(pdb_out, "w").write("\n".join(lines) + "\n")
    return serial - 1

# 이 파일은 import 전용 공용 라이브러리입니다 (단독 실행 X).
# 스텝들이 `import complex_io` 로 사용 → 스텝들과 같은 폴더에 두어야 함.
