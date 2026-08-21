#!/usr/bin/env python3
"""
4b_prep_validity.py (보강 ① 준비단계) - PoseBusters 입력 준비. boltz2 env (gemmi+rdkit).
ligand_clusters.csv 의 각 대표 포즈마다:
  - 단백질 PDB 저장
  - 리간드 SDF(SMILES connectivity + 예측 좌표) 저장
  - manifest.csv 에 (pocket,lig,cif,sdf,pdb) 기록
실제 검사는 4c_posebusters.py(casp_eval env)가 manifest를 읽어 수행 → env 분리(gemmi/posebusters).
출력: 04b_posebusters/inputs/* + manifest.csv
"""
import argparse, csv, os
import gemmi
import complex_io as cio


def build_ligand_mol(smiles, atoms):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Geometry import Point3D
    template = Chem.RemoveHs(Chem.MolFromSmiles(smiles))
    n = template.GetNumAtoms()
    if len(atoms) != n:
        raise ValueError(f"원자 수 불일치 {n} vs {len(atoms)}")
    same = all(template.GetAtomWithIdx(i).GetSymbol().capitalize() == el.capitalize()
               for i, (el, _) in enumerate(atoms))
    if same:
        m = Chem.Mol(template); conf = Chem.Conformer(n)
        for i, (_, xyz) in enumerate(atoms):
            conf.SetAtomPosition(i, Point3D(*[float(v) for v in xyz]))
        m.RemoveAllConformers(); m.AddConformer(conf, assignId=True)
        return m
    from rdkit.Chem import rdDetermineBonds
    rw = Chem.RWMol(); conf = Chem.Conformer(n)
    for i, (el, xyz) in enumerate(atoms):
        rw.AddAtom(Chem.Atom(el.capitalize())); conf.SetAtomPosition(i, Point3D(*[float(v) for v in xyz]))
    m = rw.GetMol(); m.AddConformer(conf, assignId=True)
    rdDetermineBonds.DetermineConnectivity(m)
    return AllChem.AssignBondOrdersFromTemplate(template, m)


def main():
    from rdkit import Chem
    ap = argparse.ArgumentParser()
    ap.add_argument("--ligand-clusters", required=True)
    ap.add_argument("--ligand-tsv", required=True)
    ap.add_argument("--out", required=True, help="04b_posebusters 폴더")
    args = ap.parse_args()
    inp = os.path.join(args.out, "inputs"); os.makedirs(inp, exist_ok=True)

    smis = cio.read_ligand_tsv(args.ligand_tsv)
    if not smis:
        raise SystemExit("ligand.tsv 비어있음")

    man = []
    for r in csv.DictReader(open(args.ligand_clusters)):
        cif = r["rep_cif"]; tag = f"P{r['pocket_id']}_L{r['ligand_cluster_id']}"
        _, ligs = cio.parse_complex(cif)
        if not ligs:
            continue
        match = cio.match_ligand_to_smiles(ligs[0], smis)
        row = {"pocket_id": r["pocket_id"], "ligand_cluster_id": r["ligand_cluster_id"],
               "rep_cif": cif, "sdf": "", "protein_pdb": "", "note": ""}
        if match is None:
            row["note"] = "no_smiles_match"; man.append(row); continue
        try:
            mol = build_ligand_mol(match[2], ligs[0]["atoms"])
            sdf = os.path.join(inp, f"{tag}.sdf")
            w = Chem.SDWriter(sdf); w.write(mol); w.close()
            pdb = os.path.join(inp, f"{tag}_protein.pdb")
            st = gemmi.read_structure(cif); st.setup_entities()
            st.remove_ligands_and_waters(); st.remove_empty_chains(); st.write_pdb(pdb)
            row["sdf"] = sdf; row["protein_pdb"] = pdb
        except Exception as e:
            row["note"] = f"prep_err:{str(e)[:60]}"
        man.append(row)
        print(f"  {tag}: sdf={'O' if row['sdf'] else 'X'} pdb={'O' if row['protein_pdb'] else 'X'} {row['note']}")

    mcsv = os.path.join(args.out, "manifest.csv")
    with open(mcsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pocket_id", "ligand_cluster_id", "rep_cif",
                                          "sdf", "protein_pdb", "note"])
        w.writeheader(); w.writerows(man)
    print(f"[prep_validity] {len(man)}개 준비 -> {mcsv}")


if __name__ == "__main__":
    main()

# ── 단독 실행 ── (boltz2 env python; gemmi+rdkit)
#   <python_sci> $SC/4b_prep_validity.py \
#       --ligand-clusters $OUT/04_ligand_clusters/ligand_clusters.csv \
#       --ligand-tsv <ligand_tsv> --out $OUT/04b_posebusters
