#!/usr/bin/env python3
"""
make_casp_lg.py (범용) - 선별 포즈 → CASP LG 포맷. complex_io 사용 → 리간드 개수 무관.

각 모델 cif에서:
  - 수용체: 모든 단백질 체인 ATOM 라인
  - 리간드: copy마다 SMILES를 조성으로 매칭 → connectivity+좌표로 mol block.
    리간드가 여러 개면 LIGAND/mol block을 여러 개 출력.
컬럼 스키마 유연(LSCORE = ranking_score|lscore|iptm). rdkit 필요.
"""
import argparse, csv, os, shlex
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
import complex_io as cio

AA_WATER = cio.WATER


def parse_atom_site(cif_path):
    rows, cols, in_site = [], [], False
    with open(cif_path) as h:
        for raw in h:
            line = raw.strip()
            if not line:
                continue
            if line == "loop_":
                cols, in_site = [], False; continue
            if raw.startswith("_atom_site."):
                cols.append(raw.strip().split(".")[-1]); in_site = True; continue
            if not in_site or not cols:
                continue
            if raw.startswith("_"):
                if rows: break
                continue
            parts = shlex.split(raw, posix=False)
            if len(parts) == len(cols):
                rows.append(dict(zip(cols, parts)))
            elif rows:
                break
    return rows


def norm_el(e):
    e = e.strip()
    return e[0].upper() + e[1:].lower() if e else e


def receptor_pdb_lines(atom_rows):
    lines, serial = [], 1
    for row in atom_rows:
        if row["group_PDB"] != "ATOM":
            continue
        el = norm_el(row["type_symbol"])
        if el == "H":
            continue
        nm = row["label_atom_id"].strip()[:4]
        nm = f" {nm:<3}" if len(el) == 1 and len(nm) < 4 else f"{nm:<4}"
        resname = row["label_comp_id"][:3].upper()
        chain = (row.get("auth_asym_id") or row.get("label_asym_id") or "A")[:1]
        try:
            resseq = int(float(row.get("auth_seq_id") or row.get("label_seq_id") or serial))
        except ValueError:
            resseq = serial
        x, y, z = float(row["Cartn_x"]), float(row["Cartn_y"]), float(row["Cartn_z"])
        b = float(row.get("B_iso_or_equiv", 50.0))
        ln = (f"ATOM  {serial:5d} {nm} {resname:>3} {chain}{resseq:4d}    "
              f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{b:6.2f}          {el:>2}")
        lines.append(ln[:80].ljust(80)); serial += 1
    return lines


def build_ligand_mol(smiles, atoms):
    template = Chem.RemoveHs(Chem.MolFromSmiles(smiles))
    n = template.GetNumAtoms()
    if len(atoms) != n:
        raise ValueError(f"원자 수 불일치 SMILES {n} vs cif {len(atoms)}")
    same = all(norm_el(template.GetAtomWithIdx(i).GetSymbol()) == norm_el(el)
               for i, (el, _) in enumerate(atoms))
    if same:
        m = Chem.Mol(template); conf = Chem.Conformer(n)
        for i, (_, xyz) in enumerate(atoms):
            conf.SetAtomPosition(i, Point3D(*xyz))
        m.RemoveAllConformers(); m.AddConformer(conf, assignId=True)
        return m, "order-match"
    from rdkit.Chem import rdDetermineBonds
    rw = Chem.RWMol(); conf = Chem.Conformer(n)
    for i, (el, xyz) in enumerate(atoms):
        rw.AddAtom(Chem.Atom(norm_el(el))); conf.SetAtomPosition(i, Point3D(*xyz))
    m = rw.GetMol(); m.AddConformer(conf, assignId=True)
    rdDetermineBonds.DetermineConnectivity(m)
    return AllChem.AssignBondOrdersFromTemplate(template, m), "template-match"


def mol_block(mol):
    lines = [l.rstrip() for l in Chem.MolToMolBlock(mol, forceV3000=False).splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def lscore_of(row):
    for k in ("ranking_score", "lscore", "iptm"):
        try:
            return f"{max(0.0, min(1.0, float(row.get(k, '')))):.3f}"
        except (TypeError, ValueError):
            continue
    return "0.500"


def structure_of(row):
    if row.get("structure"):
        return row["structure"]
    p = row.get("source_cif", "")
    try:
        q = p.split("/"); return f"{q[-4]}/{q[-3]}/{q[-2]}"
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ligand-tsv", default="")
    ap.add_argument("--target", default="TARGET")
    ap.add_argument("--author", default=os.environ.get("CASP_AUTHOR", "CHANGE-ME"))
    args = ap.parse_args()

    smiles_list = cio.read_ligand_tsv(args.ligand_tsv) if args.ligand_tsv and os.path.exists(args.ligand_tsv) else []
    if not smiles_list:
        raise SystemExit("ligand.tsv 없음/비어있음 — --ligand-tsv 필요")
    print(f"[ligand] {len(smiles_list)}종 SMILES 로드")

    os.makedirs(args.out_dir, exist_ok=True)
    rows = list(csv.DictReader(open(args.summary)))
    header = [
        "PFRMAT LG", f"TARGET {args.target}", f"AUTHOR {args.author}",
        "METHOD Consensus pose selection from team cofolding (multi-model, multi-seed).",
        "METHOD Ranked by interface confidence + consensus clustering + p2rank + gnina.",
        "METHOD Receptor from predicted mmCIF; ligand connectivity from SMILES, coords from prediction.",
    ]

    combined = []
    for rank, row in enumerate(rows, 1):
        cif = row.get("source_cif")
        if not cif:
            print(f"[skip] model {rank}: source_cif 없음"); continue
        atom_rows = parse_atom_site(Path(cif))
        rec = receptor_pdb_lines(atom_rows)
        _, ligands = cio.parse_complex(cif)        # 범용 리간드 검출 (개수 무관)

        body = header[:] if rank == 1 else []
        body += [f"MODEL {rank}",
                 f"REMARK structure {structure_of(row)}  iptm {row.get('iptm','')}  "
                 f"gnina {row.get('gnina_affinity','')}  n_ligands {len(ligands)}",
                 "PARENT N/A"]
        body += rec
        body.append("TER")
        body.append(f"LSCORE {lscore_of(row)}")
        for lg in ligands:                          # 리간드 copy마다 블록 출력
            match = cio.match_ligand_to_smiles(lg, smiles_list)
            if match is None:
                print(f"  [warn] model{rank} {lg['resname']}: SMILES 조성 매칭 실패 — 건너뜀")
                continue
            sid, name, smi = match
            mol, how = build_ligand_mol(smi, lg["atoms"])
            body.append(f"LIGAND {sid or lg['resname']} {name or lg['resname']}")
            body += mol_block(mol).splitlines()
            print(f"  model{rank} 리간드 {lg['resname']} ({len(lg['atoms'])}원자) -> {sid or name} [{how}]")
        body.append("END")
        text = "\n".join(body) + "\n"
        Path(args.out_dir, f"{args.target}LG_model{rank}.txt").write_text(text)
        combined.append(text)

    Path(args.out_dir, f"{args.target}LG_all_models.txt").write_text("".join(combined))
    print(f"\n저장 -> {args.out_dir} (model1~{len(combined)} + all_models)")


if __name__ == "__main__":
    main()

# ── 단독 실행 ── (먼저: source $CASP17/scripts/env_setup.sh)
#   SC=$CASP17/users/hosungkim/scripts ; OUT=$CASP17/users/hosungkim/targets/T2383
#   micromamba run -n boltz2 python $SC/9_make_casp_lg.py \
#       --summary $OUT/05_final/selection_summary.csv --out-dir $OUT/08_casp_lg \
#       --ligand-tsv <ligand_tsv> --target T2383 --author <CASP_CODE>
# (평소엔 run_pipeline.py 가 순서대로 자동 호출)
