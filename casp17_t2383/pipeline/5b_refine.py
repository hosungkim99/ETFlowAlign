#!/usr/bin/env python3
"""
5b_refine.py (보강 ②) - 최종 포즈를 포켓 내 gnina 에너지 최소화로 정제. boltz2 env + gnina.
05_final/model_*.cif 각각에 대해:
  단백질 PDB + 리간드 SDF(SMILES connectivity+좌표) 생성 → gnina --minimize 로 국소 최소화 →
  정제 리간드 좌표를 cif에 덮어써서 05b_refined/model_*.cif 작성.
안전장치: 정제 후 원본 대비 드리프트(RMSD) > --max-drift 면 원본 유지(엉뚱하게 끌려가는 것 방지).
리포트: refine_summary.csv (affinity before/after, drift, used). casp_lg가 refined를 쓰도록 refined
selection_summary.csv 도 생성.
"""
import argparse, csv, os, glob, subprocess, math
import gemmi
import complex_io as cio

GNINA_SIF = os.environ.get("GNINA_SIF", "/gpfs/deepfold/casp/casp17-ligand/models/gnina/gnina.sif")


def build_ligand_mol(smiles, atoms):
    """SMILES connectivity + cif 좌표로 RDKit mol (원자 i = cif 원자 i)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Geometry import Point3D
    tpl = Chem.RemoveHs(Chem.MolFromSmiles(smiles)); n = tpl.GetNumAtoms()
    if len(atoms) != n:
        raise ValueError(f"원자수 {n} vs {len(atoms)}")
    same = all(tpl.GetAtomWithIdx(i).GetSymbol().capitalize() == el.capitalize()
               for i, (el, _) in enumerate(atoms))
    if same:
        m = Chem.Mol(tpl); conf = Chem.Conformer(n)
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
    return AllChem.AssignBondOrdersFromTemplate(tpl, m)


def drift_and_coords(input_mol, refined_sdf):
    """입력 대비 정제 mol의 (대칭/원자대응 고려 RMSD, 입력 원자순서로 정렬된 refined 좌표)."""
    from rdkit import Chem
    from rdkit.Chem import rdMolAlign
    ref = next(Chem.SDMolSupplier(refined_sdf, removeHs=True))
    inp = Chem.RemoveHs(Chem.Mol(input_mol))
    try:
        d = rdMolAlign.CalcRMS(ref, inp)        # 원자대응+대칭 고려, 정렬 없이 제자리 RMSD
    except Exception:
        d = 999.0
    conf = ref.GetConformer()
    match = ref.GetSubstructMatch(inp)          # match[i] = inp 원자 i 에 대응하는 ref 원자 index
    if len(match) == inp.GetNumAtoms():
        coords = [[conf.GetAtomPosition(j).x, conf.GetAtomPosition(j).y, conf.GetAtomPosition(j).z]
                  for j in match]
    else:
        coords = [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                  for i in range(ref.GetNumAtoms())]
    return d, coords


def gnina_minimize(prot, lig_sdf, out_sdf, iters=10):
    # --minimize_iters 작게 = SD 몇 스텝만 → 충돌만 완화, 위치 거의 유지(국소)
    cmd = ["singularity", "exec", "--bind", "/gpfs", GNINA_SIF, "gnina",
           "--receptor", prot, "--ligand", lig_sdf, "--minimize",
           "--minimize_iters", str(iters),
           "--autobox_ligand", lig_sdf, "--out", out_sdf]
    r = subprocess.run(cmd, capture_output=True, text=True)
    aff = None
    for ln in r.stdout.splitlines():
        if ln.strip().startswith("Affinity:"):
            try: aff = float(ln.split()[1])
            except (IndexError, ValueError): pass
    return aff, r.stderr


def write_refined_cif(orig, refined, out):
    """원본 cif 리간드 heavy atom 좌표를 canonical 순서로 refined 로 덮어써서 mmCIF 저장."""
    st = gemmi.read_structure(orig); m = st[0]
    lig = []
    for ch in m:
        for res in ch:
            t = gemmi.find_tabulated_residue(res.name)
            if (t and (t.is_amino_acid() or t.is_nucleic_acid())) or res.name in cio.WATER:
                continue
            lig.append((res.name, ch.name, res.seqid.num, res))
    lig.sort(key=lambda x: (x[0], x[1], x[2]))
    k = 0
    for *_, res in lig:
        for a in res:
            if a.element.name == "H":
                continue
            if k < len(refined):
                x, y, z = refined[k]; a.pos = gemmi.Position(x, y, z); k += 1
    st.make_mmcif_document().write_file(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-dir", required=True, help="05_final 폴더(model_*.cif)")
    ap.add_argument("--ligand-tsv", required=True)
    ap.add_argument("--out", required=True, help="05b_refined 폴더")
    ap.add_argument("--max-drift", type=float, default=2.0)
    ap.add_argument("--iters", type=int, default=10, help="gnina SD 스텝 수(작게=국소)")
    args = ap.parse_args()
    work = os.path.join(args.out, "work"); os.makedirs(work, exist_ok=True)

    smis = cio.read_ligand_tsv(args.ligand_tsv)
    summ_in = os.path.join(args.final_dir, "selection_summary.csv")
    sel = {r["model"]: r for r in csv.DictReader(open(summ_in))} if os.path.exists(summ_in) else {}

    rep = []
    for cif in sorted(glob.glob(os.path.join(args.final_dir, "model_*.cif"))):
        name = os.path.splitext(os.path.basename(cif))[0]
        _, ligs = cio.parse_complex(cif)
        if not ligs:
            continue
        match = cio.match_ligand_to_smiles(ligs[0], smis)
        out_cif = os.path.join(args.out, f"{name}.cif")
        rowbase = {"model": name, "aff_before": sel.get(name, {}).get("gnina_affinity", ""),
                   "aff_after": "", "drift": "", "used": ""}
        if match is None:
            import shutil; shutil.copy(cif, out_cif); rowbase["used"] = "orig(no_smiles)"; rep.append(rowbase); continue
        try:
            from rdkit import Chem
            mol = build_ligand_mol(match[2], ligs[0]["atoms"])
            sdf = os.path.join(work, f"{name}_lig.sdf")
            w = Chem.SDWriter(sdf); w.write(mol); w.close()
            prot = os.path.join(work, f"{name}_prot.pdb")
            st = gemmi.read_structure(cif); st.setup_entities()
            st.remove_ligands_and_waters(); st.remove_empty_chains(); st.write_pdb(prot)
            ref_sdf = os.path.join(work, f"{name}_refined.sdf")
            aff, err = gnina_minimize(prot, sdf, ref_sdf, args.iters)
            if not os.path.exists(ref_sdf):
                raise RuntimeError(f"gnina minimize 출력 없음: {err.strip()[-150:]}")
            d, refined = drift_and_coords(mol, ref_sdf)
            rowbase["aff_after"] = aff; rowbase["drift"] = round(d, 3)
            if d <= args.max_drift:
                write_refined_cif(cif, refined, out_cif); rowbase["used"] = "refined"
            else:
                import shutil; shutil.copy(cif, out_cif); rowbase["used"] = f"orig(drift>{args.max_drift})"
        except Exception as e:
            import shutil; shutil.copy(cif, out_cif); rowbase["used"] = f"orig(err:{str(e)[:50]})"
        print(f"  {name}: aff {rowbase['aff_before']}->{rowbase['aff_after']} drift {rowbase['drift']} [{rowbase['used']}]")
        rep.append(rowbase)

    with open(os.path.join(args.out, "refine_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "aff_before", "aff_after", "drift", "used"])
        w.writeheader(); w.writerows(rep)

    # refined selection_summary (source_cif → refined cif; gnina_affinity → after)
    if sel:
        rows = []
        for name, r in sel.items():
            rr = dict(r); rc = os.path.join(args.out, f"{name}.cif")
            if os.path.exists(rc):
                rr["source_cif"] = rc
            aff = next((x["aff_after"] for x in rep if x["model"] == name and x["aff_after"] != ""), "")
            if aff != "":
                rr["gnina_affinity"] = aff
            rows.append(rr)
        with open(os.path.join(args.out, "selection_summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"[refine] -> {args.out} (refined model_*.cif, refine_summary.csv, selection_summary.csv)")


if __name__ == "__main__":
    main()

# ── 단독 실행 ── (boltz2 env python; gnina)
#   $PY $SC/5b_refine.py --final-dir $OUT/05_final \
#       --ligand-tsv <ligand_tsv> --out $OUT/05b_refined --max-drift 2.0
