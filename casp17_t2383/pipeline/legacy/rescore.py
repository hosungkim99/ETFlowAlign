#!/usr/bin/env python3
"""
rescore.py (범용) - gnina rescoring. complex_io 사용 → 리간드 개수 무관.

clusters.csv 상위 N개 대표 cif를:
  단백질 PDB(gemmi) + 모든 리간드 한 PDB(complex_io.write_ligands_pdb)로 분리 →
  gnina --score_only 로 affinity/CNN 산출. boltz2 env + singularity gnina.
"""
import argparse, csv, os, subprocess
import gemmi
import complex_io as cio

GNINA_SIF = os.environ.get("GNINA_SIF",
                           "/gpfs/deepfold/casp/casp17-ligand/models/gnina/gnina.sif")


def split(cif, prot_pdb, lig_pdb):
    st = gemmi.read_structure(cif); st.setup_entities()
    _, ligs = cio.parse_complex(cif)
    if not ligs:
        return False
    cio.write_ligands_pdb(ligs, lig_pdb)
    st.remove_ligands_and_waters(); st.remove_empty_chains(); st.write_pdb(prot_pdb)
    return True


def gnina(prot, lig, use_gpu=True):
    cmd = ["singularity", "exec"] + (["--nv"] if use_gpu else [])
    cmd += ["--bind", "/gpfs", GNINA_SIF, "gnina", "--receptor", prot, "--ligand", lig,
            "--score_only", "--cnn_scoring", "rescore", "--out", "/dev/null"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    sc = {}
    for ln in r.stdout.splitlines():
        if ln.startswith("Affinity:"):     sc["affinity"] = float(ln.split()[1])
        elif ln.startswith("CNNscore:"):    sc["cnn_score"] = float(ln.split()[1])
        elif ln.startswith("CNNaffinity:"): sc["cnn_affinity"] = float(ln.split()[1])
    return sc, r.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)

    reps = list(csv.DictReader(open(args.clusters)))[:args.top]
    results = []
    for r in reps:
        cid, cif = r["cluster_id"], r["rep_cif"]
        prot = os.path.join(args.work, f"c{cid}_protein.pdb")
        lig = os.path.join(args.work, f"c{cid}_ligand.pdb")
        print(f"\n=== cluster {cid} (size {r['size']}, {r['models']}) ===")
        if not split(cif, prot, lig):
            print("  리간드 없음, 건너뜀"); continue
        sc, err = gnina(prot, lig, use_gpu=True)
        if not sc:
            sc, err = gnina(prot, lig, use_gpu=False)
        if not sc:
            print(f"  gnina 실패: {err.strip()[-200:]}"); continue
        print(f"  affinity={sc.get('affinity')} CNNscore={sc.get('cnn_score')} "
              f"CNNaffinity={sc.get('cnn_affinity')}")
        results.append({"cluster_id": cid, "size": r["size"], "models": r["models"],
                        "best_iptm": r["best_iptm"],
                        "gnina_affinity": sc.get("affinity"),
                        "cnn_score": sc.get("cnn_score"),
                        "cnn_affinity": sc.get("cnn_affinity")})

    cols = ["cluster_id", "size", "models", "best_iptm",
            "gnina_affinity", "cnn_score", "cnn_affinity"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(results)
    print(f"\nrescore.csv -> {args.out}")


if __name__ == "__main__":
    main()
