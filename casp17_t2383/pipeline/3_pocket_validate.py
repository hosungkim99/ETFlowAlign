#!/usr/bin/env python3
"""
pocket_validate.py (Step 2) - 포켓 후보 검증 및 압축.
각 포켓 후보 대표에 p2rank(캐비티 존재?) + gnina(에너지)를 적용해 비교·검증하고,
기준 통과(pass) 포켓만 남겨 압축. boltz2 env + p2rank + singularity gnina.
출력: 03_pocket_validation/ {pocket_validation.csv, p2rank/, rescore/}.

autodock은 포켓별 박스 도킹이 필요해 기본 미포함(옵션 훅). gnina affinity가 1차 에너지.
"""
import argparse, csv, os, subprocess
import numpy as np
import gemmi
import complex_io as cio

PRANK = os.environ.get("PRANK", "/gpfs/deepfold/casp/casp17-ligand/models/p2rank/p2rank_2.5.1/prank")
GNINA_SIF = os.environ.get("GNINA_SIF", "/gpfs/deepfold/casp/casp17-ligand/models/gnina/gnina.sif")


def protein_pdb_centroid(cif, pdb_out):
    _, ligs = cio.parse_complex(cif)
    cen = cio.ligand_centroid(ligs)
    st = gemmi.read_structure(cif); st.setup_entities()
    st.remove_ligands_and_waters(); st.remove_empty_chains(); st.write_pdb(pdb_out)
    return np.array(cen) if cen else None


def run_prank(pdb, outdir):
    os.makedirs(outdir, exist_ok=True)
    subprocess.run([PRANK, "predict", "-f", pdb, "-o", outdir],
                   check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for root, _, files in os.walk(outdir):
        for f in files:
            if f.endswith("predictions.csv"):
                return os.path.join(root, f)
    return None


def nearest_pocket(pcsv, cen):
    best = None
    with open(pcsv) as fh:
        rd = csv.reader(fh); hdr = [h.strip() for h in next(rd)]
        idx = {h: i for i, h in enumerate(hdr)}
        gi = lambda *ns: next((idx[n] for n in ns if n in idx), None)
        cx, cy, cz, rk = gi("center_x"), gi("center_y"), gi("center_z"), gi("rank")
        for k, row in enumerate(rd):
            if cx is None or len(row) <= cx:
                continue
            try:
                c = np.array([float(row[cx]), float(row[cy]), float(row[cz])])
            except ValueError:
                continue
            d = float(np.linalg.norm(cen - c))
            if best is None or d < best[0]:
                best = (d, row[rk].strip() if rk is not None else str(k + 1), k == 0)
    return best  # (dist, rank, is_top)


def gnina(prot, lig, gpu=True):
    cmd = ["singularity", "exec"] + (["--nv"] if gpu else [])
    cmd += ["--bind", "/gpfs", GNINA_SIF, "gnina", "--receptor", prot, "--ligand", lig,
            "--score_only", "--cnn_scoring", "rescore", "--out", "/dev/null"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    sc = {}
    for ln in r.stdout.splitlines():
        if ln.startswith("Affinity:"):     sc["aff"] = float(ln.split()[1])
        elif ln.startswith("CNNscore:"):    sc["cnn"] = float(ln.split()[1])
        elif ln.startswith("CNNaffinity:"): sc["cnnaff"] = float(ln.split()[1])
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="02의 pocket_candidates.csv")
    ap.add_argument("--out", required=True, help="03_pocket_validation 폴더")
    ap.add_argument("--gnina-cutoff", type=float, default=-4.0, help="pass: gnina affinity <= 이 값")
    ap.add_argument("--p2rank-dist", type=float, default=6.0, help="pass: 최근접 p2rank 포켓 거리 <= 이 값")
    args = ap.parse_args()
    work = args.out
    os.makedirs(os.path.join(work, "p2rank"), exist_ok=True)
    os.makedirs(os.path.join(work, "rescore"), exist_ok=True)

    cands = list(csv.DictReader(open(args.candidates)))
    results = []
    for c in cands:
        pid, cif = c["pocket_id"], c["rep_cif"]
        print(f"\n=== pocket {pid} (size {c['size']}, {c['models']}) ===")
        prot = os.path.join(work, "rescore", f"P{pid}_protein.pdb")
        lig = os.path.join(work, "rescore", f"P{pid}_ligand.pdb")
        cen = protein_pdb_centroid(cif, prot)
        if cen is None:
            print("  리간드 없음"); continue
        _, ligs = cio.parse_complex(cif)
        cio.write_ligands_pdb(ligs, lig)
        # p2rank
        pcsv = run_prank(prot, os.path.join(work, "p2rank", f"P{pid}"))
        np_ = nearest_pocket(pcsv, cen) if pcsv else None
        # gnina
        sc = gnina(prot, lig, True) or gnina(prot, lig, False)
        aff = sc.get("aff")
        p2d = np_[0] if np_ else None
        passed = (aff is not None and aff <= args.gnina_cutoff) and \
                 (p2d is not None and p2d <= args.p2rank_dist)
        print(f"  p2rank dist={p2d}  gnina aff={aff} cnn={sc.get('cnn')}  pass={passed}")
        results.append({
            "pocket_id": pid, "size": c["size"], "n_models": c["n_models"],
            "models": c["models"], "conf_composition": c.get("conf_composition", ""),
            "p2rank_dist": round(p2d, 2) if p2d is not None else "NA",
            "p2rank_rank": np_[1] if np_ else "NA",
            "gnina_affinity": aff, "cnn_score": sc.get("cnn"),
            "cnn_affinity": sc.get("cnnaff"), "rep_cif": cif, "pass": passed})

    cols = ["pocket_id", "size", "n_models", "models", "conf_composition",
            "p2rank_dist", "p2rank_rank", "gnina_affinity", "cnn_score",
            "cnn_affinity", "pass", "rep_cif"]
    with open(os.path.join(work, "pocket_validation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(results)
    npass = sum(1 for r in results if r["pass"])
    print(f"\n[pocket_validate] {npass}/{len(results)} 포켓 통과 -> {work}/pocket_validation.csv")
    if npass == 0:
        print("  [주의] 통과 0 — 기준(--gnina-cutoff/--p2rank-dist) 완화 또는 데이터 재검토 필요")


if __name__ == "__main__":
    main()

# ── 단독 실행 ── (먼저: source $CASP17/scripts/env_setup.sh)
#   SC=$CASP17/users/hosungkim/scripts ; OUT=$CASP17/users/hosungkim/targets/T2383
#   micromamba run -n boltz2 python $SC/3_pocket_validate.py \
#       --candidates $OUT/02_pocket_candidates/pocket_candidates.csv --out $OUT/03_pocket_validation \
#       --gnina-cutoff -4.0 --p2rank-dist 6.0
# (평소엔 run_pipeline.py 가 순서대로 자동 호출)
