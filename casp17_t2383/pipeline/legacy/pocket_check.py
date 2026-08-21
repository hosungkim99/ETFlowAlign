#!/usr/bin/env python3
"""
pocket_check.py (범용) - p2rank 포켓 검증. complex_io 사용 → 리간드 개수 무관.

clusters.csv 상위 N개 대표의 단백질에 p2rank 실행, 모든 리간드 무게중심에 가장 가까운
포켓을 보고. boltz2 env + p2rank(prank).
"""
import argparse, csv, os, subprocess
import numpy as np
import gemmi
import complex_io as cio

PRANK = os.environ.get("PRANK",
                       "/gpfs/deepfold/casp/casp17-ligand/models/p2rank/p2rank_2.5.1/prank")


def protein_pdb_and_centroid(cif, pdb_out):
    """단백질만 PDB로 저장 + 모든 리간드 무게중심 반환(None 가능)."""
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


def parse_pockets(csv_path):
    pockets = []
    with open(csv_path) as fh:
        rd = csv.reader(fh)
        header = [h.strip() for h in next(rd)]
        idx = {h: i for i, h in enumerate(header)}
        gi = lambda *ns: next((idx[n] for n in ns if n in idx), None)
        cx, cy, cz = gi("center_x"), gi("center_y"), gi("center_z")
        sc, rk, res = gi("score"), gi("rank"), gi("residue_ids")
        for row in rd:
            if not row or cx is None or len(row) <= cx:
                continue
            try:
                center = np.array([float(row[cx]), float(row[cy]), float(row[cz])])
            except (ValueError, IndexError):
                continue
            pockets.append({"rank": row[rk].strip() if rk is not None else "",
                            "score": float(row[sc]) if sc is not None and row[sc].strip() else None,
                            "center": center,
                            "residues": row[res].strip() if res is not None else ""})
    return pockets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)

    reps = list(csv.DictReader(open(args.clusters)))[:args.top]
    results = []
    for r in reps:
        cid, cif = r["cluster_id"], r["rep_cif"]
        pdb = os.path.join(args.work, f"c{cid}_protein.pdb")
        outdir = os.path.join(args.work, f"c{cid}_p2rank")
        print(f"\n=== cluster {cid} (size {r['size']}, {r['models']}) ===")
        cen = protein_pdb_and_centroid(cif, pdb)
        if cen is None:
            print("  리간드 없음, 건너뜀"); continue
        pcsv = run_prank(pdb, outdir)
        if not pcsv:
            print("  p2rank predictions.csv 없음"); continue
        pockets = parse_pockets(pcsv)
        if not pockets:
            print("  예측 포켓 0개"); continue
        dists = [float(np.linalg.norm(cen - p["center"])) for p in pockets]
        j = int(np.argmin(dists)); near = pockets[j]
        in_top = (j == 0)
        print(f"  포켓 {len(pockets)}개; 최근접 rank={near['rank']} dist={dists[j]:.2f}Å (1순위={in_top})")
        results.append({"cluster_id": cid, "size": r["size"], "n_pockets": len(pockets),
                        "nearest_pocket_rank": near["rank"], "pocket_score": near["score"],
                        "centroid_dist": round(dists[j], 2), "in_top_pocket": in_top,
                        "pocket_residues": near["residues"]})

    cols = ["cluster_id", "size", "n_pockets", "nearest_pocket_rank", "pocket_score",
            "centroid_dist", "in_top_pocket", "pocket_residues"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(results)
    print(f"\npocket_check.csv -> {args.out}")


if __name__ == "__main__":
    main()
