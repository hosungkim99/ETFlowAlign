#!/usr/bin/env python3
"""
pose_diversity.py (범용) - 포켓 내 결합 방향 다양성. complex_io 사용 → 리간드 개수 무관.

master_table.csv 상위 K개를 모든 단백질 Cα로 정렬한 뒤, 모든 리간드 atom(concat)
좌표로 pairwise RMSD를 재서 방향 수렴/분산을 정량화. boltz2 env (numpy + gemmi).
"""
import argparse, csv, os
import numpy as np
import complex_io as cio


def kabsch(P, Q):
    Pc, Qc = P.mean(0), Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, Qc - R @ Pc


def load(path):
    ca, ligs = cio.parse_complex(path)
    elems, coords = cio.ligand_concat(ligs)
    return ca, elems, np.array(coords) if coords else np.zeros((0, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=1.5)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    rows = [r for r in csv.DictReader(open(args.table)) if r.get("cif")]
    rows.sort(key=lambda r: int(r["rank"]))
    rows = rows[:args.topk]
    if len(rows) < 2:
        open(args.out, "w").write("note,insufficient_structures\n"); return

    ref_ca, ref_elems, _ = load(rows[0]["cif"])
    ligs, ca_rmsd, keep = [], [], []
    for r in rows:
        ca, elems, L = load(r["cif"])
        tag = f"{r['model']}/s{r['seed']}"
        if not ca or elems != ref_elems:
            print(f"  skip {tag}: 리간드 조성 불일치"); continue
        common = [k for k in ref_ca if k in ca]
        if len(common) < 50:
            print(f"  skip {tag}: 공통 CA 부족"); continue
        R, t = kabsch(np.array([ca[k] for k in common]),
                      np.array([ref_ca[k] for k in common]))
        Pc = (R @ np.array([ca[k] for k in common]).T).T + t
        ca_rmsd.append(float(np.sqrt(((Pc - np.array([ref_ca[k] for k in common])) ** 2).sum(1).mean())))
        ligs.append((R @ L.T).T + t); keep.append(tag)

    n = len(ligs)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = float(np.sqrt(((ligs[i] - ligs[j]) ** 2).sum(1).mean()))
    vals = D[np.triu_indices(n, 1)]
    print(f"[pose_diversity] {n}개, 정렬 Cα RMSD 평균 {np.mean(ca_rmsd):.2f}Å; "
          f"리간드 pairwise 평균 {vals.mean():.2f} 최대 {vals.max():.2f} Å")

    unassigned, clusters = set(range(n)), []
    for c0 in sorted(range(n), key=lambda i: int((D[i] <= args.threshold).sum()), reverse=True):
        if c0 not in unassigned:
            continue
        members = [j for j in list(unassigned) if D[c0, j] <= args.threshold]
        for j in members:
            unassigned.discard(j)
        clusters.append((c0, members))
    print(f"  {args.threshold}Å 방향 클러스터: {len(clusters)}개 "
          f"=> {'수렴' if len(clusters)==1 else '분산'}")

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["topk", n, "mean_pairwise_A", f"{vals.mean():.3f}",
                    "max_pairwise_A", f"{vals.max():.3f}",
                    "orientation_clusters", len(clusters)])
        w.writerow([]); w.writerow(["tag"] + keep)
        for i in range(n):
            w.writerow([keep[i]] + [f"{D[i,j]:.3f}" for j in range(n)])
        w.writerow([])
        for k, (c0, mem) in enumerate(clusters, 1):
            w.writerow([f"orientation_{k}", "rep", keep[c0], "members",
                        ";".join(keep[j] for j in mem)])
    print(f"  pose_diversity.csv -> {args.out}")


if __name__ == "__main__":
    main()
