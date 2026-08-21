#!/usr/bin/env python3
"""
cluster_poses.py (범용) - 합의 클러스터링. complex_io 사용 → 리간드 개수·체인 수 무관.

모든 단백질 체인 Cα로 정렬 후, 모든 리간드 heavy atom을 canonical 순서로 concat해
그 전체 좌표 RMSD로 그리디 클러스터링. 대표 = medoid(클러스터 평균에 가장 가까운 멤버).
boltz2 env (numpy + gemmi).
"""
import argparse, csv, os, sys
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
    ap.add_argument("--threshold", type=float, default=2.0)
    ap.add_argument("--max", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = [r for r in csv.DictReader(open(args.table)) if r.get("cif")]
    rows.sort(key=lambda r: int(r["rank"]))
    if args.max:
        rows = rows[:args.max]
    print(f"[cluster] {len(rows)} structures with cif")

    ref_ca, ref_elems, ref_L = load(rows[0]["cif"])
    if not ref_ca or ref_L.shape[0] == 0:
        sys.exit("ERROR: 참조 구조 파싱 실패(단백질 또는 리간드 없음)")
    print(f"[cluster] ref: {len(ref_ca)} CA, ligand total {len(ref_elems)} heavy atoms "
          f"(리간드 copy 포함 전체)")

    aligned, skip = [], 0
    for i, r in enumerate(rows):
        ca, elems, L = load(r["cif"])
        if not ca or elems != ref_elems:          # 리간드 조성/개수 다르면 제외
            skip += 1; continue
        common = [k for k in ref_ca if k in ca]
        if len(common) < 50:
            skip += 1; continue
        R, t = kabsch(np.array([ca[k] for k in common]),
                      np.array([ref_ca[k] for k in common]))
        aligned.append((r, (R @ L.T).T + t))
        if (i + 1) % 300 == 0:
            print(f"  parsed {i+1}/{len(rows)}")
    print(f"[cluster] {len(aligned)} aligned & usable (제외 {skip})")

    clusters = []
    for r, L in aligned:
        for c in clusters:
            if float(np.sqrt(((L - c["rep_xyz"]) ** 2).sum(1).mean())) <= args.threshold:
                c["members"].append((r, L)); break
        else:
            clusters.append({"rep_xyz": L, "members": [(r, L)]})

    summ = []
    for c in clusters:
        mems = c["members"]
        Ls = np.array([L for _, L in mems])
        mean_L = Ls.mean(0)
        medoid = mems[int(np.argmin(np.sqrt(((Ls - mean_L) ** 2).sum(2).mean(1))))][0]
        models = {}
        for r, _ in mems:
            models[r["model"]] = models.get(r["model"], 0) + 1
        summ.append({
            "size": len(mems), "n_models": len(models),
            "models": ";".join(f"{k}:{v}" for k, v in sorted(models.items())),
            "best_rank": medoid["rank"], "best_model": medoid["model"],
            "best_iptm": medoid["iptm"], "best_seed": medoid["seed"],
            "best_sample": medoid["sample_idx"], "rep_cif": medoid["cif"],
        })
    summ.sort(key=lambda s: (s["size"], s["n_models"], float(s["best_iptm"] or 0)), reverse=True)
    for i, s in enumerate(summ, 1):
        s["cluster_id"] = i

    cols = ["cluster_id", "size", "n_models", "models", "best_rank", "best_model",
            "best_iptm", "best_seed", "best_sample", "rep_cif"]
    with open(os.path.join(args.out, "clusters.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for s in summ:
            w.writerow({k: s[k] for k in cols})

    print(f"\n[cluster] {len(summ)} clusters (medoid 대표, threshold {args.threshold} Å)")
    for s in summ[:10]:
        print(f"  c{s['cluster_id']}: size {s['size']} #mdl {s['n_models']} "
              f"iptm {float(s['best_iptm'] or 0):.3f}  {s['models']}")
    print(f"clusters.csv -> {args.out}/clusters.csv")


if __name__ == "__main__":
    main()
