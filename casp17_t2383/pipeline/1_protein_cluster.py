#!/usr/bin/env python3
"""
protein_cluster.py (Step 0) - 단백질 구조 클러스터링 → 대표 conformation.
모델에 따라 다른 단백질 형태(예: ConfA/B)가 나오므로, Cα를 정렬해 PCA→그리디 클러스터로
형태를 라벨링하고 대표(medoid)를 저장. 이후 스텝이 conf를 인지/선택할 수 있게 함.
boltz2 env (numpy+gemmi). 출력: 01_protein_clusters/.
"""
import argparse, csv, os, shutil
import numpy as np
import gemmi
import complex_io as cio
import geom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True, help="01_protein_clusters 폴더")
    ap.add_argument("--threshold", type=float, default=15.0, help="PC공간 그리디 임계(형태 분리)")
    ap.add_argument("--max", type=int, default=0, help="속도 위해 상위 N개만(0=전체)")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "reps"), exist_ok=True)

    rows = [r for r in csv.DictReader(open(args.table)) if r.get("cif")]
    rows.sort(key=lambda r: int(r["rank"]))
    if args.max:
        rows = rows[:args.max]

    ref_ca, _ = cio.parse_complex(rows[0]["cif"])
    ref_keys = sorted(ref_ca)

    feats, used = [], []
    for i, r in enumerate(rows):
        ca, _ = cio.parse_complex(r["cif"])
        rt = geom.align_to_ref(ref_ca, ca)
        if rt is None:
            continue
        R, t = rt
        # ref_keys 위치의 정렬된 Cα (없으면 제외 표시)
        vec, ok = [], True
        for k in ref_keys:
            if k in ca:
                vec.append(geom.apply_rt(R, t, [ca[k]])[0])
            else:
                ok = False; break
        if not ok:
            continue
        feats.append(np.array(vec).flatten())
        used.append(r)
        if (i + 1) % 500 == 0:
            print(f"  parsed {i+1}/{len(rows)}")

    X = np.array(feats)
    print(f"[protein_cluster] {len(X)} structures, feature dim {X.shape[1] if len(X) else 0}")
    if len(X) < 2:
        open(os.path.join(args.out, "protein_clusters.csv"), "w").write("note,insufficient\n")
        return

    Xc = X - X.mean(0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    PC = Xc @ Vt[:min(3, Vt.shape[0])].T
    var = (S ** 2) / (S ** 2).sum() * 100

    # PC공간 그리디 클러스터
    reps, labels = [], []
    for p in PC:
        for ci, rp in enumerate(reps):
            if np.linalg.norm(p - rp) <= args.threshold:
                labels.append(ci); break
        else:
            reps.append(p); labels.append(len(reps) - 1)
    labels = np.array(labels)
    sizes = np.bincount(labels)
    order = np.argsort(sizes)[::-1]
    relabel = {old: new for new, old in enumerate(order)}

    # 대표(medoid) 저장
    rep_cif = {}
    for new, old in enumerate(order):
        idx = np.where(labels == old)[0]
        sub = PC[idx]
        medoid_local = idx[int(np.argmin(np.linalg.norm(sub - sub.mean(0), axis=1)))]
        src = used[medoid_local]["cif"]
        dst = os.path.join(args.out, "reps", f"conf_{new}.cif")
        try:
            shutil.copy(src, dst)
        except Exception:
            pass
        rep_cif[new] = src

    with open(os.path.join(args.out, "protein_clusters.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cif", "model", "seed", "sample", "conf_id", "pc1", "pc2", "pc3"])
        for r, lab, p in zip(used, labels, PC):
            pc = list(p) + [0, 0, 0]
            w.writerow([r["cif"], r["model"], r["seed"], r.get("sample_idx", ""),
                        relabel[lab], f"{pc[0]:.2f}", f"{pc[1]:.2f}", f"{pc[2]:.2f}"])

    print(f"[protein_cluster] {len(reps)} conformations (PC1 {var[0]:.1f}%, PC2 {var[1]:.1f}%)")
    for new, old in enumerate(order):
        print(f"  conf_{new}: {sizes[old]} structures, rep={os.path.basename(rep_cif[new])}")
    print(f"  -> {args.out}/protein_clusters.csv  (+ reps/)")


if __name__ == "__main__":
    main()

# ── 단독 실행 ── (먼저: source $CASP17/scripts/env_setup.sh)
#   SC=$CASP17/users/hosungkim/scripts ; OUT=$CASP17/users/hosungkim/targets/T2383
#   micromamba run -n boltz2 python $SC/1_protein_cluster.py \
#       --table $OUT/00_collect/master_table.csv --out $OUT/01_protein_clusters --threshold 15.0
# (평소엔 run_pipeline.py 가 순서대로 자동 호출)
