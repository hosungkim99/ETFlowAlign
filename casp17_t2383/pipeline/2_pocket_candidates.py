#!/usr/bin/env python3
"""
pocket_candidates.py (Step 1) - 리간드 센트로이드 클러스터링 → 포켓 후보 (상위 N개).
전체 모델의 예측 리간드 위치(무게중심)를 단백질 정렬 좌표계에서 coarse 클러스터링.
작은 클러스터도 유효 사이트일 수 있으므로 top N(기본 10) 유지. boltz2 env (numpy+gemmi).
출력: 02_pocket_candidates/ {pocket_candidates.csv, members.csv}.
"""
import argparse, csv, os
import numpy as np
import gemmi
import complex_io as cio
import geom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True, help="02_pocket_candidates 폴더")
    ap.add_argument("--protein-clusters", default="", help="01의 protein_clusters.csv(선택; conf 표기)")
    ap.add_argument("--threshold", type=float, default=8.0, help="포켓(센트로이드) 클러스터 임계 Å")
    ap.add_argument("--topn", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    conf = {}
    if args.protein_clusters and os.path.exists(args.protein_clusters):
        for r in csv.DictReader(open(args.protein_clusters)):
            conf[r["cif"]] = r.get("conf_id", "")

    rows = [r for r in csv.DictReader(open(args.table)) if r.get("cif")]
    rows.sort(key=lambda r: int(r["rank"]))
    ref_ca, ref_ligs = cio.parse_complex(rows[0]["cif"])
    ref_elems, _ = cio.ligand_concat(ref_ligs)

    data = []  # (row, centroid[np3])
    for i, r in enumerate(rows):
        ca, ligs = cio.parse_complex(r["cif"])
        elems, coords = cio.ligand_concat(ligs)
        if not ca or elems != ref_elems:
            continue
        rt = geom.align_to_ref(ref_ca, ca)
        if rt is None:
            continue
        L = geom.apply_rt(rt[0], rt[1], coords)
        data.append((r, L.mean(0)))
        if (i + 1) % 500 == 0:
            print(f"  parsed {i+1}/{len(rows)}")
    print(f"[pocket] {len(data)} usable structures")

    # 센트로이드 그리디 클러스터
    clusters = []
    for r, c in data:
        for cl in clusters:
            if np.linalg.norm(c - cl["center"]) <= args.threshold:
                cl["members"].append((r, c)); break
        else:
            clusters.append({"center": c, "members": [(r, c)]})
    # center 갱신 + 통계
    for cl in clusters:
        cs = np.array([c for _, c in cl["members"]])
        cl["center"] = cs.mean(0)
    clusters.sort(key=lambda cl: len(cl["members"]), reverse=True)
    clusters = clusters[:args.topn]

    member_rows = []
    cand_rows = []
    for pid, cl in enumerate(clusters, 1):
        mems = cl["members"]
        cs = np.array([c for _, c in mems])
        # medoid 구조(센트로이드 평균에 가장 가까운)
        medoid = mems[int(np.argmin(np.linalg.norm(cs - cl["center"], axis=1)))][0]
        models, confs = {}, {}
        for r, _ in mems:
            models[r["model"]] = models.get(r["model"], 0) + 1
            cid = conf.get(r["cif"], "")
            if cid != "":
                confs[cid] = confs.get(cid, 0) + 1
            member_rows.append({"cif": r["cif"], "pocket_id": pid,
                                "model": r["model"], "iptm": r.get("iptm", "")})
        cand_rows.append({
            "pocket_id": pid, "size": len(mems), "n_models": len(models),
            "models": ";".join(f"{k}:{v}" for k, v in sorted(models.items())),
            "conf_composition": ";".join(f"{k}:{v}" for k, v in sorted(confs.items())) or "NA",
            "center_x": f"{cl['center'][0]:.2f}", "center_y": f"{cl['center'][1]:.2f}",
            "center_z": f"{cl['center'][2]:.2f}",
            "rep_cif": medoid["cif"], "rep_model": medoid["model"],
            "rep_iptm": medoid.get("iptm", "")})

    with open(os.path.join(args.out, "pocket_candidates.csv"), "w", newline="") as f:
        cols = ["pocket_id", "size", "n_models", "models", "conf_composition",
                "center_x", "center_y", "center_z", "rep_cif", "rep_model", "rep_iptm"]
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(cand_rows)
    with open(os.path.join(args.out, "members.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cif", "pocket_id", "model", "iptm"])
        w.writeheader(); w.writerows(member_rows)

    print(f"[pocket] {len(clusters)} pocket candidates (top {args.topn}, threshold {args.threshold}Å)")
    for c in cand_rows:
        print(f"  P{c['pocket_id']}: size {c['size']}, #mdl {c['n_models']}, {c['models']}, conf[{c['conf_composition']}]")
    print(f"  -> {args.out}/pocket_candidates.csv (+ members.csv)")


if __name__ == "__main__":
    main()

# ── 단독 실행 ── (먼저: source $CASP17/scripts/env_setup.sh)
#   SC=$CASP17/users/hosungkim/scripts ; OUT=$CASP17/users/hosungkim/targets/T2383
#   micromamba run -n boltz2 python $SC/2_pocket_candidates.py \
#       --table $OUT/00_collect/master_table.csv --out $OUT/02_pocket_candidates \
#       --protein-clusters $OUT/01_protein_clusters/protein_clusters.csv --threshold 8.0 --topn 10
# (평소엔 run_pipeline.py 가 순서대로 자동 호출)
