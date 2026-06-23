#!/usr/bin/env python3
"""
ligand_cluster.py (Step 3) - 포켓별 리간드 재클러스터링 (SC-RMSD) → 대표 리간드 구조.
통과(pass)한 포켓 각각에서, 그 포켓 멤버 리간드를 포켓 대표에 정렬한 뒤 대칭보정 RMSD로
fine 클러스터링하여 포켓별 대표 리간드 후보들을 도출. boltz2 env (numpy+gemmi+rdkit).
출력: 04_ligand_clusters/ligand_clusters.csv.
"""
import argparse, csv, os
import numpy as np
import gemmi
import complex_io as cio
import geom


def load(path):
    ca, ligs = cio.parse_complex(path)
    elems, coords = cio.ligand_concat(ligs)
    return ca, elems, np.array(coords) if coords else np.zeros((0, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", required=True, help="02의 members.csv")
    ap.add_argument("--validation", required=True, help="03의 pocket_validation.csv")
    ap.add_argument("--candidates", required=True, help="02의 pocket_candidates.csv (rep_cif)")
    ap.add_argument("--table", required=True, help="00의 master_table.csv (iptm 참조)")
    ap.add_argument("--ligand-tsv", default="", help="SC-RMSD 대칭순열용 SMILES")
    ap.add_argument("--out", required=True, help="04_ligand_clusters 폴더")
    ap.add_argument("--threshold", type=float, default=2.0, help="리간드 클러스터 SC-RMSD 임계 Å")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    passed = {r["pocket_id"] for r in csv.DictReader(open(args.validation)) if r["pass"] in ("True", "1", "true")}
    if not passed:
        print("[ligand_cluster] 통과 포켓 없음 — 전 포켓 사용(폴백)")
        passed = {r["pocket_id"] for r in csv.DictReader(open(args.candidates))}
    rep_of = {r["pocket_id"]: r["rep_cif"] for r in csv.DictReader(open(args.candidates))}
    iptm_of = {r["cif"]: r.get("iptm", "") for r in csv.DictReader(open(args.table)) if r.get("cif")}

    by_pocket = {}
    for m in csv.DictReader(open(args.members)):
        if m["pocket_id"] in passed:
            by_pocket.setdefault(m["pocket_id"], []).append(m["cif"])

    # 대칭 순열 (SMILES 1종 가정; 여러 종이면 첫 행)
    perms = None
    smis = cio.read_ligand_tsv(args.ligand_tsv) if args.ligand_tsv and os.path.exists(args.ligand_tsv) else []

    out_rows = []
    for pid, cifs in by_pocket.items():
        ref_ca, ref_elems, ref_L = load(rep_of[pid])
        if ref_L.shape[0] == 0:
            continue
        if perms is None and smis:
            perms = geom.ligand_automorphisms(smis[0][2], ref_L.shape[0])
            print(f"[ligand_cluster] 대칭 순열 {len(perms)}개 (SC-RMSD)")
        aligned = []
        for cif in cifs:
            ca, elems, L = load(cif)
            if not ca or elems != ref_elems:
                continue
            rt = geom.align_to_ref(ref_ca, ca)
            if rt is None:
                continue
            aligned.append((cif, geom.apply_rt(rt[0], rt[1], L)))
        # SC-RMSD 그리디
        clusters = []
        for cif, L in aligned:
            for cl in clusters:
                if geom.sc_rmsd(L, cl["rep"], perms) <= args.threshold:
                    cl["members"].append((cif, L)); break
            else:
                clusters.append({"rep": L, "members": [(cif, L)]})
        clusters.sort(key=lambda cl: len(cl["members"]), reverse=True)
        for lid, cl in enumerate(clusters, 1):
            mems = cl["members"]
            Ls = np.array([L for _, L in mems])
            mean_L = Ls.mean(0)
            medoid = mems[int(np.argmin([geom.sc_rmsd(L, mean_L, perms) for _, L in mems]))][0]
            out_rows.append({"pocket_id": pid, "ligand_cluster_id": lid, "size": len(mems),
                             "rep_cif": medoid, "rep_iptm": iptm_of.get(medoid, "")})
        print(f"  pocket {pid}: {len(aligned)} ligands -> {len(clusters)} 방향 클러스터")

    cols = ["pocket_id", "ligand_cluster_id", "size", "rep_cif", "rep_iptm"]
    with open(os.path.join(args.out, "ligand_clusters.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(out_rows)
    print(f"[ligand_cluster] {len(out_rows)} (pocket,ligand) 대표 -> {args.out}/ligand_clusters.csv")


if __name__ == "__main__":
    main()

# ── 단독 실행 ── (먼저: source $CASP17/scripts/env_setup.sh)
#   SC=$CASP17/users/hosungkim/scripts ; OUT=$CASP17/users/hosungkim/targets/T2383
#   micromamba run -n boltz2 python $SC/4_ligand_cluster.py \
#       --members $OUT/02_pocket_candidates/members.csv \
#       --validation $OUT/03_pocket_validation/pocket_validation.csv \
#       --candidates $OUT/02_pocket_candidates/pocket_candidates.csv \
#       --table $OUT/00_collect/master_table.csv --ligand-tsv <ligand_tsv> \
#       --out $OUT/04_ligand_clusters --threshold 2.0
# (평소엔 run_pipeline.py 가 순서대로 자동 호출)
