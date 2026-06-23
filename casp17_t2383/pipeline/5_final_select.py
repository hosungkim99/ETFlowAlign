#!/usr/bin/env python3
"""
final_select.py (Step 4) - 포켓별 리간드 클러스터 후보들에 에너지(gnina)를 적용해 최종 ≤5개 선정.
종합점수 = 합의(클러스터 크기) + iptm + affinity 가중합. task에 따라 가중치 분기:
  - P : pose 중심(합의·iptm↑, affinity↓)
  - PA: affinity 비중↑
boltz2 env + singularity gnina. 출력: 05_final/ {selection_summary.csv, model_*.cif, SELECTION_RATIONALE.md}.
"""
import argparse, csv, os, shutil, subprocess
import gemmi
import complex_io as cio

GNINA_SIF = os.environ.get("GNINA_SIF", "/gpfs/deepfold/casp/casp17-ligand/models/gnina/gnina.sif")
WEIGHTS = {"P": dict(consensus=0.5, iptm=0.3, aff=0.2),
           "PA": dict(consensus=0.35, iptm=0.25, aff=0.4)}


def gnina_score(cif, work, tag):
    prot = os.path.join(work, f"{tag}_prot.pdb"); lig = os.path.join(work, f"{tag}_lig.pdb")
    _, ligs = cio.parse_complex(cif)
    if not ligs:
        return {}
    cio.write_ligands_pdb(ligs, lig)
    st = gemmi.read_structure(cif); st.setup_entities()
    st.remove_ligands_and_waters(); st.remove_empty_chains(); st.write_pdb(prot)
    for gpu in (True, False):
        cmd = ["singularity", "exec"] + (["--nv"] if gpu else [])
        cmd += ["--bind", "/gpfs", GNINA_SIF, "gnina", "--receptor", prot, "--ligand", lig,
                "--score_only", "--cnn_scoring", "rescore", "--out", "/dev/null"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        sc = {}
        for ln in r.stdout.splitlines():
            if ln.startswith("Affinity:"):     sc["aff"] = float(ln.split()[1])
            elif ln.startswith("CNNscore:"):    sc["cnn"] = float(ln.split()[1])
            elif ln.startswith("CNNaffinity:"): sc["cnnaff"] = float(ln.split()[1])
        if sc:
            return sc
    return {}


def norm(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return lambda x: 0.5
    lo, hi = min(v), max(v)
    return (lambda x: 0.5) if hi == lo else (lambda x: (x - lo) / (hi - lo) if x is not None else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ligand-clusters", required=True)
    ap.add_argument("--validation", required=True)
    ap.add_argument("--out", required=True, help="05_final 폴더")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--task", default="P", choices=["P", "PA"])
    ap.add_argument("--posebusters", default="", help="04b posebusters.csv (있으면 무효 포즈 제외)")
    args = ap.parse_args()
    work = os.path.join(args.out, "score"); os.makedirs(work, exist_ok=True)

    cands = list(csv.DictReader(open(args.ligand_clusters)))
    pval = {r["pocket_id"]: r for r in csv.DictReader(open(args.validation))}
    if not cands:
        raise SystemExit("ligand_clusters.csv 비어있음")

    # ① PoseBusters 무효 포즈 제외 (파일 있을 때만; 'False'만 제외, NA/ERR은 보존)
    pbvalid = {}
    if args.posebusters and os.path.exists(args.posebusters):
        for r in csv.DictReader(open(args.posebusters)):
            pbvalid[(r["pocket_id"], r["ligand_cluster_id"])] = r.get("valid", "")
        kept = [c for c in cands
                if pbvalid.get((c["pocket_id"], c["ligand_cluster_id"]), "NA") != "False"]
        ndrop = len(cands) - len(kept)
        if kept:
            print(f"[final_select] PoseBusters 무효 {ndrop}개 제외 → {len(kept)} 후보")
            cands = kept
        else:
            print("[final_select] 경고: 전부 무효로 판정 → 필터 미적용(원본 유지)")

    # 각 후보 점수화
    for c in cands:
        sc = gnina_score(c["rep_cif"], work, f"P{c['pocket_id']}_L{c['ligand_cluster_id']}")
        c["gnina_affinity"] = sc.get("aff"); c["cnn_score"] = sc.get("cnn"); c["cnn_affinity"] = sc.get("cnnaff")
        try:
            c["_iptm"] = float(c.get("rep_iptm") or 0)
        except ValueError:
            c["_iptm"] = 0.0
        c["_size"] = int(c["size"])
        print(f"  P{c['pocket_id']}.L{c['ligand_cluster_id']} size {c['size']} "
              f"iptm {c['_iptm']:.3f} gnina {c['gnina_affinity']}")

    n_size = norm([c["_size"] for c in cands])
    n_iptm = norm([c["_iptm"] for c in cands])
    # affinity는 낮을수록 좋음 → 부호 반전
    n_aff = norm([-c["gnina_affinity"] if c["gnina_affinity"] is not None else None for c in cands])
    w = WEIGHTS[args.task]
    for c in cands:
        c["composite"] = round(
            w["consensus"] * n_size(c["_size"]) + w["iptm"] * n_iptm(c["_iptm"]) +
            w["aff"] * n_aff(-c["gnina_affinity"] if c["gnina_affinity"] is not None else None), 4)

    cands.sort(key=lambda c: c["composite"], reverse=True)
    sel = cands[:args.top]

    rows = []
    for i, c in enumerate(sel, 1):
        dst = os.path.join(args.out, f"model_{i}.cif")
        shutil.copy(c["rep_cif"], dst)
        rows.append({"model": f"model_{i}", "pocket_id": c["pocket_id"],
                     "ligand_cluster_id": c["ligand_cluster_id"], "size": c["size"],
                     "iptm": c.get("rep_iptm", ""), "lscore": c.get("rep_iptm", ""),
                     "gnina_affinity": c["gnina_affinity"], "cnn_score": c["cnn_score"],
                     "composite": c["composite"], "pocket_pass": pval.get(c["pocket_id"], {}).get("pass", ""),
                     "posebusters_valid": pbvalid.get((c["pocket_id"], c["ligand_cluster_id"]), "NA"),
                     "source_cif": c["rep_cif"]})

    cols = ["model", "pocket_id", "ligand_cluster_id", "size", "iptm", "lscore",
            "gnina_affinity", "cnn_score", "composite", "pocket_pass",
            "posebusters_valid", "source_cif"]
    with open(os.path.join(args.out, "selection_summary.csv"), "w", newline="") as f:
        w2 = csv.DictWriter(f, fieldnames=cols); w2.writeheader(); w2.writerows(rows)

    md = [f"# 최종 선정 근거 (Task={args.task})\n",
          f"종합점수 가중치: consensus {w['consensus']}, iptm {w['iptm']}, affinity {w['aff']}\n",
          "(P=pose 중심, PA=affinity 비중↑)\n\n",
          "| model | pocket | lig_cluster | size | iptm | gnina | composite |\n",
          "|---|---|---|---|---|---|---|\n"]
    for r in rows:
        md.append(f"| {r['model']} | P{r['pocket_id']} | L{r['ligand_cluster_id']} | {r['size']} "
                  f"| {r['iptm']} | {r['gnina_affinity']} | {r['composite']} |\n")
    md.append("\n선정 원칙: 포켓 검증 통과 + 합의(클러스터 크기) + interface 신뢰도 + 에너지 종합.\n")
    md.append("주의: 같은 포켓에서 여러 개가 뽑히면 방향 다양성 확인, 다른 포켓이 섞이면 hedge.\n")
    open(os.path.join(args.out, "SELECTION_RATIONALE.md"), "w").write("".join(md))

    print(f"\n[final_select] 최종 {len(sel)}개 -> {args.out}/ (selection_summary.csv, model_*.cif, RATIONALE.md)")
    for r in rows:
        print(f"  {r['model']}: P{r['pocket_id']}.L{r['ligand_cluster_id']} composite {r['composite']}")


if __name__ == "__main__":
    main()

# ── 단독 실행 ── (먼저: source $CASP17/scripts/env_setup.sh)
#   SC=$CASP17/users/hosungkim/scripts ; OUT=$CASP17/users/hosungkim/targets/T2383
#   micromamba run -n boltz2 python $SC/5_final_select.py \
#       --ligand-clusters $OUT/04_ligand_clusters/ligand_clusters.csv \
#       --validation $OUT/03_pocket_validation/pocket_validation.csv \
#       --out $OUT/05_final --top 5 --task P        # PA면 --task PA
# (평소엔 run_pipeline.py 가 순서대로 자동 호출)
