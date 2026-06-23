#!/usr/bin/env python3
"""
rank_poses.py (범용) - 팀 cofolding 결과 수집·랭킹. stdlib만. 체인 수 무관.

<results>/<model>/seed_*/sample_*/summary.json 를 모아 master_table.csv 작성.
interface 신뢰도 = chain_pair_iptm의 모든 off-diagonal 평균(체인 수/구성 무관).
없으면 global iptm으로 폴백. 출력은 --out(=users/hosungkim 내부)에만.
"""
import argparse, csv, glob, json, os, sys
from statistics import mean


def interface_iptm(cpi):
    """chain_pair_iptm의 모든 off-diagonal 평균 (단백질/리간드 체인 자동 무관)."""
    if not isinstance(cpi, dict):
        return None
    vals = []
    for a, row in cpi.items():
        if not isinstance(row, dict):
            continue
        for b, v in row.items():
            if a != b and isinstance(v, (int, float)):
                vals.append(v)
    return mean(vals) if vals else None


def _f(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else "NA"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--protein-chain", default=None,
                    help="지정 시 그 체인의 off-diagonal만; 미지정 시 전체 평균(범용)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows, per_model = [], {}
    for sj in glob.glob(os.path.join(args.results, "*", "seed_*", "sample_*", "summary.json")):
        try:
            d = json.load(open(sj))
        except Exception as e:
            sys.stderr.write(f"skip {sj}: {e}\n"); continue
        model = d.get("model_name") or sj.split(os.sep)[-4]
        cif = os.path.join(os.path.dirname(sj), "model.cif")
        cpi = d.get("chain_pair_iptm")
        if args.protein_chain and isinstance(cpi, dict) and isinstance(cpi.get(args.protein_chain), dict):
            r = cpi[args.protein_chain]
            vals = [v for k, v in r.items() if k != args.protein_chain and isinstance(v, (int, float))]
            lig_iptm = mean(vals) if vals else None
        else:
            lig_iptm = interface_iptm(cpi)
        rows.append({"model": model, "seed": d.get("seed"), "sample_idx": d.get("sample_idx"),
                     "iptm": d.get("iptm"), "ligand_iptm": lig_iptm, "ptm": d.get("ptm"),
                     "plddt": d.get("plddt"), "gpde": d.get("gpde"),
                     "has_clash": d.get("has_clash"),
                     "ranking_score_native": d.get("ranking_score"),
                     "cif": cif if os.path.exists(cif) else "", "summary": sj})
        per_model[model] = per_model.get(model, 0) + 1

    def key(r):
        li = r["ligand_iptm"] if r["ligand_iptm"] is not None else (r["iptm"] or 0)
        return (0 if r["has_clash"] else 1, li, r["plddt"] or 0,
                -(r["gpde"] if r["gpde"] is not None else 9))

    rows.sort(key=key, reverse=True)
    for i, r in enumerate(rows, 1):
        r["rank"] = i

    cols = ["rank", "model", "seed", "sample_idx", "ligand_iptm", "iptm", "ptm",
            "plddt", "gpde", "has_clash", "ranking_score_native", "cif", "summary"]
    out_csv = os.path.join(args.out, "master_table.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})
    with open(os.path.join(args.out, "progress.txt"), "w") as f:
        f.write(f"total samples: {len(rows)}\n")
        for m in sorted(per_model):
            f.write(f"  {m}: {per_model[m]}\n")

    print(f"[rank_poses] {len(rows)} samples -> {out_csv}")
    for m in sorted(per_model):
        print(f"  {m}: {per_model[m]}")


if __name__ == "__main__":
    main()

# ── 단독 실행 ── (먼저: source $CASP17/scripts/env_setup.sh)
#   SC=$CASP17/users/hosungkim/scripts ; OUT=$CASP17/users/hosungkim/targets/T2383
#   python3 $SC/0_rank_poses.py --results <results_dir> --out $OUT/00_collect
# (평소엔 run_pipeline.py 가 순서대로 자동 호출)
