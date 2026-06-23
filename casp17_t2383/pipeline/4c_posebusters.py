#!/usr/bin/env python3
"""
4c_posebusters.py (보강 ① 검사단계) - PoseBusters 유효성 검사. casp_eval env (posebusters).
4b가 만든 manifest의 (리간드 SDF + 단백질 PDB)에 PoseBusters("dock", 정답 불필요)를 돌려
분자내 기하·클래시 + 단백질-리간드 클래시 검사 → valid/실패검사 기록.
gemmi 불필요(파일만 읽음). final_select 가 이 결과로 무효 포즈를 제외.
출력: 04b_posebusters/posebusters.csv
"""
import argparse, csv, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="4b가 만든 manifest.csv")
    ap.add_argument("--out", required=True, help="posebusters.csv")
    args = ap.parse_args()

    try:
        from posebusters import PoseBusters
    except Exception as e:
        raise SystemExit(f"posebusters import 실패({e}) — posebusters 있는 env(python_pb)로 실행.")
    buster = PoseBusters(config="dock")

    rows = list(csv.DictReader(open(args.manifest)))
    out = []
    for r in rows:
        base = {"pocket_id": r["pocket_id"], "ligand_cluster_id": r["ligand_cluster_id"],
                "rep_cif": r["rep_cif"]}
        if not r.get("sdf") or not r.get("protein_pdb") or not os.path.exists(r["sdf"]):
            out.append({**base, "valid": "NA", "failed": r.get("note", "no_input")}); continue
        try:
            df = buster.bust([r["sdf"]], None, r["protein_pdb"], full_report=False)
            checks = [c for c in df.columns if str(df[c].dtype) == "bool"]
            failed = [c for c in checks if not bool(df.iloc[0][c])]
            valid = (len(failed) == 0)
        except Exception as e:
            out.append({**base, "valid": "ERR", "failed": str(e)[:80]}); continue
        out.append({**base, "valid": valid, "failed": ";".join(failed)})
        print(f"  P{r['pocket_id']}.L{r['ligand_cluster_id']}: valid={valid} 실패={failed if failed else '-'}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pocket_id", "ligand_cluster_id", "rep_cif", "valid", "failed"])
        w.writeheader(); w.writerows(out)
    nval = sum(1 for r in out if r["valid"] is True)
    print(f"[posebusters] {nval}/{len(out)} valid -> {args.out}")


if __name__ == "__main__":
    main()

# ── 단독 실행 ── (casp_eval env python; posebusters)
#   PB=/gpfs/deepfold/casp/casp17-ligand/.micromamba/envs/casp_eval/bin/python
#   $PB $SC/4c_posebusters.py --manifest $OUT/04b_posebusters/manifest.csv \
#       --out $OUT/04b_posebusters/posebusters.csv
