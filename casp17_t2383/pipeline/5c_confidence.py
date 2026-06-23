#!/usr/bin/env python3
"""
5c_confidence.py (보강 ③) - 선정 신뢰도 리포트. 기존 출력 CSV만 읽어 집계 (의존성 없음, stdlib).
"이번 자동선택을 얼마나 믿을 수 있나"를 신호별 수치 + HIGH/MEDIUM/LOW 판정으로 요약.
출력: 05c_confidence/{confidence_report.md, confidence.csv}
"""
import argparse, csv, os


def load(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def fnum(x, d=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", required=True, help="outputs 루트 폴더")
    ap.add_argument("--out", default="", help="리포트 폴더(기본 outputs/05c_confidence)")
    ap.add_argument("--p2rank-ok", type=float, default=6.0)
    ap.add_argument("--dominance-high", type=float, default=0.30)
    args = ap.parse_args()
    O = args.outputs
    out = args.out or os.path.join(O, "05c_confidence")
    os.makedirs(out, exist_ok=True)

    cand = {r["pocket_id"]: r for r in load(os.path.join(O, "02_pocket_candidates/pocket_candidates.csv"))}
    val = {r["pocket_id"]: r for r in load(os.path.join(O, "03_pocket_validation/pocket_validation.csv"))}
    ligc = load(os.path.join(O, "04_ligand_clusters/ligand_clusters.csv"))
    # 정제본 우선, 없으면 원본 final
    final = load(os.path.join(O, "05b_refined/selection_summary.csv")) or \
            load(os.path.join(O, "05_final/selection_summary.csv"))
    frag = load(os.path.join(O, "06_validation/fragment_compare.csv"))
    refine = load(os.path.join(O, "05b_refined/refine_summary.csv"))

    if not final or not cand:
        raise SystemExit("필요한 출력(final/pocket_candidates) 없음 — 파이프라인 먼저 실행")

    m1 = final[0]                                   # model_1 = 1순위
    fp = m1["pocket_id"]                             # 답 후보 포켓
    sizes = sorted((int(c["size"]) for c in cand.values()), reverse=True)
    total = sum(sizes) or 1
    p1 = cand.get(fp, {})
    p1size = int(p1.get("size", 0))

    # --- 신호들 ---
    dominance = p1size / total
    n_models = int(p1.get("n_models", 0))
    models = p1.get("models", "")
    af3 = "af3" in models
    separation = (sizes[0] / sizes[1]) if len(sizes) > 1 and sizes[1] else float("inf")
    # pose 수렴: model_1의 리간드 클러스터 size / 포켓 size
    lc = next((r for r in ligc if r["pocket_id"] == fp and r["ligand_cluster_id"] == m1.get("ligand_cluster_id")), {})
    convergence = (int(lc.get("size", 0)) / p1size) if p1size else 0.0
    p2d = fnum(val.get(fp, {}).get("p2rank_dist"))
    pocket_pass = str(val.get(fp, {}).get("pass", "")).lower() in ("true", "1")
    pb_valid = [str(r.get("posebusters_valid", "")).lower() for r in final]
    pb_all_ok = all(v in ("true", "na", "") for v in pb_valid)
    n_frag = max([int(r.get("n_shared", 0) or 0) for r in frag], default=None) if frag else None
    max_drift = max([fnum(r.get("drift"), 0) or 0 for r in refine], default=None) if refine else None

    # --- 판정 ---
    reasons = []
    high = (n_models >= 2 and af3 and dominance >= args.dominance_high
            and (p2d is not None and p2d <= args.p2rank_ok) and pb_all_ok)
    low = (n_models < 2 or dominance < 0.15 or (p2d is not None and p2d > 8.0) or not pb_all_ok)
    verdict = "HIGH" if high else ("LOW" if low else "MEDIUM")

    reasons.append(f"1순위 포켓 P{fp}: 우세도 {dominance:.0%}(total {total}), 합의 {n_models}모델({models}), af3 {'포함' if af3 else '없음'}")
    reasons.append(f"포켓 분리 P1/P2 = {separation:.1f}x" if separation != float('inf') else "단일 포켓")
    reasons.append(f"pose 수렴(top 리간드클러스터/포켓) = {convergence:.0%}")
    reasons.append(f"p2rank 거리 = {p2d}Å (pass={pocket_pass}), PoseBusters 최종 모두 valid={pb_all_ok}")
    if n_frag is not None:
        reasons.append(f"실험 fragment 공유 잔기 = {n_frag}")
    if max_drift is not None:
        reasons.append(f"정제 최대 drift = {max_drift}Å (작을수록 안정)")

    # --- 저장 ---
    with open(os.path.join(out, "confidence.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["verdict", "focus_pocket", "dominance", "n_models", "af3_present",
                    "separation", "pose_convergence", "p2rank_dist", "pocket_pass",
                    "posebusters_all_valid", "fragment_shared", "max_refine_drift"])
        w.writerow([verdict, fp, f"{dominance:.3f}", n_models, af3,
                    f"{separation:.2f}" if separation != float("inf") else "inf",
                    f"{convergence:.3f}", p2d, pocket_pass, pb_all_ok,
                    n_frag if n_frag is not None else "NA",
                    max_drift if max_drift is not None else "NA"])

    md = [f"# 선정 신뢰도 리포트\n\n",
          f"## 판정: **{verdict}**\n",
          "- HIGH: 자동선택 신뢰 가능 / MEDIUM: 검토 권장 / LOW: 사람 개입 필요\n\n",
          "## 근거\n"]
    for r in reasons:
        md.append(f"- {r}\n")
    md.append("\n## 최종 5개\n\n| model | pocket | size | iptm | gnina | posebusters |\n|---|---|---|---|---|---|\n")
    for r in final:
        md.append(f"| {r['model']} | P{r['pocket_id']} | {r['size']} | {r.get('iptm','')} "
                  f"| {r.get('gnina_affinity','')} | {r.get('posebusters_valid','')} |\n")
    md.append("\n## 해석 가이드\n")
    if verdict == "HIGH":
        md.append("- 합의·검증·유효성이 모두 일치 → 자동 선정 그대로 제출해도 무방.\n")
    elif verdict == "LOW":
        md.append("- 합의 약함/포켓 모호/검증 불일치 → 포켓 재검토·추가 샘플링·수동 선정 권장.\n")
    else:
        md.append("- 부분적으로만 일치 → 1~2순위 포켓 수동 확인, hedge 비중 조정 고려.\n")
    open(os.path.join(out, "confidence_report.md"), "w").write("".join(md))

    print(f"[confidence] 판정 = {verdict}")
    for r in reasons:
        print("  -", r)
    print(f"  -> {out}/confidence_report.md")


if __name__ == "__main__":
    main()

# ── 단독 실행 ── (stdlib; 아무 python)
#   $PY $SC/5c_confidence.py --outputs $OUT
