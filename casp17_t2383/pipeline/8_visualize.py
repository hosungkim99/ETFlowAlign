#!/usr/bin/env python3
"""
visualize.py (범용, 워크플로 7번) - 선별 결과 시각화. complex_io 사용 → 리간드 개수 무관.
boltz2 env (numpy+gemmi+matplotlib). 그래프 텍스트 영어, 주석 한글.

출력 1) <out>: 4패널 (리간드 PCA by model / by cluster / 모델별 iptm 분포 / 상위클러스터 구성)
출력 2) --contacts 주면 contact fingerprint PNG (per-residue 접촉빈도; fragment 공유잔기 강조)
"""
import argparse, csv, os
import numpy as np
import gemmi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import complex_io as cio

PALETTE = ["tab:blue", "tab:green", "tab:orange", "tab:red", "tab:purple", "tab:brown"]


def kabsch(P, Q):
    Pc, Qc = P.mean(0), Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, Qc - R @ Pc


def color_for(models):
    uniq = sorted(set(models))
    return {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(uniq)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True, help="4패널 PNG")
    ap.add_argument("--contacts", default="", help="contact_residues.csv (있으면 fingerprint PNG)")
    ap.add_argument("--fragment", default="", help="fragment_compare.csv (공유잔기 강조)")
    ap.add_argument("--threshold", type=float, default=2.0)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    rows = [r for r in csv.DictReader(open(args.table)) if r.get("cif")]
    rows.sort(key=lambda r: int(r["rank"]))
    ref_ca, ref_ligs = cio.parse_complex(rows[0]["cif"])
    ref_elems, _ = cio.ligand_concat(ref_ligs)

    vecs, models, iptms = [], [], []
    for i, r in enumerate(rows):
        ca, ligs = cio.parse_complex(r["cif"])
        elems, coords = cio.ligand_concat(ligs)
        if not ca or elems != ref_elems:
            continue
        common = [k for k in ref_ca if k in ca]
        if len(common) < 50:
            continue
        R, t = kabsch(np.array([ca[k] for k in common]), np.array([ref_ca[k] for k in common]))
        L = (R @ np.array(coords).T).T + t
        vecs.append(L.flatten()); models.append(r["model"])
        try:
            iptms.append((r["model"], float(r["iptm"])))
        except (TypeError, ValueError):
            pass
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(rows)}")
    X = np.array(vecs); models = np.array(models)
    print(f"[viz] {len(X)} structures")
    MCOL = color_for(models)

    Xc = X - X.mean(0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    PC = Xc @ Vt[:2].T
    var = (S ** 2) / (S ** 2).sum() * 100

    reps, labels = [], []
    for v in X:
        for ci, rep in enumerate(reps):
            if np.sqrt(((v - rep) ** 2).reshape(-1, 3).sum(1).mean()) <= args.threshold:
                labels.append(ci); break
        else:
            reps.append(v); labels.append(len(reps) - 1)
    labels = np.array(labels); sizes = np.bincount(labels)
    top = np.argsort(sizes)[::-1][:6]

    fig, ax = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Ligand pose selection overview", fontsize=15)
    for mdl in sorted(set(models)):
        sel = models == mdl
        ax[0, 0].scatter(PC[sel, 0], PC[sel, 1], s=10, alpha=0.5, c=MCOL[mdl], label=mdl)
    ax[0, 0].set_title(f"(1) Ligand PCA by model (PC1 {var[0]:.1f}%, PC2 {var[1]:.1f}%)")
    ax[0, 0].legend()
    other = ~np.isin(labels, top)
    ax[0, 1].scatter(PC[other, 0], PC[other, 1], s=8, alpha=0.3, c="lightgray", label="other")
    for j, ci in enumerate(top):
        sel = labels == ci
        ax[0, 1].scatter(PC[sel, 0], PC[sel, 1], s=12, alpha=0.6,
                         color=PALETTE[j % len(PALETTE)], label=f"cluster {ci+1} (n={sizes[ci]})")
    ax[0, 1].set_title("(2) Ligand PCA by cluster"); ax[0, 1].legend(fontsize=8)
    by = {}
    for mdl, v in iptms:
        by.setdefault(mdl, []).append(v)
    names = sorted(by)
    ax[1, 0].boxplot([by[n] for n in names], labels=names)
    ax[1, 0].set_title("(3) iptm distribution by model (calibration)"); ax[1, 0].set_ylabel("iptm")
    topc = np.argsort(sizes)[::-1][:8]
    bottom = np.zeros(len(topc))
    for mdl in sorted(set(models)):
        counts = [int(np.sum((labels == ci) & (models == mdl))) for ci in topc]
        ax[1, 1].bar(range(len(topc)), counts, bottom=bottom, color=MCOL[mdl], label=mdl)
        bottom += counts
    ax[1, 1].set_xticks(range(len(topc)))
    ax[1, 1].set_xticklabels([f"c{ci+1}" for ci in topc])
    ax[1, 1].set_title("(4) Model composition of top clusters"); ax[1, 1].legend()
    plt.tight_layout(); plt.savefig(args.out, dpi=130)
    print(f"[viz] saved: {args.out}")

    # contact fingerprint (선택)
    if args.contacts and os.path.exists(args.contacts):
        crows = list(csv.DictReader(open(args.contacts)))
        frag_res = set()
        if args.fragment and os.path.exists(args.fragment):
            for fr in csv.DictReader(open(args.fragment)):
                for x in (fr.get("shared_with_prediction", "") or "").split(";"):
                    if x.strip().isdigit():
                        frag_res.add(int(x))
        data = []
        for r in crows:
            try:
                data.append((int(r["residue"]), float(r["contact_frequency"])))
            except (ValueError, KeyError):
                pass
        data = [d for d in data if d[1] >= 0.2]
        data.sort(key=lambda x: x[0])
        if data:
            xs = [str(k) for k, _ in data]; ys = [v * 100 for _, v in data]
            cols = ["crimson" if k in frag_res else "steelblue" for k, _ in data]
            plt.figure(figsize=(14, 6))
            plt.bar(xs, ys, color=cols)
            plt.ylabel("contact frequency (%)"); plt.xlabel("residue number")
            plt.title("Ligand-contact fingerprint  |  red = shared with experimental fragment")
            plt.xticks(rotation=60); plt.tight_layout()
            fp = os.path.join(os.path.dirname(args.out), "contact_fingerprint.png")
            plt.savefig(fp, dpi=130)
            print(f"[viz] saved: {fp}")


if __name__ == "__main__":
    main()

# ── 단독 실행 ── (먼저: source $CASP17/scripts/env_setup.sh)
#   SC=$CASP17/users/hosungkim/scripts ; OUT=$CASP17/users/hosungkim/targets/T2383
#   micromamba run -n boltz2 python $SC/8_visualize.py \
#       --table $OUT/00_collect/master_table.csv --out $OUT/07_viz/visualization.png \
#       --contacts $OUT/06_validation/contact_residues.csv \
#       [--fragment $OUT/06_validation/fragment_compare.csv]
# (평소엔 run_pipeline.py 가 순서대로 자동 호출)
