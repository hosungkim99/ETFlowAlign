#!/usr/bin/env python3
"""
fragment_compare.py (범용, 워크플로 6번) - 실험 구조(fragment/template)와 contact residue 비교.
complex_io 사용. boltz2 env (numpy+gemmi).

실험 cif(예: 9bbh)를 우리 1순위 포즈(model_1)에 Cα 정렬(번호 offset 자동탐지)한 뒤,
실험 리간드의 접촉 잔기를 우리 번호로 매핑 → contact_residues.csv(우리 예측)와 겹침을 출력.
오버레이 PDB도 저장(PyMOL 확인용). 우리 예측 site가 실험으로 뒷받침되는지 정량 검증.
"""
import argparse, csv, os
import numpy as np
import gemmi
import complex_io as cio


def ca_by_resnum(model, offset=0):
    ca = {}
    for ch in model:
        for r in ch:
            t = gemmi.find_tabulated_residue(r.name)
            if t and t.is_amino_acid():
                for a in r:
                    if a.name == "CA":
                        ca[r.seqid.num + offset] = [a.pos.x, a.pos.y, a.pos.z]
                        break
    return ca


def kabsch(P, Q):
    Pc, Qc = P.mean(0), Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, Qc - R @ Pc


def detect_offset(ca_our, model_exp, rng=6):
    """번호 offset 자동탐지: 후보 offset 중 정렬 Cα RMSD 최소(공통잔기 충분)인 것."""
    best = (None, 1e9, 0)
    for off in range(-rng, rng + 1):
        ca_e = ca_by_resnum(model_exp, off)
        common = sorted(set(ca_our) & set(ca_e))
        if len(common) < 50:
            continue
        P = np.array([ca_e[k] for k in common]); Q = np.array([ca_our[k] for k in common])
        R, t = kabsch(P, Q)
        rmsd = float(np.sqrt((((R @ P.T).T + t - Q) ** 2).sum(1).mean()))
        if rmsd < best[1]:
            best = (off, rmsd, len(common))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model1", required=True, help="우리 1순위 포즈 cif")
    ap.add_argument("--experimental", required=True, help="실험 구조 cif (fragment/template)")
    ap.add_argument("--our-contacts", required=True, help="contact_residues.csv (워크플로 5)")
    ap.add_argument("--out", required=True, help="fragment_compare.csv")
    ap.add_argument("--overlay", default="", help="오버레이 PDB 저장 경로(선택)")
    ap.add_argument("--cutoff", type=float, default=4.0)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    our = gemmi.read_structure(args.model1)
    exp = gemmi.read_structure(args.experimental)
    m1, me = our[0], exp[0]

    ca_our = ca_by_resnum(m1, 0)
    off, rmsd, ncommon = detect_offset(ca_our, me)
    if off is None:
        raise SystemExit("offset 탐지 실패 (공통 Cα 부족) — 다른 단백질일 수 있음")
    print(f"[fragment_compare] offset={off} (our = exp + {off}), 정렬 Cα RMSD {rmsd:.2f}Å (n={ncommon})")

    # 실험 구조를 우리 좌표계로 정렬
    ca_e = ca_by_resnum(me, off)
    common = sorted(set(ca_our) & set(ca_e))
    R, t = kabsch(np.array([ca_e[k] for k in common]), np.array([ca_our[k] for k in common]))
    tr = gemmi.Transform(); tr.mat.fromlist([list(r) for r in R]); tr.vec.fromlist(list(t))
    for ch in me:
        for r in ch:
            for a in r:
                a.pos = gemmi.Position(*tr.apply(a.pos))

    # 실험 리간드(들)의 접촉 잔기 (우리 번호로; native+off)
    _, exp_ligs = cio.parse_complex(args.experimental)  # 잔기명 확인용(원본)
    ns = gemmi.NeighborSearch(me, exp.cell, args.cutoff + 1).populate()
    exp_contacts = {}   # ligand_resname -> set(our_resnum)
    for ch in me:
        for r in ch:
            t2 = gemmi.find_tabulated_residue(r.name)
            if (t2 and (t2.is_amino_acid() or t2.is_nucleic_acid())) or r.name in cio.WATER:
                continue
            s = set()
            for a in r:
                if a.element.name == "H":
                    continue
                for mk in ns.find_atoms(a.pos, "\0", radius=args.cutoff):
                    rr = mk.to_cra(me).residue
                    tt = gemmi.find_tabulated_residue(rr.name)
                    if tt and tt.is_amino_acid():
                        s.add(rr.seqid.num + off)   # 우리 번호로
            if s:
                exp_contacts.setdefault(r.name, set()).update(s)

    # 우리 예측 접촉 잔기 로드 (빈도>0 전부; resnum만)
    our_res = set()
    for row in csv.DictReader(open(args.our_contacts)):
        try:
            our_res.add(int(row["residue"]))
        except (ValueError, KeyError):
            pass

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["exp_ligand", "exp_contacts(our#)", "shared_with_prediction", "n_shared", "n_exp"])
        for lig, s in sorted(exp_contacts.items()):
            shared = sorted(s & our_res)
            w.writerow([lig, ";".join(map(str, sorted(s))), ";".join(map(str, shared)),
                        len(shared), len(s)])
            print(f"  {lig}: 접촉 {len(s)}개 중 예측과 공유 {len(shared)}개 -> {shared}")
    print(f"[fragment_compare] -> {args.out}")

    if args.overlay:
        for ch in me:
            for r in list(ch):
                t2 = gemmi.find_tabulated_residue(r.name)
                if (t2 and (t2.is_amino_acid() or t2.is_nucleic_acid())) or r.name in cio.WATER:
                    continue
                newch = gemmi.Chain("Z"); newch.add_residue(r); m1.add_chain(newch)
        our.write_pdb(args.overlay)
        print(f"[fragment_compare] overlay -> {args.overlay}")


if __name__ == "__main__":
    main()

# ── 단독 실행 ── (먼저: source $CASP17/scripts/env_setup.sh)
#   SC=$CASP17/users/hosungkim/scripts ; OUT=$CASP17/users/hosungkim/targets/T2383
#   micromamba run -n boltz2 python $SC/7_fragment_compare.py \
#       --model1 $OUT/05_final/model_1.cif --experimental <experimental.cif> \
#       --our-contacts $OUT/06_validation/contact_residues.csv \
#       --out $OUT/06_validation/fragment_compare.csv --overlay $OUT/06_validation/overlay_fragment.pdb
# (experimental_cif 있을 때만; 평소엔 run_pipeline.py 가 자동 호출)
