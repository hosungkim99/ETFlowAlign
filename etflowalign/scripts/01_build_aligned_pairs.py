"""정렬쌍 디렉터리 -> per-complex .pt 대량 빌드 (서버 실행용).

입력 레이아웃:
  <root>/complex_0001/query.sdf, reference.sdf
  <root>/complex_0002/...
출력:
  <out>/complex_0001.pt  (build_pair_payload 결과 dict)

서버 실행 예:
  PYTHONPATH=. python etflowalign/scripts/01_build_aligned_pairs.py \
      --root /path/GEOM-Drugs_AlignedPairs --out /path/geom_pt --limit 0
"""

from __future__ import annotations

import argparse
import glob
import os

import torch

from etflowalign.data.pairs import build_pair_payload, validate_payload


def find_complexes(root: str):
    """complex_* 하위에서 query/reference SDF 쌍을 찾는다."""
    pairs = []
    for d in sorted(glob.glob(os.path.join(root, "complex_*"))):
        q = os.path.join(d, "query.sdf")
        r = os.path.join(d, "reference.sdf")
        if os.path.isfile(q) and os.path.isfile(r):
            pairs.append((os.path.basename(d), q, r))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="정렬쌍 루트 디렉터리")
    ap.add_argument("--out", required=True, help=".pt 출력 디렉터리")
    ap.add_argument("--limit", type=int, default=0, help="0=전체, N=처음 N개만")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    complexes = find_complexes(args.root)
    if args.limit > 0:
        complexes = complexes[: args.limit]
    print(f"[build] 발견한 쌍: {len(complexes)}")

    ok, fail = 0, 0
    n_atoms = []
    for name, q, r in complexes:
        try:
            p = build_pair_payload(q, r, mol_id=name)
            validate_payload(p)
            torch.save(p, os.path.join(args.out, f"{name}.pt"))
            n_atoms.append(int(p["z"].numel()))
            ok += 1
        except Exception as e:  # 실패는 건너뛰고 로그
            fail += 1
            print(f"  [skip] {name}: {e}")
        if (ok + fail) % 1000 == 0:
            print(f"  진행 {ok + fail}/{len(complexes)}  (ok={ok} fail={fail})")

    if n_atoms:
        import statistics
        print(f"[build] 완료 ok={ok} fail={fail}  "
              f"원자수 min/median/max = {min(n_atoms)}/{int(statistics.median(n_atoms))}/{max(n_atoms)}")
    else:
        print(f"[build] 완료 ok={ok} fail={fail}")


if __name__ == "__main__":
    main()
