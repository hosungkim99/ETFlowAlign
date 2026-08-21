"""데이터 게이트: 정렬쌍이 실제로 형상 정렬돼 있는지 육안 + 통계 확인 (SPEC §3.2).

- 소수 쌍을 combined SDF(query+reference 한 파일)로 내보내 뷰어에서 겹침 확인
- 통계: 2D Tanimoto, 무게중심 거리, 3D shape Tanimoto(겹침)

서버 실행 예:
  PYTHONPATH=. python etflowalign/scripts/02_visualize_pairs.py \
      --root /path/GEOM-Drugs_AlignedPairs --out /path/vis --n 20
"""

from __future__ import annotations

import argparse
import os

from rdkit import Chem
from rdkit.Chem import rdShapeHelpers

from etflowalign.data.pairs import load_mol, tanimoto_2d


def centroid_distance(m1: Chem.Mol, m2: Chem.Mol) -> float:
    import numpy as np
    c1 = m1.GetConformer().GetPositions().mean(axis=0)
    c2 = m2.GetConformer().GetPositions().mean(axis=0)
    return float(np.linalg.norm(c1 - c2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    import glob
    dirs = sorted(glob.glob(os.path.join(args.root, "complex_*")))[: args.n]

    print(f"{'complex':<16} {'tani2D':>7} {'cdist':>7} {'shapeT':>7}")
    tanis, cdists, shapes = [], [], []
    for d in dirs:
        name = os.path.basename(d)
        q = os.path.join(d, "query.sdf")
        r = os.path.join(d, "reference.sdf")
        if not (os.path.isfile(q) and os.path.isfile(r)):
            continue
        mq, mr = load_mol(q), load_mol(r)
        tani = tanimoto_2d(mq, mr)
        cdist = centroid_distance(mq, mr)
        shape_t = 1.0 - rdShapeHelpers.ShapeTanimotoDist(mq, mr)  # 겹침(1=완전겹침)
        tanis.append(tani); cdists.append(cdist); shapes.append(shape_t)

        # query + reference 를 한 SDF 로 (뷰어에서 겹침 확인)
        w = Chem.SDWriter(os.path.join(args.out, f"{name}_pair.sdf"))
        mq.SetProp("_Name", f"{name}_query"); w.write(mq)
        mr.SetProp("_Name", f"{name}_reference"); w.write(mr)
        w.close()
        print(f"{name:<16} {tani:>7.3f} {cdist:>7.2f} {shape_t:>7.3f}")

    if tanis:
        import statistics as st
        print("\n[요약 median] "
              f"tani2D={st.median(tanis):.3f}  "
              f"centroid_dist={st.median(cdists):.2f}A  "
              f"shape_overlap={st.median(shapes):.3f}")
        print("[게이트] shape_overlap 이 높고(>0.5) centroid_dist 가 작으면 정렬 양호")
        print(f"[출력] combined SDF -> {args.out}")


if __name__ == "__main__":
    main()
