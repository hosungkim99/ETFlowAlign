# etflowalign/scripts/10_prepare_aligned_pairs_dataset.py
"""GEOM-Drugs 정렬쌍에서 ETFlowAlign reference 조건부 데이터셋을 구축한다.

이 스크립트는 DiffAlign 논문(MLSB 2025)의 학습 데이터 구성을 그대로 따른다:
GEOM-Drugs에서 2D Tanimoto/3D shape로 선별·정렬한 (query, reference) 분자 쌍을
denoiser가 reference에 조건부로 query 구조를 생성하도록 학습시키기 위한 데이터로 만든다.

입력 디렉터리 구조(서버):
    <pairs-root>/complex_NNNNN/query.sdf       # reference에 이미 정렬된 query 컨포머(정답)
    <pairs-root>/complex_NNNNN/reference.sdf   # 조건이 되는 reference 분자

각 쌍에 대해 ``diffalign_adapter.build_diffalign_example_train_payload``로
복합체별 ``.pt`` 페이로드(diffalign_adapter와 동일 스키마)를 생성한다:

    target_query_pos = query 컨포머 - reference_center   (정답 x0, reference 중심 좌표계)
    reference_pos    = reference 컨포머 - reference_center (조건 xr)
    query_pos        = target과 동일(기본) 또는 무작위 강체 변환 소스
    pocket_*         = 없음 (DiffAlign은 포켓을 denoiser에 넣지 않음; 포켓은 추론 UFF에서만)

실패 사례(SDF 읽기/정제 불가 등)는 건너뛰고 failures.log에 기록되며,
manifest.csv에는 AlignmentDataset / train_val_split에 사용할 성공 복합체 목록이 저장된다.

Example:
    python -m etflowalign.scripts.10_prepare_aligned_pairs_dataset \
      --pairs-root /home/deepfold/users/hosung/dataset/GEOM-Drugs_AlignedPairs \
      --out-dir /home/deepfold/users/hosung/dataset/geom_aligned_pt \
      --limit 50
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import traceback
from pathlib import Path

import torch

from etflowalign.diffalign_adapter import build_diffalign_example_train_payload


def build_argparser() -> argparse.ArgumentParser:
    """정렬쌍 데이터셋 구축 CLI의 argparse 파서를 구성한다.

    한 줄 요약:
        GEOM-Drugs 정렬쌍에서 reference 조건부 데이터셋을 만드는 데 필요한 인자를 정의한 파서를 만든다.
    생성이유:
        정렬쌍 루트/출력 디렉터리/매니페스트 경로, 시드, 소스 무작위화 여부, 개수 제한을
        CLI로 지정하기 위한 인자 사양을 한 곳에 모으기 위함.
    역할:
        --pairs-root, --out-dir, --manifest, --seed, --randomize-source, --limit 인자를 등록한다.
    메커니즘:
        argparse.ArgumentParser를 생성하고 add_argument로 각 옵션을 정의해 반환한다.
    파라미터:
        없음.
    반환:
        argparse.ArgumentParser - 위 인자들이 등록된 파서 객체.
    파이프라인 단계:
        1단계(데이터 준비) - 정렬쌍 데이터셋 CLI 인자 정의.
    """
    p = argparse.ArgumentParser(
        description="Build reference-conditioned ETFlowAlign dataset from GEOM-Drugs aligned pairs."
    )
    p.add_argument("--pairs-root", required=True,
                   help="Aligned-pairs root (contains complex_*/ dirs with query.sdf + reference.sdf).")
    p.add_argument("--out-dir", required=True, help="Output directory for per-complex .pt payloads.")
    p.add_argument("--manifest", default="", help="Manifest CSV path. Default: <out-dir>/manifest.csv")
    p.add_argument("--seed", type=int, default=0, help="Base seed for the (optional) source transform.")
    p.add_argument("--randomize-source", action="store_true",
                   help="Make query_pos a random rigid transform of the target. Default: query_pos = target "
                        "(the flow-matching source is sampled from the prior at train time, so this rarely matters).")
    p.add_argument("--query-name", default="query.sdf", help="Filename of the query SDF inside each complex dir.")
    p.add_argument("--reference-name", default="reference.sdf",
                   help="Filename of the reference SDF inside each complex dir.")
    p.add_argument("--limit", type=int, default=0, help="Process at most N complexes (0 = all).")
    return p


def find_pairs(root: str, query_name: str, reference_name: str) -> list[tuple[str, str, str]]:
    """root 하위에서 (query.sdf, reference.sdf)가 모두 있는 복합체들을 열거한다.

    한 줄 요약:
        root 아래를 재귀 탐색해 query/reference SDF가 둘 다 존재하는 복합체의
        (complex_id, query_path, reference_path) 목록을 만든다.
    생성이유:
        정렬쌍 디렉터리 구조에서 처리 대상 복합체를 자동으로 열거하기 위함.
    역할:
        ``*/<query_name>`` 파일을 찾고, 같은 디렉터리에 ``<reference_name>``이 있는 경우만 채택한다.
    메커니즘:
        glob.glob(..., recursive=True)로 query SDF를 정렬 수집하고, 형제 reference SDF의
        존재를 확인한 뒤 디렉터리명을 complex_id로 사용한다.
    파라미터:
        root (str): 정렬쌍 루트 디렉터리 경로.
        query_name (str): 복합체 디렉터리 내 query SDF 파일명.
        reference_name (str): 복합체 디렉터리 내 reference SDF 파일명.
    반환:
        list[tuple[str, str, str]]: (complex_id, query_path, reference_path) 목록.
    파이프라인 단계:
        1단계(데이터 준비) - 처리 대상 정렬쌍 열거.
    """
    out = []
    for qpath in sorted(glob.glob(os.path.join(root, "**", query_name), recursive=True)):
        cdir = os.path.dirname(qpath)
        rpath = os.path.join(cdir, reference_name)
        if os.path.exists(rpath):
            out.append((os.path.basename(cdir), qpath, rpath))
    return out


def main() -> None:
    """CLI 진입점: 정렬쌍들을 순회하며 페이로드를 생성하고 매니페스트를 작성한다.

    한 줄 요약:
        정렬쌍을 열거해 각각의 .pt 페이로드를 저장하고, 성공 목록 매니페스트와 실패 로그를 남긴다.
    생성이유:
        대규모 GEOM-Drugs 정렬쌍을 불량 쌍에 강건하게 일괄 전처리하여 학습용 데이터셋을 구축하기 위함.
    역할:
        출력 디렉터리/매니페스트 경로를 준비하고, 정렬쌍을 (옵션상 제한 적용해) 순회하며
        build_diffalign_example_train_payload로 페이로드를 만들어 저장하고, 성공/실패를 집계해
        매니페스트 CSV와 failures.log를 기록한다.
    메커니즘:
        find_pairs로 목록을 얻고 limit 적용. 각 쌍마다 try/except로 변환기(시드는 base+i)를 호출해
        reference/conditioning 메타를 덧붙여 torch.save하고 rows에 메타를 누적, 예외 시
        failures.log에 traceback을 기록한다. 200개마다 진행 상황을 출력하고, 마지막에
        manifest.csv를 쓰고 요약을 출력한다.
    파라미터:
        없음(인자는 build_argparser로 명령행에서 파싱).
    반환:
        None: 결과는 .pt 파일들, manifest.csv, failures.log 작성과 표준출력 요약이다.
    파이프라인 단계:
        1단계(데이터 준비) - 정렬쌍 데이터셋 일괄 구축 진입점.
    """
    args = build_argparser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else out_dir / "manifest.csv"

    pairs = find_pairs(args.pairs_root, args.query_name, args.reference_name)
    if args.limit > 0:
        pairs = pairs[: args.limit]
    print(f"[pairs] found {len(pairs)} aligned pairs under {args.pairs_root}")

    rows = []
    n_ok = n_fail = 0
    with open(out_dir / "failures.log", "w") as flog:
        for i, (complex_id, qpath, rpath) in enumerate(pairs):
            try:
                payload = build_diffalign_example_train_payload(
                    repo_root=".",
                    query_sdf=qpath,
                    reference_sdf=rpath,
                    pocket_pdb=None,  # DiffAlign: 포켓은 denoiser 입력 아님
                    randomize_query_pos=args.randomize_source,
                    seed=args.seed + i,
                )
                # 데이터 출처/조건 메타를 명시(포켓 조건 PDBbind와 구분).
                payload["source"] = "geom_drugs_aligned_pair"
                payload["complex_id"] = complex_id
                payload["conditioning"] = "reference"
                out_path = out_dir / f"{complex_id}.pt"
                torch.save(payload, out_path)
                rows.append({
                    "complex_id": complex_id,
                    "path": str(out_path),
                    "n_query": int(payload["query_pos"].size(0)),
                    "n_reference": int(payload["reference_pos"].size(0))
                    if payload.get("reference_pos") is not None else 0,
                })
                n_ok += 1
            except Exception as exc:  # noqa: BLE001 - 데이터셋 전처리는 불량 쌍에 강건해야 함
                n_fail += 1
                flog.write(f"{complex_id}\t{exc}\n{traceback.format_exc()}\n")
            if (i + 1) % 200 == 0:
                print(f"[pairs] {i + 1}/{len(pairs)}  ok={n_ok} fail={n_fail}")

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["complex_id", "path", "n_query", "n_reference"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[pairs] done: ok={n_ok} fail={n_fail}")
    print(f"[pairs] manifest: {manifest_path}")
    print(f"[pairs] failures log: {out_dir / 'failures.log'}")


if __name__ == "__main__":
    main()
