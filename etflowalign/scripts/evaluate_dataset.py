# etflowalign/scripts/evaluate_dataset.py
"""ETFlowAlign 체크포인트를 복합체 집합에 대해 평가한다: 포즈 RMSD + 기하 구조.

각 복합체에 대해 저장된 (강체 무작위화된) 소스에서 포즈를 샘플링하고,
타겟 결합 포즈(동일 프레임, 알려진 대응관계)에 대한 원자 단위 RMSD,
COM 거리, 결합 기하 구조를 측정한다. 분포와 성공률을 보고한다.

NOTE: 진정한 일반화 수치를 얻으려면 모델이 학습하지 않은 복합체가 필요하다.
``--val-fraction``/``--seed``/``--split val``을 사용하면 ``train.py --val-fraction``이
보류한 것과 동일한 검증 서브셋을 결정론적으로 선택하거나,
``--data-dir``을 별도의 데이터셋으로 지정할 수 있다.

Example:
    python -m etflowalign.scripts.evaluate_dataset \
      --checkpoint .../etflowalign_pdbbind_shape_best.pt \
      --data-dir .../pdbbind_refined_pt \
      --val-fraction 0.1 --seed 0 --split val \
      --solver heun --n-steps 50 --csv-out .../eval_shape_val.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from etflowalign.dataset import AlignmentDataset, train_val_split
from etflowalign.evaluation import evaluate_paths
from etflowalign.model import ETFlowAlignModel

def build_argparser() -> argparse.ArgumentParser:
    """데이터셋 평가 CLI의 argparse 파서를 구성한다.

    한 줄 요약:
        ETFlowAlign 포즈 평가에 필요한 명령행 인자를 정의한 ArgumentParser를 만든다.
    생성이유:
        체크포인트, 데이터 소스(디렉터리/매니페스트), 조건화 방식, ODE 설정, 분할 옵션,
        평가 개수 제한, CSV 출력 등 평가 실행에 필요한 인자를 한 곳에 정의하기 위함.
    역할:
        --checkpoint, --data-dir, --manifest, --conditioning, --n-steps, --solver, --device,
        --val-fraction, --seed, --split, --limit, --csv-out 인자를 등록한다.
    메커니즘:
        argparse.ArgumentParser를 생성하고 add_argument로 각 옵션을 정의해 반환한다.
    파라미터:
        없음.
    반환:
        argparse.ArgumentParser - 위 인자들이 등록된 파서 객체.
    파이프라인 단계:
        6단계(평가) - CLI 인자 정의.
    """
    p = argparse.ArgumentParser(description="Evaluate ETFlowAlign pose RMSD on a dataset.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", default="", help="Directory of per-complex .pt payloads.")
    p.add_argument("--manifest", default="", help="Text file listing .pt paths (one per line).")
    p.add_argument("--conditioning", default="pocket", choices=["pocket", "reference", "both"])
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument("--solver", default="heun", choices=["euler", "heun"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--val-fraction", type=float, default=0.0, help="If >0, select a split via train_val_split.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", default="all", choices=["all", "train", "val"])
    p.add_argument("--limit", type=int, default=0, help="Evaluate at most N complexes (0 = all).")
    p.add_argument("--csv-out", default="")
    return p


def gather_paths(args: argparse.Namespace) -> list[str]:
    """평가할 복합체 .pt 파일 경로 목록을 인자에 따라 수집한다.

    한 줄 요약:
        매니페스트 또는 데이터 디렉터리에서 경로를 모으고, 분할/개수 제한을 적용해 반환한다.
    생성이유:
        평가 대상 집합을 train.py의 검증 분할과 동일하게 결정론적으로 선택하거나,
        별도 데이터셋/매니페스트를 지정할 수 있도록 경로 수집 로직을 일원화하기 위함.
    역할:
        --manifest가 있으면 파일에서 경로를 읽고, 없으면 --data-dir에서 데이터셋 경로를 수집한다.
        --val-fraction>0이고 split이 all이 아니면 train_val_split으로 분할하고, --limit이 있으면 잘라낸다.
    메커니즘:
        매니페스트 파일을 라인 단위로 읽거나 AlignmentDataset.from_directory(...).paths로 경로를 얻고,
        train_val_split(paths, val_fraction, seed)로 분할 후 split에 따라 train/val 부분집합을 고른 뒤,
        limit>0이면 앞에서 limit개로 슬라이싱한다.
    파라미터:
        args (argparse.Namespace): manifest/data_dir/val_fraction/seed/split/limit 등을 담은 파싱 결과.
    반환:
        list[str]: 평가에 사용할 .pt 파일 경로 목록.
    예외:
        ValueError: --data-dir와 --manifest가 모두 비어 있을 때 발생.
    파이프라인 단계:
        6단계(평가) - 평가 대상 경로 수집.
    """
    if args.manifest:
        with open(args.manifest) as f:
            paths = [ln.strip() for ln in f if ln.strip()]
    elif args.data_dir:
        paths = AlignmentDataset.from_directory(args.data_dir).paths
    else:
        raise ValueError("Provide --data-dir or --manifest.")
    if args.val_fraction > 0.0 and args.split != "all":
        train_paths, val_paths = train_val_split(paths, args.val_fraction, args.seed)
        paths = train_paths if args.split == "train" else val_paths
    if args.limit > 0:
        paths = paths[: args.limit]
    return paths

@torch.no_grad()
def main() -> None:
    """CLI 진입점: 데이터셋에 대해 포즈 RMSD와 기하 구조를 평가한다.

    한 줄 요약:
        체크포인트를 불러와 대상 복합체들에 대해 포즈를 샘플링하고 RMSD/성공률/기하 통계를 출력·저장한다.
    생성이유:
        학습된 ETFlowAlign 모델의 정렬 성능을 복합체 집합 단위로 정량 평가하는 CLI 도구를 제공하기 위함.
    역할:
        모델을 복원하고 gather_paths로 평가 경로를 모은 뒤 evaluate_paths로 포즈를 샘플링·채점하여,
        소스/예측 RMSD 평균·중앙값, <2A/<5A 성공률, 소스 대비 개선 비율을 출력하고 옵션 시 CSV로 저장한다.
    메커니즘:
        torch.load로 ckpt 로드 후 ETFlowAlignModel 복원·eval. gather_paths(args)로 경로 수집,
        evaluate_paths(model, paths, conditioning, n_steps, solver, device, limit=0)로 요약/행 데이터를 받는다.
        (--limit은 gather_paths에서 이미 적용되어 여기서는 0). 평가된 복합체가 없으면 조기 반환하고,
        csv_out 지정 시 지정 필드만 골라 CSV로 기록한다.
    파라미터:
        없음(인자는 build_argparser로 명령행에서 파싱).
    반환:
        None: 결과는 표준출력과(지정 시) CSV로 출력된다.
    파이프라인 단계:
        6단계(평가) - 평가 실행 진입점.
    """
    args = build_argparser().parse_args()
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = ETFlowAlignModel(**ckpt["model_args"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[eval] checkpoint={args.checkpoint}")
    print(f"[eval] model_args={ckpt['model_args']}")
    print(f"[eval] best_loss={ckpt.get('best_loss')} best_step={ckpt.get('best_step')}")


    paths = gather_paths(args)
    print(f"[eval] {len(paths)} complexes | split={args.split} solver={args.solver} n_steps={args.n_steps}")

    summary, rows = evaluate_paths(
        model,
        paths,
        conditioning=args.conditioning,
        n_steps=args.n_steps,
        solver=args.solver,
        device=device,
        limit=0,  # gather_paths에서 --limit이 이미 적용됨
    )
    if summary.get("n", 0) == 0:
        print("[eval] no complexes evaluated.")
        return

    print("\n=== pose RMSD to target (Angstrom) ===")
    print(f"  complexes      : {summary['n']}")
    print(f"  source RMSD    : mean={summary['source_rmsd_mean']:.3f} median={summary['source_rmsd_median']:.3f}  (starting point)")
    print(f"  predicted RMSD : mean={summary['pred_rmsd_mean']:.3f} median={summary['pred_rmsd_median']:.3f}")
    print(f"  success        : <2A={summary['frac_lt2']*100:.1f}%  <5A={summary['frac_lt5']*100:.1f}%")
    print(f"  improved vs src: {summary['improved_frac']*100:.1f}% of complexes")

    if args.csv_out:
        out = Path(args.csv_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fields = ["pdb_id", "n_atoms", "source_rmsd", "pred_rmsd", "com_dist", "bond_min", "bond_mean", "bond_max"]
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fields})
        print(f"[eval] csv saved to: {out}")


if __name__ == "__main__":
    main()
