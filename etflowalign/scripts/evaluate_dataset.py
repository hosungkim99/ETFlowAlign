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
