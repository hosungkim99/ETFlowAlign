# etflowalign/scripts/diagnose_vector_field.py
"""ETFlowAlign 체크포인트에 대한 간단한 벡터 필드 진단 스크립트.

Example:
    python -m etflowalign.scripts.diagnose_vector_field \
      --checkpoint /path/to/checkpoint_best.pt \
      --train-data /path/to/diffalign_example_train.pt \
      --device cuda
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from etflowalign.data import load_alignment_batch_from_pt
from etflowalign.model import AlignmentBatch, ETFlowAlignModel
from etflowalign.flow_matching import AlignmentFlowMatcher, FlowMatchingConfig


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose ETFlowAlign vector field.")

    parser.add_argument("--checkpoint", required=True, help="Checkpoint .pt path.")
    parser.add_argument("--train-data", required=True, help="Training batch .pt path.")
    parser.add_argument("--device", default="cuda", help="cuda or cpu.")
    parser.add_argument(
        "--t-values",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="Flow times to evaluate.",
    )
    parser.add_argument(
        "--csv-out",
        default="",
        help="Optional CSV output path.",
    )

    return parser


def make_step_batch(batch: AlignmentBatch, x_t: torch.Tensor) -> AlignmentBatch:
    return AlignmentBatch(
        query_pos=x_t,
        query_atom_type=batch.query_atom_type,
        query_batch=batch.query_batch,
        reference_pos=batch.reference_pos,
        reference_atom_type=batch.reference_atom_type,
        reference_batch=batch.reference_batch,
        pocket_pos=batch.pocket_pos,
        pocket_batch=batch.pocket_batch,
        query_node_attr=batch.query_node_attr,
        reference_node_attr=batch.reference_node_attr,
    )


def main() -> None:
    args = build_argparser().parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    print("checkpoint:", args.checkpoint)
    print("train_data:", args.train_data)
    print("device:", device)
    print("model_args:", ckpt["model_args"])
    print("flow_args:", ckpt["flow_args"])
    print("best_loss:", ckpt.get("best_loss"))
    print("best_step:", ckpt.get("best_step"))
    print("final_loss:", ckpt.get("final_loss"))

    batch, target_query_pos, metadata = load_alignment_batch_from_pt(
        args.train_data,
        require_target=True,
        device=device,
    )
    assert target_query_pos is not None

    print("metadata source:", metadata.get("source") if isinstance(metadata, dict) else None)
    print("query_pos:", batch.query_pos.shape, batch.query_pos.dtype)
    print(
        "query_node_attr:",
        None if batch.query_node_attr is None else (batch.query_node_attr.shape, batch.query_node_attr.dtype),
    )
    print("reference_pos:", None if batch.reference_pos is None else batch.reference_pos.shape)
    print(
        "reference_node_attr:",
        None if batch.reference_node_attr is None else (batch.reference_node_attr.shape, batch.reference_node_attr.dtype),
    )
    print("pocket_pos:", None if batch.pocket_pos is None else batch.pocket_pos.shape)
    print("target_query_pos:", target_query_pos.shape, target_query_pos.dtype)

    model = ETFlowAlignModel(**ckpt["model_args"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    flow_args = ckpt["flow_args"].copy()
    flow_args["fixed_t"] = None
    matcher = AlignmentFlowMatcher(FlowMatchingConfig(**flow_args))

    rows = []

    num_graphs = int(batch.query_batch.max().item()) + 1
    print("source max abs diff from batch.query_pos:", (batch.query_pos - batch.query_pos).abs().max().item())

    for tval in args.t_values:
        t_graph = torch.full(
            (num_graphs,),
            float(tval),
            device=device,
            dtype=batch.query_pos.dtype,
        )

        x_t, true_u = matcher.build_training_state(
            batch=batch,
            target_query_pos=target_query_pos,
            t_graph=t_graph,
        )

        step_batch = make_step_batch(batch, x_t)

        with torch.no_grad():
            pred_u = model(step_batch, t_graph)

        err = pred_u - true_u
        mse = err.pow(2).sum(dim=-1).mean().item()
        rmse = mse ** 0.5
        max_err = err.norm(dim=-1).max().item()
        mean_err = err.norm(dim=-1).mean().item()

        print(f"t={tval:.2f}")
        print(f"  vector MSE loss-like: {mse:.6f}")
        print(f"  vector RMSE: {rmse:.6f} Å")
        print(f"  mean atom vector error: {mean_err:.6f} Å")
        print(f"  max atom vector error: {max_err:.6f} Å")

        rows.append(
            {
                "t": float(tval),
                "vector_mse": mse,
                "vector_rmse_A": rmse,
                "mean_atom_vector_error_A": mean_err,
                "max_atom_vector_error_A": max_err,
            }
        )

    if args.csv_out:
        out_path = Path(args.csv_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print("csv saved to:", out_path)


if __name__ == "__main__":
    main()
    
'''
TRAIN_ROOT=/home/deepfold/users/hosung/work/ETFlowAlign/etflowalign/smoke_tests
INPUT_PT=$TRAIN_ROOT/00_inputs/diffalign_example/diffalign_example_train.pt
OUT_ROOT=$TRAIN_ROOT/04_equivariant_basis_head/nodeattr

python -m etflowalign.scripts.diagnose_vector_field \
  --checkpoint $OUT_ROOT/checkpoints/etflowalign_basis_nodeattr_randomt_3k_best.pt \
  --train-data $INPUT_PT \
  --device cuda \
  --t-values 0.0 0.25 0.5 0.75 1.0 \
  --csv-out $OUT_ROOT/diagnostics/diagnose_basis_nodeattr_randomt_3k_best.csv
'''
