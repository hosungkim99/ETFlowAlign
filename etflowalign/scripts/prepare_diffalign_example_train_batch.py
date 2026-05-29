# etflowalign/scripts/prepare_diffalign_example_train_batch.py
"""Create an ETFlowAlign pseudo-training .pt batch from the DiffAlign example.

This script uses the original query SDF conformer as target_query_pos after
centering it by the reference ligand center. This is intended for smoke testing
and single-example overfitting, not for real supervised training.

Example:
    python -m etflowalign.scripts.prepare_diffalign_example_train_batch \
      --repo-root /home/deepfold/users/hosung/work/ETFlowAlign \
      --out /home/deepfold/users/hosung/work/ETFlowAlign/etflowalign/smoke_tests/diffalign_example_train.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from etflowalign.diffalign_adapter import (
    build_diffalign_example_train_payload,
    validate_diffalign_example_train_payload,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert DiffAlign example into an ETFlowAlign pseudo-training .pt batch."
    )

    parser.add_argument(
        "--repo-root",
        type=str,
        required=True,
        help="ETFlowAlign repository root.",
    )
    parser.add_argument(
        "--query-sdf",
        type=str,
        default="",
        help="Path to query.sdf. Defaults to external/diffalign/diffalign/example/query.sdf.",
    )
    parser.add_argument(
        "--reference-sdf",
        type=str,
        default="",
        help="Path to reference.sdf. Defaults to external/diffalign/diffalign/example/reference.sdf.",
    )
    parser.add_argument(
        "--pocket-pdb",
        type=str,
        default="",
        help="Path to pocket.pdb. Defaults to external/diffalign/diffalign/example/pocket.pdb.",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output .pt path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for query_pos initialization.",
    )
    parser.add_argument(
        "--keep-query-conformer-as-source",
        action="store_true",
        help=(
            "Use centered query SDF conformer as query_pos instead of random N(0,I). "
            "Default uses random source coordinates."
        ),
    )

    return parser


def _default_example_paths(repo_root: Path) -> tuple[Path, Path, Path]:
    example_dir = repo_root / "external" / "diffalign" / "diffalign" / "example"
    return (
        example_dir / "query.sdf",
        example_dir / "reference.sdf",
        example_dir / "pocket.pdb",
    )


def _resolve_input_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    repo_root = Path(args.repo_root).resolve()
    default_query_sdf, default_reference_sdf, default_pocket_pdb = _default_example_paths(repo_root)

    query_sdf = Path(args.query_sdf).resolve() if args.query_sdf else default_query_sdf
    reference_sdf = Path(args.reference_sdf).resolve() if args.reference_sdf else default_reference_sdf
    pocket_pdb = Path(args.pocket_pdb).resolve() if args.pocket_pdb else default_pocket_pdb

    return query_sdf, reference_sdf, pocket_pdb


def _print_tensor_shape(prefix: str, payload: dict[str, Any], key: str) -> None:
    value = payload.get(key)
    if value is None:
        print(f"{prefix} {key}=MISSING")
        return

    if torch.is_tensor(value):
        print(f"{prefix} {key}={tuple(value.shape)} {value.dtype}")
    else:
        print(f"{prefix} {key}={value}")


def print_payload_summary(prefix: str, payload: dict[str, Any]) -> None:
    summary_keys = [
        "query_pos",
        "target_query_pos",
        "query_node_attr",
        "reference_pos",
        "reference_node_attr",
        "pocket_pos",
        "pocket_batch",
        "reference_center_subtracted",
    ]

    for key in summary_keys:
        _print_tensor_shape(prefix, payload, key)


def print_payload_keys(prefix: str, payload: dict[str, Any]) -> None:
    print(f"{prefix} keys:")
    for key in sorted(payload.keys()):
        value = payload[key]
        if torch.is_tensor(value):
            print(f"  - {key}: tensor shape={tuple(value.shape)} dtype={value.dtype}")
        else:
            print(f"  - {key}: {value}")


def compute_source_target_rmsd(payload: dict[str, Any]) -> float:
    query_pos = payload["query_pos"]
    target_query_pos = payload["target_query_pos"]

    if not torch.is_tensor(query_pos) or not torch.is_tensor(target_query_pos):
        raise TypeError("query_pos and target_query_pos must be torch.Tensor values.")

    rmsd = torch.sqrt(((query_pos - target_query_pos) ** 2).sum(dim=-1).mean())
    return float(rmsd)


def main() -> None:
    args = build_argparser().parse_args()

    repo_root = Path(args.repo_root).resolve()
    query_sdf, reference_sdf, pocket_pdb = _resolve_input_paths(args)

    payload = build_diffalign_example_train_payload(
        repo_root=repo_root,
        query_sdf=query_sdf,
        reference_sdf=reference_sdf,
        pocket_pdb=pocket_pdb,
        randomize_query_pos=not args.keep_query_conformer_as_source,
        seed=args.seed,
    )

    validate_diffalign_example_train_payload(payload)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)

    print(f"[prepare-train] saved: {out_path}")
    print_payload_summary("[prepare-train]", payload)
    print_payload_keys("[prepare-train]", payload)

    source_target_rmsd = compute_source_target_rmsd(payload)
    print(f"[prepare-train] source_target_rmsd: {source_target_rmsd:.6f} Å")


if __name__ == "__main__":
    main()