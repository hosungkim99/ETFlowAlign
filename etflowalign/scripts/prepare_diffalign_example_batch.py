# scripts/prepare_diffalign_example_batch.py
"""Create an ETFlowAlign .pt inference batch from the bundled DiffAlign example.

Example:
    python scripts/prepare_diffalign_example_batch.py \
      --repo-root /home/deepfold/users/hosung/work/ETFlowAlign \
      --out /home/deepfold/users/hosung/work/ETFlowAlign/etflowalign/smoke_tests/diffalign_example_infer.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from etflowalign.diffalign_adapter import build_diffalign_example_inference_payload


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert DiffAlign example query/reference/pocket into ETFlowAlign .pt input."
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
        "--keep-query-conformer",
        action="store_true",
        help="Use query SDF conformer coordinates instead of random N(0,I) coordinates.",
    )

    return parser


def main() -> None:
    args = build_argparser().parse_args()

    repo_root = Path(args.repo_root).resolve()
    query_sdf = Path(args.query_sdf) if args.query_sdf else repo_root / "external" / "diffalign" / "diffalign" / "example" / "query.sdf"
    reference_sdf = Path(args.reference_sdf) if args.reference_sdf else repo_root / "external" / "diffalign" / "diffalign" / "example" / "reference.sdf"
    pocket_pdb = Path(args.pocket_pdb) if args.pocket_pdb else repo_root / "external" / "diffalign" / "diffalign" / "example" / "pocket.pdb"

    payload = build_diffalign_example_inference_payload(
        repo_root=repo_root,
        query_sdf=query_sdf,
        reference_sdf=reference_sdf,
        pocket_pdb=pocket_pdb,
        randomize_query_pos=not args.keep_query_conformer,
        seed=args.seed,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)

    print(f"[prepare] saved: {out_path}")
    print(f"[prepare] query_pos={tuple(payload['query_pos'].shape)}")
    print(f"[prepare] reference_pos={tuple(payload['reference_pos'].shape)}")

    if "pocket_pos" in payload:
        print(f"[prepare] pocket_pos={tuple(payload['pocket_pos'].shape)}")

    print("[prepare] keys:")
    for key in sorted(payload.keys()):
        value = payload[key]
        if torch.is_tensor(value):
            print(f"  - {key}: tensor shape={tuple(value.shape)} dtype={value.dtype}")
        else:
            print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()
