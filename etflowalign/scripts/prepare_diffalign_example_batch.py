"""Prepare DiffAlign example inference batch (.pt) for ETFlowAlign smoke tests."""
from __future__ import annotations

import argparse
import torch

from etflowalign.diffalign_adapter import build_diffalign_example_inference_payload


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    payload = build_diffalign_example_inference_payload(args.repo_root)
    torch.save(payload, args.out)
    print(f"[prepare] saved: {args.out}")


if __name__ == "__main__":
    main()
