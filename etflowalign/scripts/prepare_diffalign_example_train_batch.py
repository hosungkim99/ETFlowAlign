"""Prepare DiffAlign example train batch (.pt) for ETFlowAlign smoke tests."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from etflowalign.diffalign_adapter import diffalign_batches_to_etflowalign_payload, load_rdkit_mol


def _pos(mol):
    conf = mol.GetConformer()
    out = []
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        out.append([float(p.x), float(p.y), float(p.z)])
    return torch.tensor(out, dtype=torch.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ex = Path(args.repo_root) / "external" / "diffalign" / "diffalign" / "example"
    query = load_rdkit_mol(str(ex / "query.sdf"))
    reference = load_rdkit_mol(str(ex / "reference.sdf"))

    # For example train payload, use reference geometry as alignment target proxy.
    target_query_pos = _pos(reference)
    payload = diffalign_batches_to_etflowalign_payload(
        query_mol=query,
        reference_mol=reference,
        target_query_pos=target_query_pos,
    )
    torch.save(payload, args.out)
    print(f"[prepare] saved: {args.out}")


if __name__ == "__main__":
    main()
