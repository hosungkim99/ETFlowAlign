# etflowalign/scripts/prepare_pdbbind_dataset.py
"""Build an ETFlowAlign pocket-conditioned dataset from PDBbind refined.

Each PDBbind complex dir ``<id>/`` provides ``<id>_ligand.sdf`` (bound pose) and
``<id>_pocket.pdb`` (binding-site atoms). For each complex we emit one per-complex
``.pt`` payload (same schema as diffalign_adapter), pocket-conditioned:

    target_query_pos = ligand bound conformer - pocket_center
    query_pos        = random rigid transform of target  (direction A source)
    pocket_pos       = pocket atoms - pocket_center
    reference_*      = absent  (pocket conditioning)

Failures (unreadable ligand, etc.) are skipped and logged; a manifest.csv lists
the successful complexes for AlignmentDataset / train_val_split.

Example:
    python -m etflowalign.scripts.prepare_pdbbind_dataset \
      --pdbbind-root /home/.../dataset/PDBBind20_refined/P-L \
      --out-dir /home/.../smoke_tests/00_inputs/pdbbind_refined_pt \
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
from rdkit import Chem

from etflowalign.diffalign_adapter import (
    _make_query_source_pos,
    mol_to_bond_index_and_length_local,
    mol_to_node_attr_local,
)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build pocket-conditioned ETFlowAlign dataset from PDBbind.")
    p.add_argument("--pdbbind-root", required=True, help="PDBbind P-L root (contains year/<id>/ dirs).")
    p.add_argument("--out-dir", required=True, help="Output directory for per-complex .pt payloads.")
    p.add_argument("--manifest", default="", help="Manifest CSV path. Default: <out-dir>/manifest.csv")
    p.add_argument("--seed", type=int, default=0, help="Base seed for the rigid source transform.")
    p.add_argument("--no-randomize-source", action="store_true",
                   help="Use the centered ligand conformer as source instead of a rigid-randomized one.")
    p.add_argument("--limit", type=int, default=0, help="Process at most N complexes (0 = all).")
    p.add_argument("--keep-pocket-hs", action="store_true", help="Keep hydrogens in the pocket coordinates.")
    return p


def find_complexes(root: str) -> list[tuple[str, str]]:
    """Return (pdb_id, complex_dir) for every ``*_pocket.pdb`` found under root."""
    out = []
    for pocket in sorted(glob.glob(os.path.join(root, "**", "*_pocket.pdb"), recursive=True)):
        cdir = os.path.dirname(pocket)
        pdb_id = os.path.basename(pocket)[: -len("_pocket.pdb")]
        out.append((pdb_id, cdir))
    return out


def load_ligand(complex_dir: str, pdb_id: str) -> Chem.Mol:
    """Load the bound ligand, trying SDF (sanitized, then unsanitized), then MOL2."""
    sdf = os.path.join(complex_dir, f"{pdb_id}_ligand.sdf")
    mol2 = os.path.join(complex_dir, f"{pdb_id}_ligand.mol2")

    for sanitize in (True, False):
        if os.path.exists(sdf):
            supplier = Chem.SDMolSupplier(sdf, sanitize=sanitize, removeHs=True)
            mol = supplier[0] if supplier is not None and len(supplier) > 0 else None
            if mol is not None and mol.GetNumConformers() > 0:
                return mol
    if os.path.exists(mol2):
        mol = Chem.MolFromMol2File(mol2, sanitize=True, removeHs=True)
        if mol is None:
            mol = Chem.MolFromMol2File(mol2, sanitize=False, removeHs=True)
        if mol is not None and mol.GetNumConformers() > 0:
            return mol

    raise ValueError(f"Could not load a ligand conformer for {pdb_id}")


def parse_pocket_coords(pdb_path: str, drop_h: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Parse heavy-atom coordinates + atomic numbers from a pocket PDB (sanitization-free).

    Returns (coords[Np,3] float, atom_type[Np] long atomic numbers).
    """
    periodic = Chem.GetPeriodicTable()
    coords: list[list[float]] = []
    atom_types: list[int] = []
    with open(pdb_path, "r") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            element = line[76:78].strip()
            atom_name = line[12:16].strip()
            symbol = element if element else atom_name[:1]
            if drop_h and symbol.upper().startswith("H"):
                continue
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except ValueError:
                continue
            sym = symbol[0].upper() + symbol[1:].lower() if len(symbol) > 1 else symbol.upper()
            try:
                z_num = periodic.GetAtomicNumber(sym)
            except Exception:  # noqa: BLE001
                z_num = 0
            coords.append([x, y, z])
            atom_types.append(z_num if z_num > 0 else 6)  # default unknown -> carbon
    if not coords:
        raise ValueError(f"No atom coordinates parsed from pocket: {pdb_path}")
    return torch.tensor(coords, dtype=torch.float32), torch.tensor(atom_types, dtype=torch.long)


def ligand_atom_types_and_pos(mol: Chem.Mol) -> tuple[torch.Tensor, torch.Tensor]:
    atom_types = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long)
    conf = mol.GetConformer()
    pos = torch.tensor(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
         for i in range(mol.GetNumAtoms())],
        dtype=torch.float32,
    )
    return atom_types, pos


def build_payload(complex_dir: str, pdb_id: str, *, seed: int, randomize: bool, keep_pocket_hs: bool) -> dict:
    mol = load_ligand(complex_dir, pdb_id)
    atom_types, ligand_pos = ligand_atom_types_and_pos(mol)
    pocket_pos, pocket_atom_type = parse_pocket_coords(
        os.path.join(complex_dir, f"{pdb_id}_pocket.pdb"), drop_h=not keep_pocket_hs
    )
    
    pocket_center = pocket_pos.mean(dim=0)
    target_query_pos = ligand_pos - pocket_center
    pocket_centered = pocket_pos - pocket_center

    query_pos = _make_query_source_pos(target_query_pos, randomize_query_pos=randomize, seed=seed)
    bond_index, bond_length = mol_to_bond_index_and_length_local(mol, target_query_pos)
    node_attr = mol_to_node_attr_local(mol)

    n = ligand_pos.size(0)
    return {
        "query_pos": query_pos.float(),
        "query_atom_type": atom_types.long(),
        "query_batch": torch.zeros(n, dtype=torch.long),
        "target_query_pos": target_query_pos.float(),
        "query_node_attr": node_attr.float(),
        "query_bond_index": bond_index.long(),
        "query_bond_length": bond_length.float(),
        "pocket_pos": pocket_centered.float(),
        "pocket_batch": torch.zeros(pocket_centered.size(0), dtype=torch.long),
        "pocket_atom_type": pocket_atom_type.long(),
        "reference_center_subtracted": pocket_center.float(),
        "source": "pdbbind_refined",
        "pdb_id": pdb_id,
        "query_source": "rigid_randomized_ligand_conformer" if randomize else "centered_ligand_conformer",
        "conditioning": "pocket",
        "seed": seed,
    }


def main() -> None:
    args = build_argparser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else out_dir / "manifest.csv"

    complexes = find_complexes(args.pdbbind_root)
    if args.limit > 0:
        complexes = complexes[: args.limit]
    print(f"[pdbbind] found {len(complexes)} complexes under {args.pdbbind_root}")

    rows = []
    n_ok = n_fail = 0
    with open(out_dir / "failures.log", "w") as flog:
        for i, (pdb_id, cdir) in enumerate(complexes):
            try:
                payload = build_payload(
                    cdir, pdb_id,
                    seed=args.seed + i,
                    randomize=not args.no_randomize_source,
                    keep_pocket_hs=args.keep_pocket_hs,
                )
                out_path = out_dir / f"{pdb_id}.pt"
                torch.save(payload, out_path)
                rows.append({
                    "pdb_id": pdb_id,
                    "path": str(out_path),
                    "n_atoms": int(payload["query_pos"].size(0)),
                    "n_pocket": int(payload["pocket_pos"].size(0)),
                })
                n_ok += 1
            except Exception as exc:  # noqa: BLE001 - dataset prep must be robust to bad complexes
                n_fail += 1
                flog.write(f"{pdb_id}\t{exc}\n{traceback.format_exc()}\n")
            if (i + 1) % 200 == 0:
                print(f"[pdbbind] {i + 1}/{len(complexes)}  ok={n_ok} fail={n_fail}")

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pdb_id", "path", "n_atoms", "n_pocket"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[pdbbind] done: ok={n_ok} fail={n_fail}")
    print(f"[pdbbind] manifest: {manifest_path}")
    print(f"[pdbbind] failures log: {out_dir / 'failures.log'}")


if __name__ == "__main__":
    main()
