# etflowalign/diffalign_adapter.py
"""Adapters from DiffAlign-style molecular inputs to ETFlowAlign .pt payloads.

This module intentionally keeps the dependency boundary simple:
- DiffAlign uses RDKit Mol -> PyG Data/Batch via mol_to_graph_data_obj.
- ETFlowAlign uses plain tensor payloads loadable by etflowalign.data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from rdkit import Chem
from rdkit.Geometry import Point3D
from torch import Tensor
from torch_geometric.data import Batch


def add_diffalign_to_pythonpath(repo_root: str | Path) -> None:
    """Add external DiffAlign package path to sys.path if needed."""
    import sys

    repo_root = Path(repo_root).resolve()
    diffalign_pkg_root = repo_root / "external" / "diffalign" / "diffalign"

    if not diffalign_pkg_root.exists():
        raise FileNotFoundError(f"DiffAlign package root not found: {diffalign_pkg_root}")

    path_str = str(diffalign_pkg_root)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def load_rdkit_mol(path: str | Path, *, sanitize: bool = True, remove_hs: bool = True) -> Chem.Mol:
    """Load RDKit molecule from SDF/MOL/PDB based on file extension."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".sdf", ".mol"}:
        supplier = Chem.SDMolSupplier(str(path), sanitize=sanitize, removeHs=remove_hs)
        mol = supplier[0] if supplier and len(supplier) > 0 else None
    elif suffix == ".pdb":
        mol = Chem.MolFromPDBFile(str(path), sanitize=sanitize, removeHs=remove_hs)
    else:
        raise ValueError(f"Unsupported molecule file extension: {path.suffix}")

    if mol is None:
        raise ValueError(f"Failed to load molecule: {path}")

    if mol.GetNumConformers() == 0:
        raise ValueError(f"Molecule has no conformer coordinates: {path}")

    return mol


def rdkit_mol_to_pos(mol: Chem.Mol) -> Tensor:
    """Convert RDKit conformer coordinates to Tensor[N,3]."""
    conf = mol.GetConformer()
    coords = []

    for atom_idx in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(atom_idx)
        coords.append([pos.x, pos.y, pos.z])

    return torch.tensor(coords, dtype=torch.float32)


def shift_rdkit_mol_inplace(mol: Chem.Mol, shift: Tensor) -> Chem.Mol:
    """Shift RDKit conformer coordinates in-place by subtracting shift."""
    shift = shift.detach().cpu().float()
    shift_vec = Point3D(float(shift[0]), float(shift[1]), float(shift[2]))

    conf = mol.GetConformer()
    for atom_idx in range(mol.GetNumAtoms()):
        old = conf.GetAtomPosition(atom_idx)
        conf.SetAtomPosition(atom_idx, old - shift_vec)

    return mol


def diffalign_batches_to_etflowalign_payload(
    query_batch: Batch,
    reference_batch: Batch,
    *,
    pocket_mol: Chem.Mol | None = None,
    target_query_pos: Tensor | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert DiffAlign PyG Batch objects into an ETFlowAlign tensor payload.

    Args:
        query_batch:
            DiffAlign query ligand PyG Batch.
        reference_batch:
            DiffAlign reference ligand PyG Batch.
        pocket_mol:
            Optional RDKit pocket molecule. Coordinates should already be in the
            same coordinate frame as query/reference.
        target_query_pos:
            Optional target query coordinates for training. Inference payloads
            should omit this field.
        metadata:
            Optional extra metadata stored as top-level keys in the .pt payload.

    Returns:
        Dict that can be saved by torch.save and loaded with
        etflowalign.data.load_alignment_batch_from_pt.
    """
    required_query_fields = ["pos", "atom_type", "batch"]
    required_reference_fields = ["pos", "atom_type", "batch"]

    for name in required_query_fields:
        if not hasattr(query_batch, name):
            raise ValueError(f"query_batch is missing required field '{name}'")

    for name in required_reference_fields:
        if not hasattr(reference_batch, name):
            raise ValueError(f"reference_batch is missing required field '{name}'")

    payload: dict[str, Any] = {
        "query_pos": query_batch.pos.detach().cpu().float(),
        "query_atom_type": query_batch.atom_type.detach().cpu().long(),
        "query_batch": query_batch.batch.detach().cpu().long(),
        "reference_pos": reference_batch.pos.detach().cpu().float(),
        "reference_atom_type": reference_batch.atom_type.detach().cpu().long(),
        "reference_batch": reference_batch.batch.detach().cpu().long(),
    }

    if target_query_pos is not None:
        if target_query_pos.shape != payload["query_pos"].shape:
            raise ValueError(
                "target_query_pos must match query_pos shape. "
                f"target={tuple(target_query_pos.shape)}, query={tuple(payload['query_pos'].shape)}"
            )
        payload["target_query_pos"] = target_query_pos.detach().cpu().float()

    if pocket_mol is not None:
        pocket_pos = rdkit_mol_to_pos(pocket_mol)
        payload["pocket_pos"] = pocket_pos
        payload["pocket_batch"] = torch.zeros(pocket_pos.size(0), dtype=torch.long)

    if metadata:
        payload.update(metadata)

    return payload


def build_diffalign_example_inference_payload(
    *,
    repo_root: str | Path,
    query_sdf: str | Path,
    reference_sdf: str | Path,
    pocket_pdb: str | Path | None = None,
    randomize_query_pos: bool = True,
    seed: int | None = 0,
) -> dict[str, Any]:
    """Build an ETFlowAlign inference payload from DiffAlign example files.

    This follows DiffAlign alignment_example.ipynb:
    - reference coordinates are centered by reference mean.
    - pocket coordinates are shifted by the same reference mean.
    - query coordinates may be randomized with N(0,I).
    """
    repo_root = Path(repo_root).resolve()
    add_diffalign_to_pythonpath(repo_root)

    from diffalign.utils.chem import mol_to_graph_data_obj

    query_mol = load_rdkit_mol(query_sdf, sanitize=True, remove_hs=True)
    reference_mol = load_rdkit_mol(reference_sdf, sanitize=True, remove_hs=True)

    pocket_mol = None
    if pocket_pdb is not None:
        pocket_mol = load_rdkit_mol(pocket_pdb, sanitize=True, remove_hs=True)

    query_data = mol_to_graph_data_obj(query_mol)
    reference_data = mol_to_graph_data_obj(reference_mol)

    reference_mean = reference_data.pos.mean(dim=0)
    reference_data.pos = reference_data.pos - reference_mean

    if pocket_mol is not None:
        pocket_mol = Chem.Mol(pocket_mol)
        shift_rdkit_mol_inplace(pocket_mol, reference_mean)

    query_batch = Batch.from_data_list([query_data])
    reference_batch = Batch.from_data_list([reference_data])

    if randomize_query_pos:
        if seed is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed))
            query_batch.pos = torch.randn(
                query_batch.pos.shape,
                generator=generator,
                dtype=query_batch.pos.dtype,
            )
        else:
            query_batch.pos = torch.randn_like(query_batch.pos)

    payload = diffalign_batches_to_etflowalign_payload(
        query_batch=query_batch,
        reference_batch=reference_batch,
        pocket_mol=pocket_mol,
        target_query_pos=None,
        metadata={
            "source": "diffalign_example",
            "query_sdf": str(query_sdf),
            "reference_sdf": str(reference_sdf),
            "pocket_pdb": str(pocket_pdb) if pocket_pdb is not None else None,
            "reference_center_subtracted": reference_mean.detach().cpu(),
            "randomize_query_pos": bool(randomize_query_pos),
            "seed": seed,
        },
    )

    return payload
