"""DiffAlign-to-ETFlowAlign adapter utilities.

This module keeps backward-compatible adapter APIs used by smoke-test scripts,
and incrementally extends payloads with chemistry node attributes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from rdkit import Chem


def _atom_feature_matrix(mol: Chem.Mol) -> torch.Tensor:
    """Return node attributes [N,5] float32.

    Features: [atomic_number, atom_degree, formal_charge, is_aromatic, is_in_ring]
    """
    feats: list[list[float]] = []
    for atom in mol.GetAtoms():
        feats.append(
            [
                float(atom.GetAtomicNum()),
                float(atom.GetDegree()),
                float(atom.GetFormalCharge()),
                float(atom.GetIsAromatic()),
                float(atom.IsInRing()),
            ]
        )
    return torch.tensor(feats, dtype=torch.float32)


def _positions_from_conformer(mol: Chem.Mol) -> torch.Tensor:
    conf = mol.GetConformer()
    out = []
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        out.append([float(p.x), float(p.y), float(p.z)])
    return torch.tensor(out, dtype=torch.float32)


def _atom_types(mol: Chem.Mol) -> torch.Tensor:
    return torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long)


# ---- Backward-compatible public APIs ----

def load_rdkit_mol(path: str) -> Chem.Mol:
    """Load first molecule from an SDF file."""
    mols = [m for m in Chem.SDMolSupplier(path, removeHs=False) if m is not None]
    if not mols:
        raise ValueError(f"No molecule found in: {path}")
    return mols[0]


def shift_rdkit_mol_inplace(mol: Chem.Mol, shift_xyz: torch.Tensor) -> None:
    """In-place coordinate shift for RDKit conformer."""
    conf = mol.GetConformer()
    sx, sy, sz = [float(v) for v in shift_xyz]
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(float(p.x + sx), float(p.y + sy), float(p.z + sz)))


def mol_to_graph_data_obj_local(mol: Chem.Mol) -> dict[str, torch.Tensor]:
    """Return minimal graph tensors for one molecule with node_attr."""
    return {
        "pos": _positions_from_conformer(mol),
        "atom_type": _atom_types(mol),
        "node_attr": _atom_feature_matrix(mol),
    }


def diffalign_batches_to_etflowalign_payload(
    query_mol: Chem.Mol,
    reference_mol: Chem.Mol,
    target_query_pos: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Build ETFlowAlign payload while preserving legacy keys + adding node_attr."""
    payload: dict[str, Any] = {
        "query_pos": _positions_from_conformer(query_mol),
        "query_atom_type": _atom_types(query_mol),
        "query_batch": torch.zeros(query_mol.GetNumAtoms(), dtype=torch.long),
        "reference_pos": _positions_from_conformer(reference_mol),
        "reference_atom_type": _atom_types(reference_mol),
        "reference_batch": torch.zeros(reference_mol.GetNumAtoms(), dtype=torch.long),
        # keep pocket keys available for downstream scripts
        "pocket_pos": torch.empty(0, 3, dtype=torch.float32),
        "pocket_batch": torch.empty(0, dtype=torch.long),
        # issue #12 node attrs
        "query_node_attr": _atom_feature_matrix(query_mol),
        "reference_node_attr": _atom_feature_matrix(reference_mol),
    }
    if target_query_pos is not None:
        payload["target_query_pos"] = target_query_pos.to(dtype=torch.float32)
    return payload


def build_diffalign_example_inference_payload(repo_root: str) -> dict[str, Any]:
    """Build inference payload from bundled DiffAlign example files."""
    root = Path(repo_root)
    ex = root / "external" / "diffalign" / "diffalign" / "example"
    query = load_rdkit_mol(str(ex / "query.sdf"))
    reference = load_rdkit_mol(str(ex / "reference.sdf"))
    return diffalign_batches_to_etflowalign_payload(query_mol=query, reference_mol=reference)


# ---- Convenience wrappers retained from earlier incremental changes ----

def make_inference_payload(query_mol: Chem.Mol, reference_mol: Chem.Mol | None = None) -> dict[str, Any]:
    if reference_mol is None:
        payload = {
            "query_pos": _positions_from_conformer(query_mol),
            "query_atom_type": _atom_types(query_mol),
            "query_batch": torch.zeros(query_mol.GetNumAtoms(), dtype=torch.long),
            "query_node_attr": _atom_feature_matrix(query_mol),
        }
        return payload
    return diffalign_batches_to_etflowalign_payload(query_mol, reference_mol)


def make_train_payload(query_mol: Chem.Mol, reference_mol: Chem.Mol, target_query_pos: torch.Tensor) -> dict[str, Any]:
    return diffalign_batches_to_etflowalign_payload(query_mol, reference_mol, target_query_pos=target_query_pos)
