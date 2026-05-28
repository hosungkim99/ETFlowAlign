"""RDKit adapter utilities to build ETFlowAlign .pt payloads.

This module provides helper functions to convert RDKit molecules into
ETFlowAlign-compatible tensors, including optional node attributes.
"""

from __future__ import annotations

from typing import Any

import torch
from rdkit import Chem


def _atom_feature_matrix(mol: Chem.Mol) -> torch.Tensor:
    """Build node_attr matrix [N,5] as float32.

    Features per atom:
    [atomic_number, atom_degree, formal_charge, is_aromatic, is_in_ring]
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
    pos = []
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        pos.append([float(p.x), float(p.y), float(p.z)])
    return torch.tensor(pos, dtype=torch.float32)


def _atom_types(mol: Chem.Mol) -> torch.Tensor:
    return torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long)


def make_inference_payload(query_mol: Chem.Mol, reference_mol: Chem.Mol | None = None) -> dict[str, Any]:
    """Create an inference payload dict for torch.save.

    Includes query/reference node_attr when reference is provided.
    """
    payload: dict[str, Any] = {
        "query_pos": _positions_from_conformer(query_mol),
        "query_atom_type": _atom_types(query_mol),
        "query_batch": torch.zeros(query_mol.GetNumAtoms(), dtype=torch.long),
        "query_node_attr": _atom_feature_matrix(query_mol),
    }

    if reference_mol is not None:
        payload.update(
            {
                "reference_pos": _positions_from_conformer(reference_mol),
                "reference_atom_type": _atom_types(reference_mol),
                "reference_batch": torch.zeros(reference_mol.GetNumAtoms(), dtype=torch.long),
                "reference_node_attr": _atom_feature_matrix(reference_mol),
            }
        )

    return payload


def make_train_payload(query_mol: Chem.Mol, reference_mol: Chem.Mol, target_query_pos: torch.Tensor) -> dict[str, Any]:
    """Create a training payload dict for torch.save."""
    payload = make_inference_payload(query_mol=query_mol, reference_mol=reference_mol)
    payload["target_query_pos"] = target_query_pos.to(dtype=torch.float32)
    return payload
