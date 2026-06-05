# etflowalign/diffalign_adapter.py
"""DiffAlign 스타일 분자 입력에서 ETFlowAlign .pt 페이로드로의 어댑터.

이 모듈은 의존성 경계를 단순하게 유지한다:
- DiffAlign은 mol_to_graph_data_obj를 통해 RDKit Mol -> PyG Data/Batch를 사용한다.
- ETFlowAlign은 etflowalign.data로 로드 가능한 순수 텐서 페이로드를 사용한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from rdkit import Chem
from rdkit.Geometry import Point3D
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.data import Data

def load_rdkit_mol(path: str | Path, *, sanitize: bool = True, remove_hs: bool = True) -> Chem.Mol:
    """파일 확장자에 따라 SDF/MOL/PDB에서 RDKit 분자를 로드한다."""
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
    """RDKit 컨포머 좌표를 Tensor[N,3]으로 변환한다."""
    conf = mol.GetConformer()
    coords = []

    for atom_idx in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(atom_idx)
        coords.append([pos.x, pos.y, pos.z])

    return torch.tensor(coords, dtype=torch.float32)

def mol_to_node_attr_local(mol: Chem.Mol) -> Tensor:
    """ETFlowAlign 노드 컨디셔닝을 위한 최소한의 RDKit 원자 특성을 구성한다.

    특성 순서:
        [atomic_number, atom_degree, formal_charge, is_aromatic, is_in_ring]
    """
    features = []
    for atom in mol.GetAtoms():
        features.append(
            [
                float(atom.GetAtomicNum()),
                float(atom.GetDegree()),
                float(atom.GetFormalCharge()),
                float(atom.GetIsAromatic()),
                float(atom.IsInRing()),
            ]
        )

    return torch.tensor(features, dtype=torch.float32)

def _bond_type_to_int(bond: Chem.Bond) -> int:
    bond_type = bond.GetBondType()

    if bond_type == Chem.rdchem.BondType.SINGLE:
        return 1
    if bond_type == Chem.rdchem.BondType.DOUBLE:
        return 2
    if bond_type == Chem.rdchem.BondType.TRIPLE:
        return 3
    if bond_type == Chem.rdchem.BondType.AROMATIC:
        return 12

    raise ValueError(f"Unsupported bond type: {bond_type}")

def mol_to_graph_data_obj_local(mol: Chem.Mol) -> Data:
    """DiffAlign의 RDKit Mol -> PyG Data 변환의 로컬 복사본.

    출력 필드:
        atom_type: Tensor[N], 원자 번호
        edge_index: Tensor[2, 2E], 방향 있는 결합 엣지
        edge_type: Tensor[2E], float 형태의 결합 타입
        pos: Tensor[N, 3], 컨포머 좌표
    """
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(atom.GetAtomicNum())

    atom_type = torch.tensor(atom_features, dtype=torch.long)

    edges = []
    bond_types = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = _bond_type_to_int(bond)

        edges.append((i, j))
        bond_types.append(bond_type)
        edges.append((j, i))
        bond_types.append(bond_type)

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(bond_types, dtype=torch.float32).view(-1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.float32)

    conf = mol.GetConformer()
    coordinates = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        coordinates.append([pos.x, pos.y, pos.z])

    pos = torch.tensor(coordinates, dtype=torch.float32)
    node_attr = mol_to_node_attr_local(mol)

    return Data(
        atom_type=atom_type,
        edge_index=edge_index,
        edge_type=edge_type,
        pos=pos,
        node_attr=node_attr,
    )
    
def mol_to_bond_index_and_length_local(mol: Chem.Mol, pos: Tensor) -> tuple[Tensor, Tensor]:
    """비방향 쿼리 결합 인덱스와 목표 결합 길이를 구성한다.

    Returns:
        bond_index: LongTensor[2, E], RDKit 결합당 하나의 비방향 엣지.
        bond_length: FloatTensor[E], 각 결합의 컨포머 결합 길이.
    """
    edges = []
    lengths = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        edges.append((i, j))
        lengths.append(torch.linalg.norm(pos[i] - pos[j]))

    if not edges:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=pos.dtype),
        )

    bond_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    bond_length = torch.stack(lengths).to(dtype=pos.dtype)

    return bond_index, bond_length    
    
def _make_cpu_generator(seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator

def _random_rotation_matrix(
    *,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> Tensor:
    """Sample a proper 3D rotation matrix on CPU."""
    q = torch.randn(4, dtype=dtype, generator=generator)
    q = q / q.norm().clamp_min(1e-8)

    w, x, y, z = q.unbind()

    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)]),
            torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)]),
            torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]),
        ],
        dim=0,
    )


def _make_query_source_pos(
    centered_query_pos: Tensor,
    *,
    randomize_query_pos: bool,
    seed: int | None,
    translation_scale: float = 1.0,
) -> Tensor:
    """Build source query coordinates without breaking ligand bond geometry.

    If randomize_query_pos=True, apply a random rigid transform to the centered
    query conformer. If False, return the centered query conformer itself.
    """
    if not randomize_query_pos:
        return centered_query_pos.clone()

    generator = _make_cpu_generator(seed)
    rotation = _random_rotation_matrix(
        dtype=centered_query_pos.dtype,
        generator=generator,
    )

    center = centered_query_pos.mean(dim=0, keepdim=True)
    translation = torch.randn(
        1,
        3,
        dtype=centered_query_pos.dtype,
        generator=generator,
    ) * float(translation_scale)

    return (centered_query_pos - center) @ rotation.T + center + translation

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
    if hasattr(query_batch, "node_attr"):
        payload["query_node_attr"] = query_batch.node_attr.detach().cpu().float()

    if hasattr(reference_batch, "node_attr"):
        payload["reference_node_attr"] = reference_batch.node_attr.detach().cpu().float()
        
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

    Coordinate convention:
    - reference coordinates are centered by reference mean.
    - pocket coordinates are shifted by the same reference mean.
    - query coordinates are in the same centered frame.
    - randomize_query_pos=True applies a rigid random transform to the centered
      query conformer, preserving bond geometry.
    """
    repo_root = Path(repo_root).resolve()

    query_mol = load_rdkit_mol(query_sdf, sanitize=True, remove_hs=True)
    reference_mol = load_rdkit_mol(reference_sdf, sanitize=True, remove_hs=True)

    pocket_mol = None
    if pocket_pdb is not None:
        pocket_mol = load_rdkit_mol(pocket_pdb, sanitize=True, remove_hs=True)

    query_data = mol_to_graph_data_obj_local(query_mol)
    reference_data = mol_to_graph_data_obj_local(reference_mol)

    reference_mean = reference_data.pos.mean(dim=0)
    centered_query_pos = query_data.pos.clone() - reference_mean
    reference_data.pos = reference_data.pos - reference_mean

    if pocket_mol is not None:
        pocket_mol = Chem.Mol(pocket_mol)
        shift_rdkit_mol_inplace(pocket_mol, reference_mean)

    query_batch = Batch.from_data_list([query_data])
    reference_batch = Batch.from_data_list([reference_data])
    query_batch.pos = _make_query_source_pos(
        centered_query_pos,
        randomize_query_pos=randomize_query_pos,
        seed=seed,
    )

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
            "query_source": "rigid_randomized_query_conformer"if randomize_query_pos 
            else "centered_query_conformer",
        },
    )

    return payload

def build_diffalign_example_train_payload(
    *,
    repo_root: str | Path,
    query_sdf: str | Path,
    reference_sdf: str | Path,
    pocket_pdb: str | Path | None = None,
    randomize_query_pos: bool = True,
    seed: int | None = 0,
) -> dict[str, Any]:
    """Build an ETFlowAlign pseudo-training payload from DiffAlign example files.

    Training target convention:
        target_query_pos = original_query_conformer - reference_center

    Source convention:
        query_pos = random rigid transform of target_query_pos
        if randomize_query_pos=True;
        otherwise query_pos = target_query_pos.
    """
    repo_root = Path(repo_root).resolve()

    query_mol = load_rdkit_mol(query_sdf, sanitize=True, remove_hs=True)
    reference_mol = load_rdkit_mol(reference_sdf, sanitize=True, remove_hs=True)

    pocket_mol = None
    if pocket_pdb is not None:
        pocket_mol = load_rdkit_mol(pocket_pdb, sanitize=True, remove_hs=True)

    query_data = mol_to_graph_data_obj_local(query_mol)
    reference_data = mol_to_graph_data_obj_local(reference_mol)

    reference_mean = reference_data.pos.mean(dim=0)

    # Important: target must be query-shaped [N_query, 3], not reference-shaped.
    target_query_pos = query_data.pos.clone() - reference_mean
    query_bond_index, query_bond_length = mol_to_bond_index_and_length_local(
    query_mol,
    target_query_pos,
    )
    # Put reference and pocket into the same centered coordinate frame.
    reference_data.pos = reference_data.pos - reference_mean

    if pocket_mol is not None:
        pocket_mol = Chem.Mol(pocket_mol)
        shift_rdkit_mol_inplace(pocket_mol, reference_mean)

    query_batch = Batch.from_data_list([query_data])
    reference_batch = Batch.from_data_list([reference_data])

    query_batch.pos = _make_query_source_pos(
        target_query_pos,
        randomize_query_pos=randomize_query_pos,
        seed=seed,
    )

    payload = diffalign_batches_to_etflowalign_payload(
        query_batch=query_batch,
        reference_batch=reference_batch,
        pocket_mol=pocket_mol,
        target_query_pos=target_query_pos,
        metadata={
            "source": "diffalign_example_pseudo_train",
            "query_sdf": str(query_sdf),
            "reference_sdf": str(reference_sdf),
            "pocket_pdb": str(pocket_pdb) if pocket_pdb is not None else None,
            "reference_center_subtracted": reference_mean.detach().cpu(),
            "target_definition": "query_sdf_conformer_minus_reference_center",
            "randomize_query_pos": bool(randomize_query_pos),
            "query_source": "rigid_randomized_query_conformer" if randomize_query_pos 
            else "centered_query_conformer",
            "seed": seed,
        },
    )
    
    payload["query_bond_index"] = query_bond_index.detach().cpu().long()
    payload["query_bond_length"] = query_bond_length.detach().cpu().float()
    
    return payload

def _require_tensor(payload: dict[str, object], key: str) -> torch.Tensor:
    """Return a required tensor field from a payload.

    This helper keeps CLI scripts short while making payload errors explicit.
    """
    if key not in payload:
        raise KeyError(f"Missing required payload key: {key}")

    value = payload[key]
    if not torch.is_tensor(value):
        raise TypeError(f"{key} must be a torch.Tensor, got {type(value)}")

    return value

def _validate_pos_tensor(name: str, pos: torch.Tensor) -> None:
    """Validate coordinate tensor shape [N, 3]."""
    if pos.dim() != 2 or pos.size(1) != 3:
        raise ValueError(f"{name} must have shape [N, 3], got {tuple(pos.shape)}")
    if pos.dtype not in (torch.float16, torch.float32, torch.float64):
        raise TypeError(f"{name} must be a floating tensor, got {pos.dtype}")

def _validate_batch_tensor(name: str, batch: torch.Tensor, pos: torch.Tensor) -> None:
    """Validate batch index tensor shape [N]."""
    if batch.dim() != 1:
        raise ValueError(f"{name} must have shape [N], got {tuple(batch.shape)}")
    if batch.size(0) != pos.size(0):
        raise ValueError(
            f"{name} and position tensor must have the same first dimension: "
            f"{tuple(batch.shape)} vs {tuple(pos.shape)}"
        )
    if batch.dtype != torch.long:
        raise TypeError(f"{name} must have dtype torch.long, got {batch.dtype}")

def _validate_atom_type_tensor(name: str, atom_type: torch.Tensor, pos: torch.Tensor) -> None:
    """Validate atom type tensor shape [N]."""
    if atom_type.dim() != 1:
        raise ValueError(f"{name} must have shape [N], got {tuple(atom_type.shape)}")
    if atom_type.size(0) != pos.size(0):
        raise ValueError(
            f"{name} and position tensor must have the same first dimension: "
            f"{tuple(atom_type.shape)} vs {tuple(pos.shape)}"
        )
    if atom_type.dtype != torch.long:
        raise TypeError(f"{name} must have dtype torch.long, got {atom_type.dtype}")

def _validate_node_attr_tensor(name: str, node_attr: torch.Tensor, pos: torch.Tensor) -> None:
    """Validate node attribute tensor shape [N, 5]."""
    if node_attr.dim() != 2:
        raise ValueError(f"{name} must have shape [N, F], got {tuple(node_attr.shape)}")
    if node_attr.size(0) != pos.size(0):
        raise ValueError(
            f"{name} and position tensor must have the same first dimension: "
            f"{tuple(node_attr.shape)} vs {tuple(pos.shape)}"
        )
    if node_attr.size(1) != 5:
        raise ValueError(f"{name} feature dimension must be 5, got {node_attr.size(1)}")
    if node_attr.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32, got {node_attr.dtype}")

def _validate_reference_center_metadata(payload: dict[str, object]) -> None:
    """Validate reference_center_subtracted metadata."""
    if "reference_center_subtracted" not in payload:
        raise KeyError(
            "Missing metadata key: reference_center_subtracted. "
            "DiffAlign example payloads should record the reference center used for centering."
        )

    center = payload["reference_center_subtracted"]
    if not torch.is_tensor(center):
        raise TypeError(
            "reference_center_subtracted must be a torch.Tensor, "
            f"got {type(center)}"
        )
    if center.shape != (3,):
        raise ValueError(
            "reference_center_subtracted must have shape [3], "
            f"got {tuple(center.shape)}"
        )


def validate_diffalign_example_inference_payload(payload: dict[str, object]) -> None:
    """Validate a DiffAlign-example inference payload.

    Required conventions:
    - query/reference coordinates use the same centered coordinate frame.
    - pocket coordinates are real coordinates from pocket.pdb, not an empty placeholder.
    - query_node_attr/reference_node_attr are RDKit chemistry features with shape [N, 5].
    """

    query_pos = _require_tensor(payload, "query_pos")
    query_atom_type = _require_tensor(payload, "query_atom_type")
    query_batch = _require_tensor(payload, "query_batch")
    query_node_attr = _require_tensor(payload, "query_node_attr")

    reference_pos = _require_tensor(payload, "reference_pos")
    reference_atom_type = _require_tensor(payload, "reference_atom_type")
    reference_batch = _require_tensor(payload, "reference_batch")
    reference_node_attr = _require_tensor(payload, "reference_node_attr")

    pocket_pos = _require_tensor(payload, "pocket_pos")
    pocket_batch = _require_tensor(payload, "pocket_batch")

    _validate_pos_tensor("query_pos", query_pos)
    _validate_pos_tensor("reference_pos", reference_pos)
    _validate_pos_tensor("pocket_pos", pocket_pos)

    _validate_atom_type_tensor("query_atom_type", query_atom_type, query_pos)
    _validate_atom_type_tensor("reference_atom_type", reference_atom_type, reference_pos)

    _validate_batch_tensor("query_batch", query_batch, query_pos)
    _validate_batch_tensor("reference_batch", reference_batch, reference_pos)
    _validate_batch_tensor("pocket_batch", pocket_batch, pocket_pos)

    _validate_node_attr_tensor("query_node_attr", query_node_attr, query_pos)
    _validate_node_attr_tensor("reference_node_attr", reference_node_attr, reference_pos)

    if pocket_pos.numel() == 0:
        raise ValueError(
            "pocket_pos is empty. DiffAlign example inference payload should include "
            "real pocket coordinates from pocket.pdb shifted by reference_center."
        )

    _validate_reference_center_metadata(payload)
    
def validate_diffalign_example_train_payload(payload: dict[str, object]) -> None:
    """Validate a DiffAlign-example pseudo-training payload.

    The train payload must satisfy all inference payload conventions and additionally
    include target_query_pos with the same shape as query_pos.
    """

    validate_diffalign_example_inference_payload(payload)

    query_pos = _require_tensor(payload, "query_pos")
    target_query_pos = _require_tensor(payload, "target_query_pos")

    _validate_pos_tensor("target_query_pos", target_query_pos)

    if query_pos.shape != target_query_pos.shape:
        raise ValueError(
            "query_pos and target_query_pos must have identical shape for flow matching. "
            f"query_pos={tuple(query_pos.shape)}, "
            f"target_query_pos={tuple(target_query_pos.shape)}. "
            "Expected target_query_pos = query_sdf_conformer - reference_center."
        )