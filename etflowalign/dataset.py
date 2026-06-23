"""Multi-example dataset + collation for ETFlowAlign.

A dataset is a set of per-complex ``.pt`` payloads (the same schema produced by
``diffalign_adapter`` / the PDBbind extractor): each holds one complex with
``query_pos``, ``query_atom_type``, ``target_query_pos`` and optional
``reference_*`` / ``pocket_*`` / bond fields.

This module turns those single-complex payloads into batched, multi-graph
``AlignmentBatch`` objects the model already understands, with a conditioning
switch so the same data can be trained pocket-only, reference-only, or both.
"""

from __future__ import annotations

import dataclasses
import glob
import os
from typing import Literal, Optional

import torch
from torch import Tensor

from .data import load_alignment_batch_from_pt
from .model import AlignmentBatch

Conditioning = Literal["pocket", "reference", "both"]


class AlignmentDataset:
    """Lazy dataset over a list of per-complex ``.pt`` payload paths."""

    def __init__(self, paths: list[str], require_target: bool = True) -> None:
        if not paths:
            raise ValueError("AlignmentDataset received an empty path list.")
        self.paths = list(paths)
        self.require_target = bool(require_target)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[AlignmentBatch, Optional[Tensor]]:
        batch, target, _ = load_alignment_batch_from_pt(
            self.paths[index],
            require_target=self.require_target,
            device=None,  # collation/move-to-device happens later
        )
        return batch, target

    @classmethod
    def from_directory(cls, root: str, pattern: str = "*.pt", require_target: bool = True) -> "AlignmentDataset":
        paths = sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))
        return cls(paths, require_target=require_target)


def train_val_split(
    paths: list[str],
    val_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[list[str], list[str]]:
    """Deterministic random split of payload paths into (train, val)."""
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}.")
    ordered = sorted(paths)
    generator = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(len(ordered), generator=generator).tolist()
    n_val = int(round(len(ordered) * val_fraction))
    val_idx = set(perm[:n_val])
    train = [ordered[i] for i in range(len(ordered)) if i not in val_idx]
    val = [ordered[i] for i in range(len(ordered)) if i in val_idx]
    return train, val


def apply_conditioning(batch: AlignmentBatch, conditioning: Conditioning) -> AlignmentBatch:
    """Mask reference and/or pocket fields so the model trains on the chosen signal."""
    drop: dict[str, None] = {}
    if conditioning == "pocket":
        drop = {"reference_pos": None, "reference_atom_type": None, "reference_batch": None, "reference_node_attr": None}
    elif conditioning == "reference":
        drop = {"pocket_pos": None, "pocket_batch": None, "pocket_atom_type": None}
    elif conditioning == "both":
        return batch
    else:
        raise ValueError(f"Unknown conditioning: {conditioning!r}")
    return dataclasses.replace(batch, **drop)


def _all_present(items: list[Optional[Tensor]]) -> bool:
    return all(x is not None and x.numel() > 0 for x in items)


def collate_alignment(
    items: list[tuple[AlignmentBatch, Optional[Tensor]]],
    conditioning: Conditioning = "both",
    device: str | torch.device | None = None,
) -> tuple[AlignmentBatch, Optional[Tensor]]:
    """Merge single-complex examples into one multi-graph ``AlignmentBatch``.

    Each input example is treated as graph ``g`` (0..B-1). Node tensors are
    concatenated; ``*_batch`` indices and ``query_bond_index`` are re-offset so
    the merged batch is internally consistent.
    """
    if not items:
        raise ValueError("collate_alignment received no items.")

    batches = [it[0] for it in items]
    targets = [it[1] for it in items]

    query_pos = torch.cat([b.query_pos for b in batches], dim=0)
    query_atom_type = torch.cat([b.query_atom_type for b in batches], dim=0)
    query_batch = torch.cat(
        [torch.full((b.query_pos.size(0),), g, dtype=torch.long) for g, b in enumerate(batches)],
        dim=0,
    )

    # atom offsets per graph for bond-index remapping
    atom_counts = [b.query_pos.size(0) for b in batches]
    atom_offsets = [0]
    for n in atom_counts[:-1]:
        atom_offsets.append(atom_offsets[-1] + n)

    def cat_node(field: str) -> Optional[Tensor]:
        vals = [getattr(b, field) for b in batches]
        return torch.cat(vals, dim=0) if _all_present(vals) else None

    def cat_with_batch(pos_field: str) -> tuple[Optional[Tensor], Optional[Tensor]]:
        vals = [getattr(b, pos_field) for b in batches]
        if not _all_present(vals):
            return None, None
        pos = torch.cat(vals, dim=0)
        idx = torch.cat(
            [torch.full((v.size(0),), g, dtype=torch.long) for g, v in enumerate(vals)],
            dim=0,
        )
        return pos, idx

    reference_pos, reference_batch = cat_with_batch("reference_pos")
    reference_atom_type = cat_node("reference_atom_type") if reference_pos is not None else None
    reference_node_attr = cat_node("reference_node_attr") if reference_pos is not None else None
    pocket_pos, pocket_batch = cat_with_batch("pocket_pos")
    pocket_atom_type = cat_node("pocket_atom_type") if pocket_pos is not None else None
    query_node_attr = cat_node("query_node_attr")

    # bonds: offset each graph's atom indices, then concatenate
    bond_indices = [b.query_bond_index for b in batches]
    bond_lengths = [b.query_bond_length for b in batches]
    if _all_present(bond_indices) and _all_present(bond_lengths):
        query_bond_index = torch.cat(
            [bi + atom_offsets[g] for g, bi in enumerate(bond_indices)], dim=1
        )
        query_bond_length = torch.cat(bond_lengths, dim=0)
    else:
        query_bond_index = None
        query_bond_length = None

    merged = AlignmentBatch(
        query_pos=query_pos,
        query_atom_type=query_atom_type,
        query_batch=query_batch,
        reference_pos=reference_pos,
        reference_atom_type=reference_atom_type,
        reference_batch=reference_batch,
        pocket_pos=pocket_pos,
        pocket_batch=pocket_batch,
        pocket_atom_type=pocket_atom_type,
        query_node_attr=query_node_attr,
        reference_node_attr=reference_node_attr,
        query_bond_index=query_bond_index,
        query_bond_length=query_bond_length,
    )
    merged = apply_conditioning(merged, conditioning)

    target = torch.cat(targets, dim=0) if _all_present(targets) else None

    if device is not None:
        merged = _batch_to_device(merged, device)
        if target is not None:
            target = target.to(device)
    return merged, target


def _batch_to_device(batch: AlignmentBatch, device: str | torch.device) -> AlignmentBatch:
    moved = {}
    for field in dataclasses.fields(batch):
        value = getattr(batch, field.name)
        moved[field.name] = value.to(device) if torch.is_tensor(value) else value
    return AlignmentBatch(**moved)


def sample_training_batch(
    dataset: AlignmentDataset,
    batch_size: int,
    conditioning: Conditioning,
    device: str | torch.device | None = None,
    generator: torch.Generator | None = None,
) -> tuple[AlignmentBatch, Optional[Tensor]]:
    """Randomly draw ``batch_size`` complexes and collate them (with replacement-free draw)."""
    n = len(dataset)
    k = min(batch_size, n)
    idx = torch.randperm(n, generator=generator)[:k].tolist()
    items = [dataset[i] for i in idx]
    return collate_alignment(items, conditioning=conditioning, device=device)