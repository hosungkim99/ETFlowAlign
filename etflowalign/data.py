"""I/O helpers for loading ETFlowAlign batches from .pt files."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from .model import AlignmentBatch
from .validation import validate_alignment_batch


def _to_device(x: Tensor | None, device: torch.device | str | None) -> Tensor | None:
    if x is None:
        return None
    return x.to(device=device) if device is not None else x


def load_alignment_batch_from_pt(
    path: str,
    require_target: bool,
    device: str | torch.device | None = None,
) -> tuple[AlignmentBatch, torch.Tensor | None, dict[str, Any]]:
    """Load a serialized ETFlowAlign batch and normalize tensor types/devices."""
    raw = torch.load(path, map_location=device)
    if not isinstance(raw, dict):
        raise ValueError("Expected .pt file to contain a dict payload.")

    query_pos = _to_device(raw.get("query_pos"), device)
    reference_pos = _to_device(raw.get("reference_pos"), device)
    pocket_pos = _to_device(raw.get("pocket_pos"), device)
    target_query_pos = _to_device(raw.get("target_query_pos"), device)

    query_atom_type = _to_device(raw.get("query_atom_type"), device)
    if query_atom_type is not None:
        query_atom_type = query_atom_type.long()
    reference_atom_type = _to_device(raw.get("reference_atom_type"), device)
    if reference_atom_type is not None:
        reference_atom_type = reference_atom_type.long()

    query_batch = _to_device(raw.get("query_batch"), device)
    if query_batch is not None:
        query_batch = query_batch.long()
    reference_batch = _to_device(raw.get("reference_batch"), device)
    if reference_batch is not None:
        reference_batch = reference_batch.long()
    pocket_batch = _to_device(raw.get("pocket_batch"), device)
    if pocket_batch is not None:
        pocket_batch = pocket_batch.long()

    query_node_attr = _to_device(raw.get("query_node_attr"), device)
    reference_node_attr = _to_device(raw.get("reference_node_attr"), device)

    batch = AlignmentBatch(
        query_pos=query_pos,
        query_atom_type=query_atom_type,
        query_batch=query_batch,
        reference_pos=reference_pos,
        reference_atom_type=reference_atom_type,
        reference_batch=reference_batch,
        pocket_pos=pocket_pos,
        pocket_batch=pocket_batch,
        query_node_attr=query_node_attr,
        reference_node_attr=reference_node_attr,
    )

    validate_alignment_batch(batch, target_query_pos=target_query_pos, require_target=require_target)

    core_keys = {
        "query_pos",
        "query_atom_type",
        "query_batch",
        "reference_pos",
        "reference_atom_type",
        "reference_batch",
        "pocket_pos",
        "pocket_batch",
        "query_node_attr",
        "reference_node_attr",
        "target_query_pos",
    }
    metadata = {k: v for k, v in raw.items() if k not in core_keys}
    return batch, target_query_pos, metadata
