"""I/O helpers for loading ETFlowAlign batches from .pt files."""
from __future__ import annotations

from typing import Any

import torch

from .model import AlignmentBatch
from .validation import validate_alignment_batch


def load_alignment_batch_from_pt(path: str, require_target: bool, device: str | torch.device | None = None) -> tuple[AlignmentBatch, torch.Tensor | None, dict[str, Any]]:
    raw = torch.load(path, map_location=device)
    if not isinstance(raw, dict):
        raise ValueError("Expected .pt file to contain a dict payload.")

    batch = AlignmentBatch(
        query_pos=raw.get("query_pos"),
        query_atom_type=raw.get("query_atom_type"),
        query_batch=raw.get("query_batch"),
        reference_pos=raw.get("reference_pos"),
        reference_atom_type=raw.get("reference_atom_type"),
        reference_batch=raw.get("reference_batch"),
        pocket_pos=raw.get("pocket_pos"),
        pocket_batch=raw.get("pocket_batch"),
        query_node_attr=raw.get("query_node_attr"),
        reference_node_attr=raw.get("reference_node_attr"),
    )
    target = raw.get("target_query_pos")

    validate_alignment_batch(batch, target_query_pos=target, require_target=require_target)
    metadata = {k: v for k, v in raw.items() if k not in {
        "query_pos", "query_atom_type", "query_batch", "reference_pos", "reference_atom_type", "reference_batch",
        "pocket_pos", "pocket_batch", "query_node_attr", "reference_node_attr", "target_query_pos"
    }}
    return batch, target, metadata
