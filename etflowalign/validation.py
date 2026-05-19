"""Batch validation utilities for ETFlowAlign."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .model import AlignmentBatch


def _require_tensor(name: str, value: Optional[Tensor]) -> Tensor:
    if value is None:
        raise ValueError(f"Missing required field '{name}'.")
    if not torch.is_tensor(value):
        raise ValueError(f"Field '{name}' must be a torch.Tensor.")
    return value


def _check_shape(name: str, x: Tensor, dim1: Optional[int] = None, rank: Optional[int] = None) -> None:
    if rank is not None and x.dim() != rank:
        raise ValueError(f"Field '{name}' must have rank {rank}, got shape {tuple(x.shape)}.")
    if dim1 is not None and (x.dim() < 2 or x.size(1) != dim1):
        raise ValueError(f"Field '{name}' must have shape [N,{dim1}], got shape {tuple(x.shape)}.")


def validate_alignment_batch(
    batch: AlignmentBatch,
    target_query_pos: Tensor | None = None,
    require_reference: bool = False,
    require_pocket: bool = False,
    require_target: bool = False,
) -> None:
    """Validate batch structure and shapes before model/sampler execution."""
    qpos = _require_tensor("query_pos", batch.query_pos)
    qtype = _require_tensor("query_atom_type", batch.query_atom_type)
    qbatch = _require_tensor("query_batch", batch.query_batch)

    _check_shape("query_pos", qpos, dim1=3, rank=2)
    _check_shape("query_atom_type", qtype, rank=1)
    _check_shape("query_batch", qbatch, rank=1)
    if qpos.size(0) != qtype.size(0) or qpos.size(0) != qbatch.size(0):
        raise ValueError("query_pos, query_atom_type, query_batch must share first dimension Nq.")
    if not torch.is_floating_point(qpos):
        raise ValueError("query_pos must be floating point.")
    if qtype.dtype not in (torch.int32, torch.int64):
        raise ValueError("query_atom_type must be int32 or int64.")
    if qbatch.dtype not in (torch.int32, torch.int64):
        raise ValueError("query_batch must be int32 or int64.")

    if require_reference or batch.reference_pos is not None or batch.reference_batch is not None:
        rpos = _require_tensor("reference_pos", batch.reference_pos)
        rbatch = _require_tensor("reference_batch", batch.reference_batch)
        _check_shape("reference_pos", rpos, dim1=3, rank=2)
        _check_shape("reference_batch", rbatch, rank=1)
        if rpos.size(0) != rbatch.size(0):
            raise ValueError("reference_pos and reference_batch must share first dimension Nr.")

    pocket_pos = batch.pocket_pos
    pocket_batch = getattr(batch, "pocket_batch", None)
    if require_pocket or pocket_pos is not None or pocket_batch is not None:
        ppos = _require_tensor("pocket_pos", pocket_pos)
        pbatch = _require_tensor("pocket_batch", pocket_batch)
        _check_shape("pocket_pos", ppos, dim1=3, rank=2)
        _check_shape("pocket_batch", pbatch, rank=1)
        if ppos.size(0) != pbatch.size(0):
            raise ValueError("pocket_pos and pocket_batch must share first dimension Np.")

    if require_target:
        target_query_pos = _require_tensor("target_query_pos", target_query_pos)
    if target_query_pos is not None:
        _check_shape("target_query_pos", target_query_pos, dim1=3, rank=2)
        if target_query_pos.size(0) != qpos.size(0):
            raise ValueError("target_query_pos must have same first dimension as query_pos (Nq).")
