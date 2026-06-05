"""ETFlowAlign을 위한 다중 예제 데이터셋 및 콜레이션.

데이터셋은 복합체별 ``.pt`` 페이로드의 집합이다(``diffalign_adapter`` / PDBbind
추출기가 생성하는 동일한 스키마): 각 파일은 ``query_pos``, ``query_atom_type``,
``target_query_pos`` 및 선택적 ``reference_*`` / ``pocket_*`` / 결합 필드를
포함하는 하나의 복합체를 담는다.

이 모듈은 단일 복합체 페이로드를 모델이 이미 이해하는 배치형 멀티-그래프
``AlignmentBatch`` 객체로 변환하며, 동일한 데이터를 포켓 전용, 레퍼런스 전용,
또는 둘 다로 학습할 수 있는 컨디셔닝 스위치를 제공한다.
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
    """복합체별 ``.pt`` 페이로드 경로 목록에 대한 지연 로딩 데이터셋."""

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
            device=None,  # 콜레이션/디바이스 이동은 나중에 수행된다
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
    """페이로드 경로를 (train, val)로 결정론적 무작위 분할한다."""
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
    """모델이 선택된 신호로 학습하도록 reference 및/또는 pocket 필드를 마스킹한다."""
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
    """단일 복합체 예제들을 하나의 멀티-그래프 ``AlignmentBatch``로 합친다.

    각 입력 예제는 그래프 ``g`` (0..B-1)로 취급된다. 노드 텐서는 이어 붙이고;
    ``*_batch`` 인덱스와 ``query_bond_index``는 병합된 배치가 내부적으로
    일관성을 유지하도록 오프셋이 재조정된다.
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

    # 결합 인덱스 재매핑을 위한 그래프별 원자 오프셋
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

    # 결합: 각 그래프의 원자 인덱스에 오프셋을 적용한 후 이어 붙인다
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
    """``batch_size``개의 복합체를 비복원 추출로 무작위 선택하여 콜레이션한다."""
    n = len(dataset)
    k = min(batch_size, n)
    idx = torch.randperm(n, generator=generator)[:k].tolist()
    items = [dataset[i] for i in idx]
    return collate_alignment(items, conditioning=conditioning, device=device)