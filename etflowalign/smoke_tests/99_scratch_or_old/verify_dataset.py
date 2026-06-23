"""Verify multi-example dataset: collation offsets, conditioning, rigid training step."""
import tempfile
from pathlib import Path

import torch

from etflowalign.dataset import (
    AlignmentDataset,
    apply_conditioning,
    collate_alignment,
    sample_training_batch,
    train_val_split,
)
from etflowalign.flow_matching import (
    AlignmentFlowMatcher,
    FlowMatchingConfig,
    _axis_angle_to_matrix,
    flow_matching_step,
)
from etflowalign.model import ETFlowAlignModel


def make_complex_payload(n_atoms: int, n_pocket: int, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    conf = torch.randn(n_atoms, 3, generator=g) * 1.5
    target = conf - conf.mean(0)  # centered bound pose
    R = _axis_angle_to_matrix(torch.randn(3, generator=g))
    t = torch.randn(3, generator=g) * 3.0
    source = target @ R.transpose(0, 1) + t  # rigid-randomized source (same internal geometry)
    # simple chain bonds
    bi = torch.stack([torch.arange(n_atoms - 1), torch.arange(1, n_atoms)], dim=0)
    bl = torch.linalg.norm(target[bi[0]] - target[bi[1]], dim=-1)
    return {
        "query_pos": source.float(),
        "query_atom_type": torch.randint(0, 16, (n_atoms,), generator=g),
        "query_batch": torch.zeros(n_atoms, dtype=torch.long),
        "target_query_pos": target.float(),
        "pocket_pos": (torch.randn(n_pocket, 3, generator=g)).float(),
        "pocket_batch": torch.zeros(n_pocket, dtype=torch.long),
        "reference_pos": (torch.randn(n_atoms, 3, generator=g)).float(),
        "reference_atom_type": torch.randint(0, 16, (n_atoms,), generator=g),
        "reference_batch": torch.zeros(n_atoms, dtype=torch.long),
        "query_bond_index": bi.long(),
        "query_bond_length": bl.float(),
        "reference_center_subtracted": conf.mean(0).float(),
    }


def main():
    tmp = Path(tempfile.mkdtemp())
    sizes = [(8, 20), (12, 35), (10, 28), (15, 40), (9, 22)]
    for i, (na, npk) in enumerate(sizes):
        torch.save(make_complex_payload(na, npk, seed=i), tmp / f"cplx_{i:03d}.pt")

    ds = AlignmentDataset.from_directory(str(tmp), require_target=True)
    print(f"[dataset] {len(ds)} complexes")
    train, val = train_val_split(ds.paths, val_fraction=0.4, seed=0)
    print(f"[split] train={len(train)} val={len(val)}")

    # collation correctness
    batch, target = sample_training_batch(ds, batch_size=3, conditioning="both",
                                          generator=torch.Generator().manual_seed(1))
    nq = batch.query_pos.size(0)
    print(f"[collate both] Nq={nq} graphs={int(batch.query_batch.max())+1} "
          f"target={tuple(target.shape)} pocket={tuple(batch.pocket_pos.shape)} "
          f"ref={tuple(batch.reference_pos.shape)}")
    assert batch.query_batch.max().item() == 2
    assert target.size(0) == nq
    assert int(batch.query_bond_index.max()) < nq, "bond index offset out of range"
    # per-graph atom counts match query_batch
    counts = torch.bincount(batch.query_batch)
    print(f"[collate both] per-graph atom counts: {counts.tolist()}")

    # conditioning masks
    pocket_only = apply_conditioning(batch, "pocket")
    ref_only = apply_conditioning(batch, "reference")
    assert pocket_only.reference_pos is None and pocket_only.pocket_pos is not None
    assert ref_only.pocket_pos is None and ref_only.reference_pos is not None
    print("[conditioning] pocket-only drops reference; reference-only drops pocket  OK")

    # rigid training step on a pocket-conditioned multi-graph batch
    model = ETFlowAlignModel(use_rigid_head=True, hidden_dim=64, num_blocks=3)
    matcher = AlignmentFlowMatcher(FlowMatchingConfig(path_type="rigid", source_type="input_query", sigma=0.0))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    gen = torch.Generator().manual_seed(7)
    train_ds = AlignmentDataset(train, require_target=True)
    for step in range(60):
        b, tgt = sample_training_batch(train_ds, batch_size=2, conditioning="pocket", generator=gen)
        opt.zero_grad(set_to_none=True)
        loss = flow_matching_step(model, matcher, b, target_query_pos=tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        losses.append(float(loss.detach()))
    print(f"[train] pocket-conditioned multi-graph rigid step: loss {losses[0]:.4f} -> {min(losses[-10:]):.4f}")
    assert all(torch.isfinite(torch.tensor(l)) for l in losses)
    print("ALL DATASET CHECKS PASSED")


if __name__ == "__main__":
    main()
