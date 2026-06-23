"""Isolating test for pocket-shape conditioning.

Same molecule M placed in K different orientations -> K complexes. The pocket of each
complex is the target shape itself (perfect complementary surface). Centroid-only pocket
info is identical across complexes, so a pocket-blind model cannot fit all K (same ligand
input, K different targets). A pocket-shape-conditioned model can.

Expect: pocket ON loss << pocket OFF loss.
"""
import torch

from etflowalign.dataset import collate_alignment
from etflowalign.flow_matching import (
    AlignmentFlowMatcher,
    FlowMatchingConfig,
    _axis_angle_to_matrix,
    flow_matching_step,
)
from etflowalign.model import AlignmentBatch, ETFlowAlignModel


def make_complexes(n=12, K=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(n, 3, generator=g) * 1.5            # one shared molecule
    atom_type = torch.randint(0, 16, (n,), generator=g)
    items = []
    for k in range(K):
        R = _axis_angle_to_matrix(torch.randn(3, generator=g))
        target = M @ R.transpose(0, 1)
        target = target - target.mean(0)                # centered bound pose for complex k
        # source: rigid-randomized version of the target (direction A)
        Rs = _axis_angle_to_matrix(torch.randn(3, generator=g))
        ts = torch.randn(3, generator=g) * 2.0
        source = target @ Rs.transpose(0, 1) + ts
        pocket = target.clone()                         # pocket == target shape (centered at origin)
        zb = torch.zeros(n, dtype=torch.long)
        b = AlignmentBatch(
            query_pos=source.float(), query_atom_type=atom_type, query_batch=zb,
            pocket_pos=pocket.float(), pocket_batch=zb,
        )
        items.append((b, target.float()))
    return items


def train(use_pocket: bool, items, steps=800, lr=1e-3):
    torch.manual_seed(0)
    model = ETFlowAlignModel(
        use_rigid_head=True, hidden_dim=128, num_blocks=4,
        use_pocket_conditioning=use_pocket, pocket_cutoff=20.0, pocket_max_neighbors=32,
    )
    matcher = AlignmentFlowMatcher(FlowMatchingConfig(path_type="rigid", source_type="input_query", sigma=0.0))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    batch, target = collate_alignment(items, conditioning="pocket")
    losses = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = flow_matching_step(model, matcher, batch, target_query_pos=target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        for p in model.parameters():
            if p.grad is not None:
                torch.nan_to_num_(p.grad, 0.0, 0.0, 0.0)
        opt.step()
        losses.append(float(loss.detach()))
    return min(losses[-50:]), losses[0]


def main():
    items = make_complexes()
    off_best, off_first = train(use_pocket=False, items=items)
    on_best, on_first = train(use_pocket=True, items=items)
    print(f"[pocket OFF] first={off_first:.4f}  best_last50={off_best:.4f}")
    print(f"[pocket ON ] first={on_first:.4f}  best_last50={on_best:.4f}")
    print(f"ratio (OFF/ON) = {off_best / max(on_best, 1e-9):.1f}x")
    assert on_best < 0.5 * off_best, "pocket conditioning did not help enough"
    print("POCKET CONDITIONING HELPS (ON << OFF)  OK")


if __name__ == "__main__":
    main()
