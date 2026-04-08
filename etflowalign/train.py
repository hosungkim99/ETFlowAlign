"""Training script and utilities for ETFlowAlign.

Usage example:
    python -m etflowalign.train --steps 200 --batch-size 8 --save-path etflowalign_ckpt.pt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from torch import optim

from .flow_matching import AlignmentFlowMatcher, FlowMatchingConfig, flow_matching_step
from .model import AlignmentBatch, ETFlowAlignModel


@dataclass
class TrainConfig:
    lr: float = 1e-4
    weight_decay: float = 0.0


def build_training_components(model: ETFlowAlignModel, train_config: TrainConfig, fm_config: FlowMatchingConfig):
    matcher = AlignmentFlowMatcher(config=fm_config)
    optimizer = optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
    return matcher, optimizer


def train_step(
    model: ETFlowAlignModel,
    matcher: AlignmentFlowMatcher,
    optimizer: optim.Optimizer,
    batch: AlignmentBatch,
    target_query_pos: torch.Tensor,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = flow_matching_step(model=model, matcher=matcher, batch=batch, target_query_pos=target_query_pos)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    return float(loss.detach().item())


def make_synthetic_alignment_batch(batch_size: int, n_atoms: int, device: torch.device) -> tuple[AlignmentBatch, torch.Tensor]:
    """Create a toy alignment dataset for smoke-test training.

    target_query_pos: reference + smooth deformation + translation.
    input query_pos in batch is initialized as noisy source state.
    """
    query_batch = torch.arange(batch_size, device=device).repeat_interleave(n_atoms)
    reference_batch = query_batch.clone()

    # Base molecular-like cloud around origin per graph.
    ref = 0.7 * torch.randn(batch_size * n_atoms, 3, device=device)
    atom_type = torch.randint(low=0, high=16, size=(batch_size * n_atoms,), device=device)

    # Construct alignment target from reference with global transform + local deformation.
    t = torch.randn(batch_size, 3, device=device) * 0.25
    target = ref.clone()
    target = target + t[query_batch]
    target = target + 0.05 * torch.sin(target * 2.0)

    # Initial query is coarse/un-aligned state.
    query_init = ref + 0.8 * torch.randn_like(ref)

    batch = AlignmentBatch(
        query_pos=query_init,
        query_atom_type=atom_type,
        query_batch=query_batch,
        reference_pos=ref,
        reference_atom_type=atom_type,
        reference_batch=reference_batch,
        pocket_pos=None,
    )
    return batch, target


def run_training(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    model = ETFlowAlignModel(hidden_dim=args.hidden_dim, num_blocks=args.num_blocks).to(device)

    fm_config = FlowMatchingConfig(
        sigma=args.sigma,
        source_type=args.source_type,
        source_noise_scale=args.source_noise_scale,
    )
    train_config = TrainConfig(lr=args.lr, weight_decay=args.weight_decay)
    matcher, optimizer = build_training_components(model, train_config, fm_config)

    for step in range(1, args.steps + 1):
        batch, target = make_synthetic_alignment_batch(args.batch_size, args.n_atoms, device)
        loss = train_step(model, matcher, optimizer, batch, target)
        if step % args.log_every == 0 or step == 1:
            print(f"[train] step={step:04d} loss={loss:.6f}")

    ckpt = {
        "model_state": model.state_dict(),
        "model_args": {
            "hidden_dim": args.hidden_dim,
            "num_blocks": args.num_blocks,
        },
        "flow_args": {
            "sigma": args.sigma,
            "source_type": args.source_type,
            "source_noise_scale": args.source_noise_scale,
        },
    }
    torch.save(ckpt, args.save_path)
    print(f"[train] checkpoint saved to: {args.save_path}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train ETFlowAlign on synthetic alignment data.")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-atoms", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-blocks", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--source-type", type=str, default="reference_anchored", choices=["gaussian", "reference_anchored", "query_perturbed"])
    p.add_argument("--source-noise-scale", type=float, default=0.5)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-path", type=str, default="etflowalign_ckpt.pt")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
