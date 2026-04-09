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
    """Optimization hyperparameters for ETFlowAlign training."""
    lr: float = 1e-4
    weight_decay: float = 0.0


def load_alignment_batch_from_pt(path: str, device: torch.device) -> tuple[AlignmentBatch, torch.Tensor]:
    """Load a real-task style batch from a torch serialized dictionary.

    Required keys:
        query_pos, query_atom_type, query_batch, target_query_pos
    Optional keys:
        reference_pos, reference_atom_type, reference_batch, pocket_pos
    """
    payload = torch.load(path, map_location=device)
    batch = AlignmentBatch(
        query_pos=payload["query_pos"].to(device),
        query_atom_type=payload["query_atom_type"].to(device),
        query_batch=payload["query_batch"].to(device),
        reference_pos=payload.get("reference_pos", None).to(device) if payload.get("reference_pos", None) is not None else None,
        reference_atom_type=payload.get("reference_atom_type", None).to(device) if payload.get("reference_atom_type", None) is not None else None,
        reference_batch=payload.get("reference_batch", None).to(device) if payload.get("reference_batch", None) is not None else None,
        pocket_pos=payload.get("pocket_pos", None).to(device) if payload.get("pocket_pos", None) is not None else None,
        query_node_attr=payload.get("query_node_attr", None).to(device) if payload.get("query_node_attr", None) is not None else None,
        reference_node_attr=payload.get("reference_node_attr", None).to(device) if payload.get("reference_node_attr", None) is not None else None,
    )
    target = payload["target_query_pos"].to(device)
    return batch, target


def build_training_components(model: ETFlowAlignModel, train_config: TrainConfig, fm_config: FlowMatchingConfig):
    """Create matcher + optimizer objects used in training."""
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
    """Run one optimization step and return scalar loss."""
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
    query_batch = torch.arange(batch_size, device=device).repeat_interleave(n_atoms)  # Graph id per query atom.
    reference_batch = query_batch.clone()  # Graph id per reference atom.

    # Base molecular-like cloud around origin per graph.
    ref = 0.7 * torch.randn(batch_size * n_atoms, 3, device=device)  # Reference coordinates.
    atom_type = torch.randint(low=0, high=16, size=(batch_size * n_atoms,), device=device)  # Toy atom ids.

    # Construct alignment target from reference with global transform + local deformation.
    t = torch.randn(batch_size, 3, device=device) * 0.25  # Per-graph translation.
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
    """Entry point used by CLI: train on synthetic data and save checkpoint."""
    device = torch.device(args.device)
    model = ETFlowAlignModel(
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        time_embed_dim=args.time_embed_dim,
        edge_cutoff=args.edge_cutoff,
        max_neighbors=args.max_neighbors,
        extra_feat_dim=args.extra_feat_dim,
    ).to(device)

    fm_config = FlowMatchingConfig(
        sigma=args.sigma,
        source_type=args.source_type,
        source_noise_scale=args.source_noise_scale,
        time_weighting=args.time_weighting,
        allow_query_perturbed_for_inference=args.allow_query_perturbed_for_inference,
    )
    train_config = TrainConfig(lr=args.lr, weight_decay=args.weight_decay)
    matcher, optimizer = build_training_components(model, train_config, fm_config)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.steps)) if args.use_scheduler else None

    val_batch = None
    val_target = None
    if args.val_data:
        val_batch, val_target = load_alignment_batch_from_pt(args.val_data, device)

    final_step = 0
    best_val = float("inf")
    best_state = None
    for step in range(1, args.steps + 1):
        if args.train_data:
            batch, target = load_alignment_batch_from_pt(args.train_data, device)
        else:
            batch, target = make_synthetic_alignment_batch(args.batch_size, args.n_atoms, device)
        loss = train_step(model, matcher, optimizer, batch, target)
        if step % args.log_every == 0 or step == 1:
            print(f"[train] step={step:04d} loss={loss:.6f}")

        if scheduler is not None:
            scheduler.step()

        if val_batch is not None and (step % args.val_every == 0 or step == args.steps):
            model.eval()
            with torch.no_grad():
                vloss = float(flow_matching_step(model=model, matcher=matcher, batch=val_batch, target_query_pos=val_target).item())
            print(f"[val] step={step:04d} loss={vloss:.6f}")
            if vloss < best_val:
                best_val = vloss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.train()
        final_step = step

    ckpt = {
        "step": final_step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_args": {
            "hidden_dim": args.hidden_dim,
            "num_blocks": args.num_blocks,
            "time_embed_dim": args.time_embed_dim,
            "edge_cutoff": args.edge_cutoff,
            "max_neighbors": args.max_neighbors,
            "extra_feat_dim": args.extra_feat_dim,
        },
        "train_config": {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "n_atoms": args.n_atoms,
        },
        "flow_args": {
            "sigma": args.sigma,
            "source_type": args.source_type,
            "source_noise_scale": args.source_noise_scale,
            "time_weighting": args.time_weighting,
            "allow_query_perturbed_for_inference": args.allow_query_perturbed_for_inference,
        },
        "best_val_loss": best_val if val_batch is not None else None,
        "best_model_state": best_state,
    }
    torch.save(ckpt, args.save_path)
    print(f"[train] checkpoint saved to: {args.save_path}")


def build_argparser() -> argparse.ArgumentParser:
    """Define CLI arguments for the standalone training script."""
    p = argparse.ArgumentParser(description="Train ETFlowAlign on synthetic alignment data.")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-atoms", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-blocks", type=int, default=4)
    p.add_argument("--time-embed-dim", type=int, default=64)
    p.add_argument("--edge-cutoff", type=float, default=6.0)
    p.add_argument("--max-neighbors", type=int, default=32)
    p.add_argument("--extra-feat-dim", type=int, default=0, help="Feature dim for optional query/reference node attributes.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument(
        "--source-type",
        type=str,
        default="reference_anchored",
        choices=["gaussian", "reference_anchored", "query_perturbed", "rigid_reference_perturbed"],
    )
    p.add_argument("--source-noise-scale", type=float, default=0.5)
    p.add_argument("--time-weighting", type=str, default="uniform", choices=["uniform", "mid"])
    p.add_argument("--allow-query-perturbed-for-inference", action="store_true")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--train-data", type=str, default="", help="Path to torch file for real training batch.")
    p.add_argument("--val-data", type=str, default="", help="Path to torch file for real validation batch.")
    p.add_argument("--val-every", type=int, default=50)
    p.add_argument("--use-scheduler", action="store_true")
    p.add_argument("--save-path", type=str, default="etflowalign_ckpt.pt")
    return p


def main() -> None:
    """CLI main."""
    args = build_argparser().parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
