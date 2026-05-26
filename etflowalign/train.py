"""Training script and utilities for ETFlowAlign.

Examples:
    python -m etflowalign.train --synthetic-smoke --steps 2 --batch-size 2 --n-atoms 8 --save-path /tmp/etflowalign_smoke.pt
    python -m etflowalign.train --train-data /path/to/train_batch.pt --steps 1 --save-path /tmp/etflowalign_ckpt.pt
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, optim

from .data import load_alignment_batch_from_pt
from .flow_matching import AlignmentFlowMatcher, FlowMatchingConfig, flow_matching_step
from .model import AlignmentBatch, ETFlowAlignModel


@dataclass
class TrainConfig:
    """Optimization hyperparameters for ETFlowAlign training."""

    lr: float = 1e-4
    weight_decay: float = 0.0


def build_training_components(
    model: ETFlowAlignModel,
    train_config: TrainConfig,
    fm_config: FlowMatchingConfig,
) -> tuple[AlignmentFlowMatcher, optim.Optimizer]:
    """Create flow matcher and optimizer."""
    matcher = AlignmentFlowMatcher(config=fm_config)
    optimizer = optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
    return matcher, optimizer


def train_step(
    model: ETFlowAlignModel,
    matcher: AlignmentFlowMatcher,
    optimizer: optim.Optimizer,
    batch: AlignmentBatch,
    target_query_pos: Tensor,
) -> float:
    """Run one optimization step and return scalar loss."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = flow_matching_step(model=model, matcher=matcher, batch=batch, target_query_pos=target_query_pos)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    return float(loss.detach().item())


def make_synthetic_alignment_batch(batch_size: int, n_atoms: int, device: torch.device) -> tuple[AlignmentBatch, Tensor]:
    """Create toy alignment batch for smoke tests."""
    query_batch = torch.arange(batch_size, device=device).repeat_interleave(n_atoms)
    reference_batch = query_batch.clone()

    reference_pos = 0.7 * torch.randn(batch_size * n_atoms, 3, device=device)
    atom_type = torch.randint(low=0, high=16, size=(batch_size * n_atoms,), device=device)

    translation = 0.25 * torch.randn(batch_size, 3, device=device)
    target_query_pos = reference_pos + translation[query_batch]
    target_query_pos = target_query_pos + 0.05 * torch.sin(target_query_pos * 2.0)

    query_init = reference_pos + 0.8 * torch.randn_like(reference_pos)
    batch = AlignmentBatch(
        query_pos=query_init,
        query_atom_type=atom_type,
        query_batch=query_batch,
        reference_pos=reference_pos,
        reference_atom_type=atom_type,
        reference_batch=reference_batch,
    )
    return batch, target_query_pos


def run_training(args: argparse.Namespace) -> None:
    """CLI training entrypoint."""
    device = torch.device(args.device)
    model = ETFlowAlignModel(
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        use_atom_index_embed=args.use_atom_index_embed,
        use_direct_vector_head=args.use_direct_vector_head,
        max_atoms=args.max_atoms,
    ).to(device)

    fm_config = FlowMatchingConfig(
        sigma=args.sigma,
        source_type=args.source_type,
        source_noise_scale=args.source_noise_scale,
        center_source=args.center_source,
        center_target=args.center_target,
        use_kabsch_alignment=args.use_kabsch_alignment,
        fixed_t=args.fixed_t,
    )
    train_config = TrainConfig(lr=args.lr, weight_decay=args.weight_decay)
    matcher, optimizer = build_training_components(model, train_config, fm_config)

    best_loss = float("inf")
    best_step = -1
    best_model_state: dict[str, torch.Tensor] | None = None
    final_loss = float("nan")

    for step in range(1, args.steps + 1):
        if args.synthetic_smoke:
            batch, target_query_pos = make_synthetic_alignment_batch(args.batch_size, args.n_atoms, device)
        else:
            batch, target_query_pos, _ = load_alignment_batch_from_pt(
                args.train_data,
                require_target=True,
                device=device,
            )
            assert target_query_pos is not None

        loss = train_step(model, matcher, optimizer, batch, target_query_pos)
        final_loss = float(loss)
        if not math.isfinite(final_loss):
            raise FloatingPointError(f"Non-finite loss at step={step}: {final_loss}")

        if final_loss < best_loss:
            best_loss = final_loss
            best_step = step
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if step % args.log_every == 0 or step == 1:
            print(f"[train] step={step:04d} loss={final_loss:.6f}")

    model_args = {
        "hidden_dim": args.hidden_dim,
        "num_blocks": args.num_blocks,
        "use_atom_index_embed": args.use_atom_index_embed,
        "use_direct_vector_head": args.use_direct_vector_head,
        "max_atoms": args.max_atoms,
    }
    flow_args = {
        "sigma": args.sigma,
        "source_type": args.source_type,
        "source_noise_scale": args.source_noise_scale,
        "center_source": args.center_source,
        "center_target": args.center_target,
        "use_kabsch_alignment": args.use_kabsch_alignment,
        "fixed_t": args.fixed_t,
    }
    train_args = vars(args).copy()

    final_ckpt = {
        "model_state": model.state_dict(),
        "model_args": model_args,
        "flow_args": flow_args,
        "train_args": train_args,
        "final_loss": final_loss,
        "best_loss": best_loss,
        "best_step": best_step,
    }
    torch.save(final_ckpt, args.save_path)

    if best_model_state is None:
        raise RuntimeError("best_model_state is missing despite finite training loop.")

    save_path = Path(args.save_path)
    best_path = str(save_path.with_name(f"{save_path.stem}_best.pt"))
    best_ckpt = {
        "model_state": best_model_state,
        "model_args": model_args,
        "flow_args": flow_args,
        "train_args": train_args,
        "best_loss": best_loss,
        "best_step": best_step,
    }
    torch.save(best_ckpt, best_path)

    print(f"[train] checkpoint saved to: {args.save_path}")
    print(f"[train] best checkpoint saved to: {best_path} at step={best_step} loss={best_loss:.6f}")


def build_argparser() -> argparse.ArgumentParser:
    """Define training CLI."""
    p = argparse.ArgumentParser(description="Train ETFlowAlign.")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-atoms", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-blocks", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument(
        "--source-type",
        type=str,
        default="reference_anchored",
        choices=["gaussian", "reference_anchored", "query_perturbed", "input_query"],
    )
    p.add_argument("--source-noise-scale", type=float, default=0.5)
    p.add_argument("--use-atom-index-embed", action="store_true")
    p.add_argument("--use-direct-vector-head", action="store_true")
    p.add_argument("--max-atoms", type=int, default=256)

    p.add_argument("--center-source", dest="center_source", action="store_true")
    p.add_argument("--no-center-source", dest="center_source", action="store_false")
    p.set_defaults(center_source=True)
    p.add_argument("--center-target", dest="center_target", action="store_true")
    p.add_argument("--no-center-target", dest="center_target", action="store_false")
    p.set_defaults(center_target=True)
    p.add_argument("--use-kabsch-alignment", dest="use_kabsch_alignment", action="store_true")
    p.add_argument("--no-use-kabsch-alignment", dest="use_kabsch_alignment", action="store_false")
    p.set_defaults(use_kabsch_alignment=True)
    p.add_argument("--fixed-t", type=float, default=None)

    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-path", type=str, default="etflowalign_ckpt.pt")

    p.add_argument("--synthetic-smoke", action="store_true")
    p.add_argument("--train-data", type=str, default="")
    return p


def main() -> None:
    """CLI main."""
    args = build_argparser().parse_args()
    if args.synthetic_smoke == bool(args.train_data):
        raise ValueError("Provide exactly one of --synthetic-smoke or --train-data.")
    run_training(args)


if __name__ == "__main__":
    main()
