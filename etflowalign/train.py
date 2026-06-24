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
from .dataset import AlignmentDataset, sample_training_batch, train_val_split
from .evaluation import evaluate_paths, format_summary
from .flow_matching import AlignmentFlowMatcher, FlowMatchingConfig, flow_matching_step
from .model import AlignmentBatch, ETFlowAlignModel


@dataclass
class TrainConfig:
    """Optimization hyperparameters for ETFlowAlign training."""

    lr: float = 1e-4
    weight_decay: float = 0.0
    lambda_bond: float = 0.0


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
    lambda_bond: float = 0.0,
) -> float:
    """Run one optimization step and return scalar loss."""
    model.train()
    optimizer.zero_grad(set_to_none=True)

    loss = flow_matching_step(
        model,
        matcher,
        batch,
        target_query_pos,
        lambda_bond=lambda_bond,
    )

    loss_check = float(loss.detach())
    if not math.isfinite(loss_check) or abs(loss_check) > 1e4:
        # Skip non-finite OR exploding losses (normal loss is ~10-200). Raising here
        # bypasses backward/step so one runaway batch cannot corrupt the weights; the
        # caller rolls back to the best checkpoint and continues.
        raise FloatingPointError(f"Non-finite/exploding training loss before backward: {loss_check}")

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    # Sanitize non-finite gradients before stepping. clip_grad_norm_ does NOT guard against
    # this: a single NaN/Inf gradient makes its total_norm NaN and then poisons every weight,
    # so the next forward pass produces NaN and training dies. Zeroing the bad grads turns a
    # rare unstable step into a no-op instead of a crash.
    for param in model.parameters():
        if param.grad is not None:
            torch.nan_to_num_(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
    optimizer.step()

    loss_value = float(loss.detach().cpu().item())
    if not math.isfinite(loss_value):
        raise FloatingPointError(
            f"Non-finite training loss after optimization step: {loss_value}"
        )

    return loss_value


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


def _clone_state_dict_to_cpu(model: ETFlowAlignModel) -> dict[str, torch.Tensor]:
    """Clone model parameters to CPU for checkpointing."""
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _make_best_checkpoint_path(save_path: str) -> Path:
    """Create best-checkpoint path from final checkpoint path."""
    path = Path(save_path)
    if path.suffix == ".pt":
        return path.with_name(f"{path.stem}_best{path.suffix}")
    return Path(f"{save_path}.best.pt")


def run_training(args: argparse.Namespace) -> None:
    """CLI training entrypoint."""
    device = torch.device(args.device)

    model_args = {
        "hidden_dim": args.hidden_dim,
        "num_blocks": args.num_blocks,
        "use_atom_index_embed": args.use_atom_index_embed,
        "use_equivariant_basis_head": args.use_equivariant_basis_head,
        "use_rigid_head": args.use_rigid_head,
        "use_pocket_conditioning": args.use_pocket_conditioning,
        "pocket_cutoff": args.pocket_cutoff,
        "pocket_max_neighbors": args.pocket_max_neighbors,
        "use_reference_conditioning": args.use_reference_conditioning,
        "reference_cutoff": args.reference_cutoff,
        "reference_max_neighbors": args.reference_max_neighbors,
        "use_node_attr": args.use_node_attr,
        "node_attr_dim": args.node_attr_dim,
        "max_atoms": args.max_atoms,
    }

    model = ETFlowAlignModel(**model_args).to(device)

    flow_args = {
        "sigma": args.sigma,
        "source_type": args.source_type,
        "source_noise_scale": args.source_noise_scale,
        "center_source": args.center_source,
        "center_target": args.center_target,
        "use_kabsch_alignment": args.use_kabsch_alignment,
        "fixed_t": args.fixed_t,
        "path_type": args.path_type,
    }

    fm_config = FlowMatchingConfig(**flow_args)

    train_config = TrainConfig(lr=args.lr, weight_decay=args.weight_decay, lambda_bond=args.lambda_bond,)
    matcher, optimizer = build_training_components(model, train_config, fm_config)

    train_args = vars(args).copy()

    dataset: AlignmentDataset | None = None
    sample_generator: torch.Generator | None = None
    val_paths: list[str] = []
    if args.train_dir:
        all_paths = AlignmentDataset.from_directory(args.train_dir, require_target=True).paths
        if args.train_limit > 0:
            all_paths = all_paths[: args.train_limit]
        if args.val_fraction > 0.0:
            train_paths, val_paths = train_val_split(all_paths, args.val_fraction, args.split_seed)
            val_manifest = Path(f"{args.save_path}.val.txt")
            val_manifest.parent.mkdir(parents=True, exist_ok=True)
            val_manifest.write_text("\n".join(val_paths))
            dataset = AlignmentDataset(train_paths, require_target=True)
            print(f"[train] held-out val: {len(val_paths)} complexes (never trained) -> {val_manifest}")
        else:
            dataset = AlignmentDataset(all_paths, require_target=True)
        sample_generator = torch.Generator().manual_seed(0)
        print(
            f"[train] dataset: {len(dataset)} train complexes from {args.train_dir} "
            f"| conditioning={args.conditioning} batch_size={args.batch_size}"
        )
        if args.use_pocket_conditioning:
            _probe_batch, _ = dataset[0]
            _has_chem = _probe_batch.pocket_atom_type is not None
            print(
                "[train] pocket conditioning: ON | pocket atom types (chemistry): "
                + ("present" if _has_chem else "ABSENT -> fallback token (re-extract the dataset!)")
            )

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    best_path = _make_best_checkpoint_path(args.save_path)
    best_path.parent.mkdir(parents=True, exist_ok=True)

    def _checkpoint(model_state: dict[str, torch.Tensor], extra: dict) -> dict:
        ckpt = {
            "model_state": model_state,
            "model_args": model_args,
            "flow_args": flow_args,
            "train_args": train_args,
        }
        ckpt.update(extra)
        return ckpt
    
    best_loss = float("inf")
    best_step: int | None = None
    best_model_state: dict[str, torch.Tensor] | None = None
    final_loss: float | None = None
    consecutive_failures = 0
    max_consecutive_failures = 20

    for step in range(1, args.steps + 1):
        if args.synthetic_smoke:
            batch, target_query_pos = make_synthetic_alignment_batch(
                args.batch_size,
                args.n_atoms,
                device,
            )
        elif dataset is not None:
            batch, target_query_pos = sample_training_batch(
                dataset,
                batch_size=args.batch_size,
                conditioning=args.conditioning,
                device=device,
                generator=sample_generator,
            )
            assert target_query_pos is not None
        else:
            batch, target_query_pos, _ = load_alignment_batch_from_pt(
                args.train_data,
                require_target=True,
                device=device,
            )
            assert target_query_pos is not None

        try:
            loss = train_step(
                model=model,
                matcher=matcher,
                optimizer=optimizer,
                batch=batch,
                target_query_pos=target_query_pos,
                lambda_bond=train_config.lambda_bond,
            )
            step_loss = float(loss)
            if not math.isfinite(step_loss):
                raise FloatingPointError(f"Non-finite loss at step={step}: {step_loss}.")
        except FloatingPointError as exc:
            consecutive_failures += 1
            # The guard fires BEFORE backward/step, so the weights are unchanged by this bad
            # batch. The right response is to SKIP it and keep the (good) current weights, so
            # progress accumulates. Rolling back to best on every spike would reset all
            # progress and trap training (doom loop) when spikes are frequent.
            if consecutive_failures > max_consecutive_failures:
                # Many failures in a row -> weights likely in a bad region. Escape to best
                # (or abort if we never had a finite step).
                if best_model_state is None:
                    raise
                print(f"[train] step={step:04d} {consecutive_failures} consecutive non-finite; restoring best@{best_step} and continuing.")
                model.load_state_dict(best_model_state)
                optimizer.state.clear()
                consecutive_failures = 0
            elif step % args.log_every == 0 or consecutive_failures <= 3:
                print(f"[train] step={step:04d} skipped (non-finite/exploding loss); weights unchanged.")
            continue

        consecutive_failures = 0
        final_loss = step_loss
        
        if final_loss < best_loss:
            best_loss = final_loss
            best_step = step
            best_model_state = _clone_state_dict_to_cpu(model)
            # Persist the best immediately so a later crash cannot throw it away.
            torch.save(
                _checkpoint(best_model_state, {"best_loss": best_loss, "best_step": best_step}),
                best_path,
            )

        if step % args.log_every == 0 or step == 1:
            best_info = f" best={best_loss:.6f}@{best_step}" if best_step is not None else ""
            print(f"[train] step={step:04d} loss={final_loss:.6f}{best_info}")
        
        if val_paths and args.val_every > 0 and step % args.val_every == 0:
            try:
                model.eval()
                val_summary, _ = evaluate_paths(
                    model,
                    val_paths,
                    conditioning=args.conditioning,
                    n_steps=args.val_eval_n_steps,
                    solver=args.val_eval_solver,
                    device=device,
                    limit=args.val_eval_limit,
                    flow_config=matcher.config,  # x0를 학습 source 분포에서 시작(일관성)
                )
                print(f"[val]   step={step:04d} {format_summary(val_summary)}")
            except Exception as exc:  # noqa: BLE001 - val monitoring must never kill training
                print(f"[val]   step={step:04d} skipped (eval error: {exc})")
            finally:
                model.train()

    if final_loss is None:
        raise RuntimeError("Training finished without running any optimization step.")

    if best_model_state is None or best_step is None:
        raise RuntimeError(
            "No finite best checkpoint was captured. "
            "Training likely failed before producing a valid loss."
        )

    torch.save(
        _checkpoint(
            _clone_state_dict_to_cpu(model),
            {"final_loss": final_loss, "best_loss": best_loss, "best_step": best_step},
        ),
        save_path,
    )
    torch.save(
        _checkpoint(best_model_state, {"best_loss": best_loss, "best_step": best_step}),
        best_path,
    )

    print(f"[train] checkpoint saved to: {save_path}")
    print(
        f"[train] best checkpoint saved to: {best_path} "
        f"at step={best_step} loss={best_loss:.6f}"
    )

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
    p.add_argument("--max-atoms", type=int, default=256)
    p.add_argument(
        "--use-equivariant-basis-head",
        action="store_true",
        help="Use an equivariant vector basis head instead of the legacy gate*x head.",
    )
    p.add_argument(
        "--use-rigid-head",
        action="store_true",
        help="Direction A: predict a per-graph rigid-body velocity field (preserves geometry).",
    )
    p.add_argument(
        "--use-pocket-conditioning",
        action="store_true",
        help="Feed pocket atoms into the model (ligand<->pocket messages + pocket-shape basis).",
    )
    p.add_argument("--pocket-cutoff", type=float, default=8.0, help="Ligand-pocket edge cutoff (A).")
    p.add_argument("--pocket-max-neighbors", type=int, default=16, help="Max pocket neighbors per ligand atom.")
    p.add_argument(
        "--use-reference-conditioning",
        action="store_true",
        help="Feed reference atoms into the model (query<->reference messages + reference-shape basis). "
             "DiffAlign-faithful reference shape conditioning.",
    )
    p.add_argument("--reference-cutoff", type=float, default=8.0, help="Query-reference edge cutoff (A).")
    p.add_argument("--reference-max-neighbors", type=int, default=16, help="Max reference neighbors per query atom.")
    p.add_argument(
        "--path-type",
        type=str,
        default="linear",
        choices=["linear", "rigid"],
        help="Flow-matching probability path: linear coordinate interp or SE(3) rigid geodesic.",
    )
    p.add_argument(
        "--use-node-attr",
        action="store_true",
        help="Use query_node_attr/reference_node_attr chemistry features if present.",
    )
    p.add_argument(
        "--node-attr-dim",
        type=int,
        default=5,
        help="Input dimension of node_attr features.",
    )
    p.add_argument(
        "--lambda-bond",
        type=float,
        default=0.0,
        help="Weight for endpoint bond length regularization.",
    )
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
    p.add_argument("--train-data", type=str, default="", help="Single .pt batch (overfit/smoke).")
    p.add_argument("--train-dir", type=str, default="", help="Directory of per-complex .pt payloads (dataset training).")
    p.add_argument(
        "--conditioning",
        type=str,
        default="pocket",
        choices=["pocket", "reference", "both"],
        help="Which conditioning signal to keep when training from --train-dir.",
    )
    p.add_argument("--val-fraction", type=float, default=0.0, help="Hold out this fraction of --train-dir as val (never trained on; written to <save-path>.val.txt).")
    p.add_argument("--split-seed", type=int, default=0, help="Seed for the train/val split.")
    p.add_argument("--train-limit", type=int, default=0, help="Use only the first N complexes of --train-dir (0 = all). For overfit diagnostics.") 
    p.add_argument("--val-every", type=int, default=0, help="Evaluate held-out val pose RMSD every N steps (0 = off; needs --val-fraction>0).")
    p.add_argument("--val-eval-limit", type=int, default=100, help="Evaluate at most this many val complexes per check (speed).")
    p.add_argument("--val-eval-n-steps", type=int, default=50, help="ODE steps for val pose sampling.")
    p.add_argument("--val-eval-solver", type=str, default="heun", choices=["euler", "heun"], help="Solver for val pose sampling.")
    return p


def main() -> None:
    """CLI main."""
    args = build_argparser().parse_args()
    sources = [bool(args.synthetic_smoke), bool(args.train_data), bool(args.train_dir)]
    if sum(sources) != 1:
        raise ValueError("Provide exactly one of --synthetic-smoke, --train-data, or --train-dir.")
    run_training(args)


if __name__ == "__main__":
    main()