"""Inference script and APIs for ETFlowAlign.

Usage example:
    python -m etflowalign.inference --checkpoint etflowalign_ckpt.pt --num-samples 16
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor

from .flow_matching import AlignmentFlowMatcher, FlowMatchingConfig
from .guidance import UFFGuidanceConfig, UFFPocketGuidance
from .model import AlignmentBatch, ETFlowAlignModel
from .rankers import LegacyReferenceMseRanker, PluginRanker, RankerConfig
from .sampler import ETFlowAlignSampler, GuidanceFn, ODESamplerConfig
from .train import load_alignment_batch_from_pt, make_synthetic_alignment_batch


@dataclass
class InferenceConfig:
    """Minimal inference configuration container."""

    num_samples: int = 8


def generate_candidates(
    sampler: ETFlowAlignSampler,
    batch: AlignmentBatch,
    source_sampler: Callable[[AlignmentBatch], Tensor],
    guidance_fn: Optional[GuidanceFn] = None,
    num_samples: int = 8,
) -> Tensor:
    """Generate aligned query candidates as ``[num_samples, N, 3]``.

    Args:
        sampler: ODE sampler.
        batch: Conditioning batch.
        source_sampler: Callback generating initial coordinates ``x0``.
        guidance_fn: Optional guidance callback.
        num_samples: Number of candidates to generate.
    """
    outputs = []
    for _ in range(num_samples):
        x0 = source_sampler(batch)
        xT = sampler.sample(batch=batch, x0=x0, guidance_fn=guidance_fn)
        outputs.append(xT)
    return torch.stack(outputs, dim=0)


def rank_candidates(
    candidates: Tensor,
    batch: AlignmentBatch,
    ranker: Optional[PluginRanker | LegacyReferenceMseRanker] = None,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Rank candidates in descending score order with component breakdown."""
    ranker = ranker or PluginRanker()
    breakdown = ranker.score(candidates, batch)
    order = torch.argsort(breakdown.total, descending=True)
    components = {
        "tanimoto": breakdown.tanimoto[order],
        "physics": breakdown.physics[order],
    }
    return candidates[order], breakdown.total[order], components


def make_pocket_pull_guidance(scale: float = 1.0) -> GuidanceFn:
    """Create a toy pocket guidance: pull query toward pocket center if available."""

    def _guidance(batch: AlignmentBatch, t_graph: Tensor, v: Tensor) -> Tensor:
        """Guidance callback used by sampler.

        Variables:
            batch: Current sampler batch/state.
            t_graph: Current time per graph.
            v: Current model velocity prediction.
        """
        if batch.pocket_pos is None or batch.pocket_pos.numel() == 0:
            return torch.zeros_like(v)
        pocket_center = batch.pocket_pos.mean(dim=0, keepdim=True)
        g = pocket_center - batch.query_pos
        return scale * g

    return _guidance


def run_inference(args: argparse.Namespace) -> None:
    """CLI inference routine: load checkpoint, sample, rank, and optionally save."""
    device = torch.device(args.device)
    ckpt, model_args, flow_args, sampler_args, sampler, fm = _build_inference_runtime(args=args, device=device)

    if args.input_batch:
        batch, _ = load_alignment_batch_from_pt(args.input_batch, device)
    else:
        # Demo input (replace with real preprocessed task data).
        batch, _ = make_synthetic_alignment_batch(batch_size=1, n_atoms=args.n_atoms, device=device)
    if args.use_pocket_guidance:
        batch.pocket_pos = batch.reference_pos + 0.1 * torch.randn_like(batch.reference_pos)

    source_sampler = lambda b: fm.sample_inference_source(b)  # noqa: E731
    guidance_fn = None
    if args.use_pocket_guidance:
        if args.guidance_backend == "uff":
            guidance_fn = UFFPocketGuidance(
                UFFGuidanceConfig(
                    scale=1.0,
                    query_repulsion_weight=args.uff_query_repulsion_weight,
                    pocket_attraction_weight=args.uff_pocket_attraction_weight,
                    pocket_repulsion_weight=args.uff_pocket_repulsion_weight,
                    epsilon=args.uff_epsilon,
                    sigma=args.uff_sigma,
                    backend_mode=args.uff_backend_mode,
                    backend_module=args.uff_backend_module,
                    backend_symbol=args.uff_backend_symbol,
                )
            )
        else:
            guidance_fn = make_pocket_pull_guidance(scale=1.0)

    candidates = generate_candidates(
        sampler=sampler,
        batch=batch,
        source_sampler=source_sampler,
        guidance_fn=guidance_fn,
        num_samples=args.num_samples,
    )
    ranker = _build_ranker(args)
    ranked, scores, components = rank_candidates(candidates=candidates, batch=batch, ranker=ranker)
    if args.top_k > 0:
        ranked = ranked[: args.top_k]
        scores = scores[: args.top_k]
        components = {k: v[: args.top_k] for k, v in components.items()}

    print(f"[inference] candidates={candidates.shape}")
    print(f"[inference] top_score={scores[0].item():.6f}")

    if args.save_path:
        torch.save(
            {
                "candidates": ranked.cpu(),
                "scores": scores.cpu(),
                "component_scores": {k: v.cpu() for k, v in components.items()},
                "metadata": {
                    "checkpoint": args.checkpoint,
                    "model_args": model_args,
                    "flow_args": flow_args,
                    "sampler_args": sampler_args,
                    "source_policy": flow_args.get("source_type", "reference_anchored"),
                    "input_batch_path": args.input_batch if args.input_batch else None,
                    "input_has_query_node_attr": bool(batch.query_node_attr is not None),
                    "input_has_reference_node_attr": bool(batch.reference_node_attr is not None),
                    "guidance_used": bool(args.use_pocket_guidance and args.guidance_scale > 0.0),
                    "guidance_backend": args.guidance_backend,
                    "uff_backend_mode": args.uff_backend_mode,
                    "uff_backend_module": args.uff_backend_module if args.uff_backend_module else None,
                    "uff_backend_symbol": args.uff_backend_symbol if args.uff_backend_symbol else None,
                    "ranker": args.ranker,
                    "tanimoto_weight": args.tanimoto_weight,
                    "physics_weight": args.physics_weight,
                    "num_samples": args.num_samples,
                    "top_k": args.top_k,
                    "n_atoms": args.n_atoms,
                    "training_step": ckpt.get("step"),
                },
            },
            args.save_path,
        )
        print(f"[inference] saved to: {args.save_path}")


def _build_ranker(args: argparse.Namespace) -> PluginRanker | LegacyReferenceMseRanker:
    """Build ranking backend from CLI options."""
    if args.ranker == "legacy_reference_mse":
        return LegacyReferenceMseRanker()

    config = RankerConfig(
        tanimoto_weight=args.tanimoto_weight,
        physics_weight=args.physics_weight,
        shape_sigma=args.shape_sigma,
        clash_cutoff=args.clash_cutoff,
        clash_penalty_weight=args.clash_penalty_weight,
    )
    return PluginRanker(config=config)


def _build_inference_runtime(args: argparse.Namespace, device: torch.device):
    """Load checkpoint and construct model/flow/sampler runtime objects."""
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_args = ckpt.get("model_args", {})
    if "backbone_type" not in model_args:
        # Backward compatibility for early checkpoints trained with EGNN blocks.
        model_args["backbone_type"] = "egnn"
    if "num_heads" not in model_args:
        model_args["num_heads"] = 4
    flow_args = ckpt.get("flow_args", {})

    model = ETFlowAlignModel(**model_args).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    fm = AlignmentFlowMatcher(FlowMatchingConfig(**flow_args))
    sampler_args = {
        "n_steps": args.n_steps,
        "solver": args.solver,
        "guidance_scale": args.guidance_scale,
        "guidance_mode": args.guidance_mode,
        "max_guidance_norm": args.max_guidance_norm,
        "pc_corrector_step_scale": args.pc_corrector_step_scale,
        "adaptive_dt": args.adaptive_dt,
        "adaptive_dt_min_scale": args.adaptive_dt_min_scale,
        "adaptive_dt_max_scale": args.adaptive_dt_max_scale,
        "log_trajectory": args.log_trajectory,
        "max_trace_steps": args.max_trace_steps,
    }
    sampler = ETFlowAlignSampler(model=model, config=ODESamplerConfig(**sampler_args))
    return ckpt, model_args, flow_args, sampler_args, sampler, fm


def build_argparser() -> argparse.ArgumentParser:
    """Define CLI arguments for inference script."""
    p = argparse.ArgumentParser(description="Run ETFlowAlign inference on synthetic demo data.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--input-batch", type=str, default="", help="Path to torch batch file with query/reference/target tensors.")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--top-k", type=int, default=0, help="If >0, keep only top-k ranked candidates.")
    p.add_argument("--ranker", type=str, default="plugin_combo", choices=["plugin_combo", "legacy_reference_mse"])
    p.add_argument("--tanimoto-weight", type=float, default=1.0)
    p.add_argument("--physics-weight", type=float, default=1.0)
    p.add_argument("--shape-sigma", type=float, default=1.2)
    p.add_argument("--clash-cutoff", type=float, default=1.0)
    p.add_argument("--clash-penalty-weight", type=float, default=2.0)
    p.add_argument("--n-atoms", type=int, default=16)
    p.add_argument("--n-steps", type=int, default=64)
    p.add_argument("--solver", type=str, default="heun", choices=["euler", "heun"])
    p.add_argument("--guidance-scale", type=float, default=0.0)
    p.add_argument("--guidance-mode", type=str, default="vector_field", choices=["vector_field", "predictor_corrector"])
    p.add_argument("--max-guidance-norm", type=float, default=5.0)
    p.add_argument("--pc-corrector-step-scale", type=float, default=1.0)
    p.add_argument("--adaptive-dt", action="store_true")
    p.add_argument("--adaptive-dt-min-scale", type=float, default=0.1)
    p.add_argument("--adaptive-dt-max-scale", type=float, default=2.0)
    p.add_argument("--log-trajectory", action="store_true")
    p.add_argument("--max-trace-steps", type=int, default=2000)
    p.add_argument("--use-pocket-guidance", action="store_true")
    p.add_argument("--guidance-backend", type=str, default="uff", choices=["uff", "toy_pull"])
    p.add_argument("--uff-query-repulsion-weight", type=float, default=0.2)
    p.add_argument("--uff-pocket-attraction-weight", type=float, default=1.0)
    p.add_argument("--uff-pocket-repulsion-weight", type=float, default=0.5)
    p.add_argument("--uff-epsilon", type=float, default=0.1)
    p.add_argument("--uff-sigma", type=float, default=1.5)
    p.add_argument("--uff-backend-mode", type=str, default="auto", choices=["auto", "uff_pytorch", "surrogate"])
    p.add_argument("--uff-backend-module", type=str, default="")
    p.add_argument("--uff-backend-symbol", type=str, default="")
    p.add_argument("--save-path", type=str, default="")
    return p


def main() -> None:
    """CLI main."""
    args = build_argparser().parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
