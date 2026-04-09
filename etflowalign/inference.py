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
from .model import AlignmentBatch, ETFlowAlignModel
from .sampler import ETFlowAlignSampler, GuidanceFn, ODESamplerConfig
from .train import make_synthetic_alignment_batch

RankFn = Callable[[Tensor, AlignmentBatch], Tensor]


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


def reference_fit_ranker(candidates: Tensor, batch: AlignmentBatch) -> Tensor:
    """Simple ranking adapter: negative MSE to reference positions per candidate.

    In real usage this should be replaced with docking/ranking metrics such as
    TanimotoCombo + pocket-aware scores.
    """
    if batch.reference_pos is None:
        return torch.zeros(candidates.size(0), device=candidates.device)

    diff = candidates - batch.reference_pos.unsqueeze(0)
    mse = (diff * diff).mean(dim=(1, 2))
    return -mse


def rank_candidates(candidates: Tensor, batch: AlignmentBatch, rank_fn: Optional[RankFn] = None) -> tuple[Tensor, Tensor]:
    """Rank candidates in descending score order."""
    if rank_fn is None:
        scores = reference_fit_ranker(candidates, batch)
    else:
        scores = rank_fn(candidates, batch)
    order = torch.argsort(scores, descending=True)
    return candidates[order], scores[order]


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

    # Demo input (replace with real preprocessed task data).
    batch, _ = make_synthetic_alignment_batch(batch_size=1, n_atoms=args.n_atoms, device=device)
    if args.use_pocket_guidance:
        batch.pocket_pos = batch.reference_pos + 0.1 * torch.randn_like(batch.reference_pos)

    source_sampler = lambda b: fm.sample_source(b)  # noqa: E731
    guidance_fn = make_pocket_pull_guidance(scale=1.0) if args.use_pocket_guidance else None

    candidates = generate_candidates(
        sampler=sampler,
        batch=batch,
        source_sampler=source_sampler,
        guidance_fn=guidance_fn,
        num_samples=args.num_samples,
    )
    ranked, scores = rank_candidates(candidates=candidates, batch=batch)

    print(f"[inference] candidates={candidates.shape}")
    print(f"[inference] top_score={scores[0].item():.6f}")

    if args.save_path:
        torch.save(
            {
                "candidates": ranked.cpu(),
                "scores": scores.cpu(),
                "metadata": {
                    "checkpoint": args.checkpoint,
                    "model_args": model_args,
                    "flow_args": flow_args,
                    "sampler_args": sampler_args,
                    "guidance_used": bool(args.use_pocket_guidance and args.guidance_scale > 0.0),
                    "num_samples": args.num_samples,
                    "n_atoms": args.n_atoms,
                    "training_step": ckpt.get("step"),
                },
            },
            args.save_path,
        )
        print(f"[inference] saved to: {args.save_path}")


def _build_inference_runtime(args: argparse.Namespace, device: torch.device):
    """Load checkpoint and construct model/flow/sampler runtime objects."""
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_args = ckpt.get("model_args", {})
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
    }
    sampler = ETFlowAlignSampler(model=model, config=ODESamplerConfig(**sampler_args))
    return ckpt, model_args, flow_args, sampler_args, sampler, fm


def build_argparser() -> argparse.ArgumentParser:
    """Define CLI arguments for inference script."""
    p = argparse.ArgumentParser(description="Run ETFlowAlign inference on synthetic demo data.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--n-atoms", type=int, default=16)
    p.add_argument("--n-steps", type=int, default=64)
    p.add_argument("--solver", type=str, default="heun", choices=["euler", "heun"])
    p.add_argument("--guidance-scale", type=float, default=0.0)
    p.add_argument("--guidance-mode", type=str, default="vector_field", choices=["vector_field", "predictor_corrector"])
    p.add_argument("--max-guidance-norm", type=float, default=5.0)
    p.add_argument("--use-pocket-guidance", action="store_true")
    p.add_argument("--save-path", type=str, default="")
    return p


def main() -> None:
    """CLI main."""
    args = build_argparser().parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
