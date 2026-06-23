"""Inference script and APIs for ETFlowAlign."""

from __future__ import annotations

import argparse
from typing import Any,Callable, Optional

import torch
from torch import Tensor

from .data import load_alignment_batch_from_pt
from .flow_matching import AlignmentFlowMatcher, FlowMatchingConfig
from .model import AlignmentBatch, ETFlowAlignModel
from .sampler import ETFlowAlignSampler, GuidanceFn, ODESamplerConfig
from .train import make_synthetic_alignment_batch
from .validation import validate_alignment_batch

RankFn = Callable[[Tensor, AlignmentBatch], Tensor]


def generate_candidates(
    sampler: ETFlowAlignSampler,
    batch: AlignmentBatch,
    source_sampler: Callable[[AlignmentBatch], Tensor],
    guidance_fn: Optional[GuidanceFn] = None,
    num_samples: int = 8,
) -> Tensor:
    """Generate aligned candidates with strict interface checks."""
    validate_alignment_batch(batch)
    outputs: list[Tensor] = []
    for _ in range(num_samples):
        x0 = source_sampler(batch)
        if x0.shape != batch.query_pos.shape:
            raise ValueError(
                "source_sampler must return query_pos-matched shape "
                f"{tuple(batch.query_pos.shape)}, got {tuple(x0.shape)}."
            )
        outputs.append(sampler.sample(batch=batch, x0=x0, guidance_fn=guidance_fn))
    return torch.stack(outputs, dim=0)


def rank_candidates(
    candidates: Tensor,
    batch: AlignmentBatch,
    rank_fn: Optional[RankFn] = None,
) -> tuple[Tensor, Tensor]:
    """Rank candidates with v0.1 one-complex-per-call validation."""
    num_graphs = int(batch.query_batch.max().item()) + 1 if batch.query_batch.numel() else 0
    if num_graphs != 1:
        raise ValueError("v0.1 ranking supports exactly one complex per inference call.")

    scores = rank_fn(candidates, batch) if rank_fn is not None else torch.zeros(candidates.size(0), device=candidates.device)
    if scores.dim() != 1 or scores.size(0) != candidates.size(0):
        raise ValueError("rank_fn must return Tensor[num_samples].")

    order = torch.argsort(scores, descending=True)
    return candidates[order], scores[order]


def run_inference(args: argparse.Namespace) -> None:
    """CLI inference entrypoint."""
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)

    model = ETFlowAlignModel(**ckpt.get("model_args", {})).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    flow_args = ckpt.get("flow_args", {})
    flow_matcher = AlignmentFlowMatcher(FlowMatchingConfig(**flow_args))
    sampler = ETFlowAlignSampler(
        model=model,
        config=ODESamplerConfig(
            n_steps=args.n_steps,
            solver=args.solver,
            guidance_scale=args.guidance_scale,
            guidance_mode=args.guidance_mode,
            max_guidance_norm=args.max_guidance_norm,
        ),
    )

    input_metadata: dict[str, Any] = {}
    if args.synthetic_smoke:
        batch, _ = make_synthetic_alignment_batch(batch_size=1, n_atoms=args.n_atoms, device=device)
    else:
        batch, _, input_metadata = load_alignment_batch_from_pt(
            args.input_batch,
            require_target=False,
            device=device,
        )

    candidates = generate_candidates(
        sampler=sampler,
        batch=batch,
        source_sampler=lambda b: flow_matcher.sample_source(b),
        guidance_fn=None,
        num_samples=args.num_samples,
    )
    ranked, scores = rank_candidates(candidates, batch)

    print(f"[inference] candidates={ranked.shape}")
    print(f"[inference] top_score={scores[0].item():.6f}")

    if args.save_path:
        run_metadata = {
            "checkpoint": args.checkpoint,
            "input_batch": args.input_batch if args.input_batch else None,
            "synthetic_smoke": bool(args.synthetic_smoke),
            "num_samples": int(args.num_samples),
            "n_steps": int(args.n_steps),
            "solver": args.solver,
            "guidance_scale": float(args.guidance_scale),
            "guidance_mode": args.guidance_mode,
            "source_type": flow_args.get("source_type"),
            "input_metadata": input_metadata,
        }
        torch.save(
            {
                "candidates": ranked.cpu(),
                "scores": scores.cpu(),
                "metadata": run_metadata,
            },
            args.save_path,
        )
        print(f"[inference] saved to: {args.save_path}")


def build_argparser() -> argparse.ArgumentParser:
    """Define inference CLI."""
    p = argparse.ArgumentParser(description="Run ETFlowAlign inference.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--n-atoms", type=int, default=16)
    p.add_argument("--n-steps", type=int, default=64)
    p.add_argument("--solver", type=str, default="heun", choices=["euler", "heun"])
    p.add_argument("--guidance-scale", type=float, default=0.0)
    p.add_argument("--guidance-mode", type=str, default="vector_field", choices=["vector_field", "predictor_corrector"])
    p.add_argument("--max-guidance-norm", type=float, default=5.0)

    p.add_argument("--synthetic-smoke", action="store_true")
    p.add_argument("--input-batch", type=str, default="")
    p.add_argument("--save-path", type=str, default="")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if args.synthetic_smoke == bool(args.input_batch):
        raise ValueError("Provide exactly one of --synthetic-smoke or --input-batch.")
    run_inference(args)


if __name__ == "__main__":
    main()