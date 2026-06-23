"""Shared pose-RMSD evaluation, used by the eval script and by in-training monitoring.

Given a model and a list of per-complex ``.pt`` paths, sample a pose from each stored
source and measure atomwise RMSD to the target (same frame, known correspondence).
"""

from __future__ import annotations

import statistics

import torch

from .data import load_alignment_batch_from_pt
from .dataset import Conditioning, apply_conditioning
from .sampler import ETFlowAlignSampler, ODESamplerConfig


def _rmsd(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a - b) ** 2).sum(dim=-1).mean().sqrt())


def summarize(rows: list[dict]) -> dict:
    """Aggregate per-complex rows into summary statistics."""
    if not rows:
        return {"n": 0}
    pred = [r["pred_rmsd"] for r in rows]
    src = [r["source_rmsd"] for r in rows]
    n = len(pred)
    return {
        "n": n,
        "source_rmsd_mean": statistics.mean(src),
        "source_rmsd_median": statistics.median(src),
        "pred_rmsd_mean": statistics.mean(pred),
        "pred_rmsd_median": statistics.median(pred),
        "frac_lt2": sum(p < 2.0 for p in pred) / n,
        "frac_lt5": sum(p < 5.0 for p in pred) / n,
        "improved_frac": sum(p < s for p, s in zip(pred, src)) / n,
    }


def format_summary(s: dict) -> str:
    if s.get("n", 0) == 0:
        return "no complexes evaluated."
    return (
        f"n={s['n']} | source mean={s['source_rmsd_mean']:.3f} med={s['source_rmsd_median']:.3f}"
        f" | pred mean={s['pred_rmsd_mean']:.3f} med={s['pred_rmsd_median']:.3f}"
        f" | <2A={s['frac_lt2'] * 100:.1f}% <5A={s['frac_lt5'] * 100:.1f}% improved={s['improved_frac'] * 100:.1f}%"
    )


@torch.no_grad()
def evaluate_paths(
    model,
    paths: list[str],
    *,
    conditioning: Conditioning = "pocket",
    n_steps: int = 50,
    solver: str = "heun",
    device: str | torch.device = "cpu",
    limit: int = 0,
) -> tuple[dict, list[dict]]:
    """Run pose sampling on each complex and return (summary, per-complex rows).

    The caller is responsible for the model's train/eval mode; sampling itself runs
    under ``no_grad``.
    """
    if limit and limit > 0:
        paths = paths[:limit]
    sampler = ETFlowAlignSampler(model, ODESamplerConfig(n_steps=n_steps, solver=solver))

    rows: list[dict] = []
    for path in paths:
        try:
            batch, target, meta = load_alignment_batch_from_pt(path, require_target=True, device=device)
        except Exception:  # noqa: BLE001 - evaluation must skip unreadable complexes
            continue
        batch = apply_conditioning(batch, conditioning)
        x0 = batch.query_pos
        pred = sampler.sample(batch=batch, x0=x0.clone())

        row = {
            "pdb_id": meta.get("pdb_id"),
            "n_atoms": int(x0.size(0)),
            "source_rmsd": _rmsd(x0, target),
            "pred_rmsd": _rmsd(pred, target),
            "com_dist": float((pred.mean(0) - target.mean(0)).norm()),
        }
        if batch.query_bond_index is not None and batch.query_bond_index.numel() > 0:
            bi = batch.query_bond_index
            bl = torch.linalg.norm(pred[bi[0]] - pred[bi[1]], dim=-1)
            row.update(bond_min=float(bl.min()), bond_mean=float(bl.mean()), bond_max=float(bl.max()))
        rows.append(row)

    return summarize(rows), rows
