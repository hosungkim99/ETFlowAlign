# ETFlowAlign

ETFlowAlign keeps the DiffAlign task framing, but swaps diffusion reverse dynamics for flow matching + ODE integration.

## Implemented now (scaffold)
- Flow-matching training step with configurable source sampling.
- Optional Kabsch source-to-target rigid alignment in training path.
- Optional harmonic prior regularization (safe no-op when strength <= 0).
- Strict batch validation before forward/sampling.
- `.pt` batch loading for training/inference.
- ODE sampling inference and candidate generation.

## Current placeholders (v0.1)
- `UFFPocketGuidance` is a **batch-safe surrogate** guidance implementation.
  It is **not full UFF**, and **not production-grade pocket-aware guidance** yet.
- CLI inference ranking currently uses zero scores unless a Python `rank_fn` is provided programmatically.
- Inference CLI supports `--save-path` to save `{candidates, scores, metadata}`.

## Planned (future)
- Production-scale ranking/guidance backends.
- Large-batch multi-complex inference/ranking.

## CLI

### Synthetic smoke tests
```bash
python -m etflowalign.train --synthetic-smoke --steps 2 --batch-size 2 --n-atoms 8 --save-path /tmp/etflowalign_smoke.pt
python -m etflowalign.inference --synthetic-smoke --checkpoint /tmp/etflowalign_smoke.pt --num-samples 4 --n-steps 8 --save-path /tmp/etflowalign_infer.pt
```

### Real `.pt` batch training
```bash
python -m etflowalign.train --train-data /path/to/train_batch.pt --steps 1 --save-path /tmp/etflowalign_ckpt.pt
```
`train_batch.pt` must include: `query_pos`, `query_atom_type`, `query_batch`, `target_query_pos`.
Optional: `reference_*`, `pocket_*`, node attributes, metadata.

### Real `.pt` batch inference
```bash
python -m etflowalign.inference --checkpoint /tmp/etflowalign_ckpt.pt --input-batch /path/to/infer_batch.pt --num-samples 4 --n-steps 8 --save-path /tmp/etflowalign_candidates.pt
```
`infer_batch.pt` must include: `query_pos`, `query_atom_type`, `query_batch`.
`target_query_pos` is not required.

## v0.1 interface notes
- `GuidanceFn` must return `Tensor[Nq, 3]` exactly.
- Ranking currently validates and supports one complex per inference call.
