# ETFlowAlign

ETFlowAlign keeps the DiffAlign task framing, but swaps diffusion reverse dynamics for flow matching + ODE integration.

## Implemented now (scaffold)
- Flow-matching training step with configurable source sampling.
- Optional Kabsch source-to-target rigid alignment in training path.
- Optional harmonic prior regularization (safe no-op when strength <= 0).
- Strict batch validation before forward/sampling.
- `.pt` batch loading for training/inference.
- ODE sampling inference and candidate generation.

## Planned (future)
- Production-scale ranking/guidance backends.
- Large-batch multi-complex inference/ranking.

## CLI

### Synthetic smoke tests
```bash
python -m etflowalign.train --synthetic-smoke --steps 2 --batch-size 2 --n-atoms 8 --save-path /tmp/etflowalign_smoke.pt
python -m etflowalign.inference --synthetic-smoke --checkpoint /tmp/etflowalign_smoke.pt --num-samples 4 --n-steps 8
```
`infer_batch.pt` must include: `query_pos`, `query_atom_type`, `query_batch`.
`target_query_pos` is not required.

### Real `.pt` batch training
```bash
python -m etflowalign.train --train-data /path/to/train_batch.pt --steps 1 --save-path /tmp/etflowalign_ckpt.pt
```
`train_batch.pt` must include: `query_pos`, `query_atom_type`, `query_batch`, `target_query_pos`. Optional: `reference_*`, `pocket_*`, metadata.

### Real `.pt` batch inference
```bash
python -m etflowalign.inference --checkpoint /tmp/etflowalign_ckpt.pt --input-batch /path/to/infer_batch.pt --num-samples 4 --n-steps 8
```
`infer_batch.pt` must include: `query_pos`, `query_atom_type`, `query_batch`. `target_query_pos` is not required.


### Debug overfit sanity result

- Batch: DiffAlign example pseudo train batch
- Source: input_query
- Target: query_sdf_conformer_minus_reference_center
- Model mode:
  - use_atom_index_embed=True
  - use_direct_vector_head=True
- Flow:
  - sigma=0.0
  - center_source=False
  - center_target=False
  - use_kabsch_alignment=False
- fixed_t=0.5:
  - best_loss ≈ 4.4e-08
  - t=0.5 RMSE ≈ 0.0002 Å
- random t:
  - best_loss ≈ 0.0030
  - all-t diagnostic RMSE < 0.5 Å
- inference/SDF:
  - atomwise RMSD to query_original ≈ 0.213 Å
  - COM distance ≈ 0.062 Å
  - bond length mean ≈ 1.455 Å

## v0.1 interface notes
- `GuidanceFn` must return `Tensor[Nq, 3]` exactly.
- Ranking currently validates and supports one complex per inference call.
