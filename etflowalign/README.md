# ETFlowAlign

ETFlowAlign is a **flow-matching re-design of DiffAlign** for flexible molecular alignment.

It preserves DiffAlign's task contract (query ligand alignment conditioned on a reference ligand and optional pocket context) while replacing diffusion-style denoising dynamics with an ET-Flow-style time-dependent vector field and ODE sampling.

---

## Design summary

```text
ETFlowAlign = DiffAlign task framework
            - diffusion reverse process (DDPM/DDIM-style)
            + flow matching objective + ODE integration
```

### Preserved from DiffAlign

1. Query ligand alignment task under reference context.
2. Reference-conditioned generation setup.
3. Direct Cartesian coordinate generation.
4. Equivariant vector-field output contract.
5. Pocket-aware guidance as an inference-time steering signal.
6. Multi-sample generation + ranking compatibility.

### Imported from ET-Flow

1. Flow-matching vector-field regression objective.
2. Continuous-time vector field `v_theta(x_t, t, cond)`.
3. Interpolation probability path between source and target states.
4. ODE solver-based sampling.

---

## File guide

Read in this order:

1. `model.py` – equivariant vector-field model
2. `flow_matching.py` – path/source/target and training objective
3. `sampler.py` – ODE integration and guidance injection
4. `train.py` – runnable training script (synthetic demo)
5. `inference.py` – runnable inference script + ranking adapter

`utils.py` remains intentionally lightweight.

---

## Environment setup

`etflowalign/` now includes its own environment file:

- `env.yml` (Conda environment for ETFlowAlign scripts)
- `setup_env.sh` (CPU/CUDA-aware setup script)

The environment specification was assembled by referencing:

- `external/diffalign/diffalign/env.yml`
- `external/etflow/env.yml`

Create and activate:

```bash
cd etflowalign
bash setup_env.sh --mode cuda
conda activate etflowalign
```

CPU-only install:

```bash
cd etflowalign
bash setup_env.sh --mode cpu
conda activate etflowalign
```

Reproducible UFF install (pin tag/commit):

```bash
cd etflowalign
bash setup_env.sh --mode cuda --uff-ref <commit_or_tag>
```

Optional post-install smoke test:

```bash
cd etflowalign
bash setup_env.sh --mode cpu --smoke-test
```

Then run scripts from repository root (or with `python -m ...` from root):

```bash
python -m etflowalign.train --help
python -m etflowalign.inference --help
```

---

## Review points: current scaffold limitations

1. **Backbone simplification**: uses a compact EGNN-style block, but not yet a full production molecular transformer.
2. **Synthetic data only**: training/inference scripts currently run on synthetic alignment batches for smoke testing.
3. **Guidance placeholder**: pocket guidance is provided as a safe hook with clipping, not yet full UFF physics integration.
4. **Ranking placeholder**: ranking adapter defaults to a simple geometric score instead of production metrics (e.g., TanimotoCombo + docking score).
5. **No benchmark pipeline yet**: dataset preprocessing/evaluation scripts for real alignment benchmarks are pending.

---

## Next-commit concrete checklist (per file)

### `model.py`
- [ ] Replace compact EGNN-style block with a stronger E(3)-equivariant transformer-style backbone.
- [ ] Add richer conditioning channels (reference atom features, cross-graph attention).
- [ ] Add optional chirality-aware auxiliary head.

### `flow_matching.py`
- [ ] Add alignment-aware source distributions beyond Gaussian/reference COM (e.g., rigidly perturbed reference-driven prior).
- [ ] Add alternative path families and ablation flags.
- [ ] Add robust weighting / curriculum over time samples.

### `sampler.py`
- [ ] Add adaptive-step ODE solver option.
- [ ] Implement UFF/pocket guidance with predictor-corrector stability safeguards.
- [ ] Add trajectory logging for debugging stiff dynamics.

### `train.py`
- [ ] Replace synthetic batch generator with real dataset/datamodule.
- [ ] Add validation loop and checkpoint-by-metric selection.
- [ ] Add distributed and mixed-precision training support.

### `inference.py`
- [ ] Replace toy ranker with pluggable TanimotoCombo + docking/physics rank adaptor.
- [ ] Add batch inference over benchmark sets and structured output export.
- [ ] Add reranking ensemble hooks.

### `utils.py`
- [ ] Keep only non-core helpers; avoid moving algorithmic logic here.

---

## Runnable scripts

### Train (synthetic smoke test)

```bash
python -m etflowalign.train \
  --steps 200 \
  --batch-size 8 \
  --n-atoms 16 \
  --save-path etflowalign_ckpt.pt
```

### Train (real task batch file)

```bash
python -m etflowalign.train \
  --train-data /path/to/train_batch.pt \
  --val-data /path/to/val_batch.pt \
  --val-every 20 \
  --use-scheduler \
  --save-path etflowalign_ckpt.pt
```

Real-task `.pt` batch expected keys:
- required: `query_pos`, `query_atom_type`, `query_batch`, `target_query_pos`
- optional: `reference_pos`, `reference_atom_type`, `reference_batch`, `pocket_pos`,
  `query_node_attr`, `reference_node_attr`

### Inference (from trained checkpoint)

```bash
python -m etflowalign.inference \
  --checkpoint etflowalign_ckpt.pt \
  --num-samples 16 \
  --n-steps 64 \
  --solver heun \
  --guidance-scale 0.2 \
  --use-pocket-guidance \
  --save-path etflowalign_samples.pt
```

### Inference (real task batch file)

```bash
python -m etflowalign.inference \
  --checkpoint etflowalign_ckpt.pt \
  --input-batch /path/to/infer_batch.pt \
  --num-samples 32 \
  --top-k 8 \
  --adaptive-dt \
  --save-path etflowalign_samples.pt
```
