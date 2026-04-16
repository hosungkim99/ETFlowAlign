# ETFlowAlign

ETFlowAlign is a **flow-matching re-design of DiffAlign** for flexible molecular alignment.

It preserves DiffAlign's task contract (query ligand alignment conditioned on a reference ligand and optional pocket context) while replacing diffusion-style denoising dynamics with an ET-Flow-style time-dependent vector field and ODE sampling.

---

## Design summary

Start here:
1. `model.py` for the overall architecture
2. `flow_matching.py` for the training objective
3. `sampler.py` for inference dynamics
# ETFlowAlign

## flow_matching.py
ETFlowAlign is a **flow-matching re-design of DiffAlign** for flexible molecular alignment.

여기에는:
It preserves DiffAlign's task contract (query ligand alignment conditioned on a reference ligand and optional pocket context) while replacing diffusion-style denoising dynamics with an ET-Flow-style time-dependent vector field and ODE sampling.

ET-Flow 기반 objective
---

path definition
## Design summary

velocity target
```text
ETFlowAlign = DiffAlign task framework
            - diffusion reverse process (DDPM/DDIM-style)
            + flow matching objective + ODE integration
```

time sampling
### Preserved from DiffAlign

training target 생성
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

## inference.py
여기에는:
## File guide

inference entry point
Read in this order:

evaluation/inference pipeline
1. `model.py` – equivariant vector-field model
2. `flow_matching.py` – path/source/target and training objective
3. `sampler.py` – ODE integration and guidance injection
4. `train.py` – runnable training script (synthetic demo)
5. `inference.py` – runnable inference script + ranking adapter

를 둔다.
`utils.py` remains intentionally lightweight.

---

## model.py
가장 중요하다.
## Review points: current scaffold limitations

여기에는:
1. **Backbone simplification**: uses a compact EGNN-style block, but not yet a full production molecular transformer.
2. **Synthetic data only**: training/inference scripts currently run on synthetic alignment batches for smoke testing.
3. **Guidance placeholder**: pocket guidance is provided as a safe hook with clipping, not yet full UFF physics integration.
4. **Ranking placeholder**: ranking adapter defaults to a simple geometric score instead of production metrics (e.g., TanimotoCombo + docking score).
5. **No benchmark pipeline yet**: dataset preprocessing/evaluation scripts for real alignment benchmarks are pending.

ETFlowAlign 전체 모델 구조
---

DiffAlign에서 유지한 부분
## Next-commit concrete checklist (per file)

ET-Flow로 바꾼 부분
### `model.py`
- [ ] Replace compact EGNN-style block with a stronger E(3)-equivariant transformer-style backbone.
- [ ] Add richer conditioning channels (reference atom features, cross-graph attention).
- [ ] Add optional chirality-aware auxiliary head.

forward 흐름
### `flow_matching.py`
- [ ] Add alignment-aware source distributions beyond Gaussian/reference COM (e.g., rigidly perturbed reference-driven prior).
- [ ] Add alternative path families and ablation flags.
- [ ] Add robust weighting / curriculum over time samples.

을 넣는다.
### `sampler.py`
- [ ] Add adaptive-step ODE solver option.
- [ ] Implement UFF/pocket guidance with predictor-corrector stability safeguards.
- [ ] Add trajectory logging for debugging stiff dynamics.

이 파일 하나만 봐도
“아, 이 모델이 어떻게 생겼는지”
알 수 있어야 한다.
### `train.py`
- [ ] Replace synthetic batch generator with real dataset/datamodule.
- [ ] Add validation loop and checkpoint-by-metric selection.
- [ ] Add distributed and mixed-precision training support.

## sampler.py
여기에는:
### `inference.py`
- [ ] Replace toy ranker with pluggable TanimotoCombo + docking/physics rank adaptor.
- [ ] Add batch inference over benchmark sets and structured output export.
- [ ] Add reranking ensemble hooks.

inference / generation / integration loop
### `utils.py`
- [ ] Keep only non-core helpers; avoid moving algorithmic logic here.

Euler / ODE step
---

iterative update
## Runnable scripts

를 둔다.
### Train (synthetic smoke test)

즉 “학습된 flow를 가지고 실제로 어떻게 샘플을 얻는가”를 정리한다.
```bash
python -m etflowalign/train.py \
  --steps 200 \
  --batch-size 8 \
  --n-atoms 16 \
  --save-path etflowalign_ckpt.pt
```

### Inference (from trained checkpoint)

---

## Review points: current scaffold limitations

1. **Backbone simplification**: uses a compact EGNN-style block, but not yet a full production molecular transformer.
2. **Synthetic data only**: training/inference scripts currently run on synthetic alignment batches for smoke testing.
3. **Guidance placeholder**: pocket guidance is provided as a safe hook with clipping, not yet full UFF physics integration.
4. **Ranking plugin baseline**: inference now exposes a pluggable TanimotoCombo+physics ranker, but production docking engines/ROCS plugins still need to be wired for benchmark-grade scoring.
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

```bash
python -m etflowalign/inference.py \
  --checkpoint etflowalign_ckpt.pt \
  --num-samples 16 \
  --n-steps 64 \
  --solver heun \
  --guidance-scale 0.2 \
  --use-pocket-guidance \
  --save-path etflowalign_samples.pt

하지만 너무 많은 핵심 로직을 여기 숨기면 안 된다.
1. **Scaffold phase (current)**
   - Provide clean interfaces and explicit TODOs.
   - Keep code minimal and self-contained.
2. **Backbone phase**
   - Replace placeholder vector-field backbone with a full E(3)-equivariant architecture.
3. **Data/task phase**
   - Connect real molecular graph featurization and batching.
4. **Guidance phase**
   - Implement robust pocket/UFF guidance with numerical safeguards.
5. **Evaluation phase**
   - Integrate candidate ranking and benchmark scripts.

This staged approach avoids premature over-engineering while making the final ETFlowAlign design easy to audit from `etflowalign/` alone.


Start here:
1. `model.py` for the overall architecture
2. `flow_matching.py` for the training objective
3. `sampler.py` for inference dynamics

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
- [x] Replace toy ranker with pluggable TanimotoCombo + docking/physics rank adaptor.
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
  --guidance-backend uff \
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


### Ranking backend notes

- `--ranker plugin_combo` (default): combines a TanimotoCombo-like proxy score and docking/physics proxy score.
- `--ranker legacy_reference_mse`: compatibility mode using the old negative-reference-MSE score.
- Saved inference artifacts now include `component_scores` (`tanimoto`, `physics`) plus ranker metadata.
- For production use, inject external plugin callbacks (e.g., ROCS/OpenEye + docking engine) through `PluginRanker`.
