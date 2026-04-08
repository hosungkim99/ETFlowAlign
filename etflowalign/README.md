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

을 넣는다.
### Imported from ET-Flow

즉 diffusion loss를 대체하는 수학적 핵심을 둔다.
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
python -m etflowalign.train \
  --steps 200 \
  --batch-size 8 \
  --n-atoms 16 \
  --save-path etflowalign_ckpt.pt
```

### Inference (from trained checkpoint)

## train.py
여기에는:

training step

loss 호출

optimizer step

batch 처리

를 둔다.

## utils.py
공통 유틸은 여기에 모은다.

하지만 너무 많은 핵심 로직을 여기 숨기면 안 된다.

```bash
python -m etflowalign.inference \
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

## flow_matching.py

여기에는:

ET-Flow 기반 objective

path definition

velocity target

time sampling

training target 생성

을 넣는다.

즉 diffusion loss를 대체하는 수학적 핵심을 둔다.


## inference.py
여기에는:

inference entry point

evaluation/inference pipeline

를 둔다.


## model.py
가장 중요하다.

여기에는:

ETFlowAlign 전체 모델 구조

DiffAlign에서 유지한 부분

ET-Flow로 바꾼 부분

forward 흐름

을 넣는다.

이 파일 하나만 봐도
“아, 이 모델이 어떻게 생겼는지”
알 수 있어야 한다.

## sampler.py
여기에는:

inference / generation / integration loop

Euler / ODE step

iterative update

를 둔다.

즉 “학습된 flow를 가지고 실제로 어떻게 샘플을 얻는가”를 정리한다.


## train.py
여기에는:

training step

loss 호출

optimizer step

batch 처리

를 둔다.

## utils.py
공통 유틸은 여기에 모은다.

하지만 너무 많은 핵심 로직을 여기 숨기면 안 된다.

