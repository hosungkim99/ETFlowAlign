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
Conceptually:

time sampling
```text
ETFlowAlign = DiffAlign task framework
            - diffusion reverse process (DDPM/DDIM-style)
            + flow matching objective + ODE integration
```

training target 생성
### What is preserved from DiffAlign

을 넣는다.
1. **Task definition**: query ligand alignment under reference context.
2. **Reference-conditioned setup**: conditioning remains first-class in the model API.
3. **Cartesian coordinates**: generation happens directly in atom-level 3D coordinates.
4. **E(3)-equivariant contract**: the model predicts a per-atom vector field.
5. **Pocket-aware guidance philosophy**: pocket or physics terms are optional inference-time steering signals.
6. **Multi-sample + ranking compatibility**: inference supports generating multiple candidates for downstream ranking (e.g., TanimotoCombo).

즉 diffusion loss를 대체하는 수학적 핵심을 둔다.
### What is imported from ET-Flow

1. **Flow matching training objective** (vector field regression).
2. **Time-dependent vector field model** `v_theta(x_t, t, cond)`.
3. **Interpolation-based probability path** between source and target coordinates.
4. **ODE-based sampling** instead of diffusion reverse updates.
5. Optional ideas retained as extension points:
   - stochastic sampling,
   - chirality correction,
   - stronger equivariant backbones.

## inference.py
여기에는:
---

inference entry point
## Component map

evaluation/inference pipeline
Read these files first:

를 둔다.
1. `model.py`: ETFlowAlign model interface and conditional vector-field contract.
2. `flow_matching.py`: source distribution, path sampling, time sampling, and loss.
3. `sampler.py`: ODE inference loop and optional guidance injection.
4. `train.py`: minimal training step wiring.
5. `inference.py`: multi-sample inference entrypoint.

`utils.py` intentionally only stores small reusable helpers; core algorithmic logic is kept in the three core modules above.

## model.py
가장 중요하다.
---

여기에는:
## Upstream provenance and redesign notes

ETFlowAlign 전체 모델 구조
This implementation is informed by:

DiffAlign에서 유지한 부분
- `external/diffalign/` (task framing, reference conditioning, pocket-aware guidance intent).
- `external/etflow/` (flow matching objective, time-dependent vector field, ODE sampling structure).

ET-Flow로 바꾼 부분
### DiffAlign components by migration type

forward 흐름
**Preserved conceptually**
- Alignment task and conditioning semantics.
- Coordinate-space generation over query atoms.
- Equivariant prediction target over coordinates.

을 넣는다.
**Replaced entirely**
- Diffusion schedules, epsilon/v parameterization, reverse diffusion recursion.
- DDPM/DDIM-specific training and sampling equations.

이 파일 하나만 봐도
“아, 이 모델이 어떻게 생겼는지”
알 수 있어야 한다.
**Redesigned partially**
- Source/base state for alignment (not copied from conformer-generation prior as-is).
- Probability path and target field for alignment-specific generation.
- Guidance injection policy under ODE integration (stability-aware scaling/clipping).

## sampler.py
여기에는:
---

inference / generation / integration loop
## Initial roadmap (this scaffold)

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

