# ETFlowAlign — 설계 명세 (SPEC)

> 이 문서는 ETFlowAlign의 **캐논 명세**다. 코드가 이 문서와 어긋나면 코드가 틀린 것이다.
> 백지에서 새로 구현한다. 과거 `legacy_etflowalign/`은 참고만 하고 의존하지 않는다.

---

## 0. 한 문장 정의

> harmonic prior에서 뽑은 노이즈 `x0`를, **reference 리간드 형상**을 조건으로,
> **등변 벡터-피처 백본이 예측한 속도장 `v_θ`를 ODE로 적분**하여
> reference에 정렬된 query 3D 구조 `x1`로 흘려보낸다.

- **태스크는 DiffAlign에서**: reference 조건부 flexible 분자 정렬 생성.
- **엔진은 ET-Flow에서**: flow matching + ODE, harmonic prior, 등변 트랜스포머 백본.

원본 위치: `external/diffalign/`, `external/etflow/` (둘 다 MIT).

---

## 1. 태스크 정의

| | 내용 |
|---|---|
| **입력** | query 분자 그래프(원자종 `z`, 결합 `bonds`), reference 리간드의 3D 형상 |
| **출력** | reference 형상에 정렬된 query의 3D 좌표 `x1 ∈ R^{Nq×3}` |
| **생성 방식** | 노이즈 `x0`(harmonic prior) → 구조 `x1`, flexible (conformation + pose 동시 탐색) |
| **NOT** | rigid placement 아님, torsion-only 아님, 포켓을 네트워크에 넣지 않음 |

**정렬(alignment)의 의미**: query의 내부 conformation과 전역 pose를 함께 생성하되,
결과가 reference 리간드의 3D 형상(shape)과 겹치도록 만든다. 학습 타깃 `x1`은
데이터셋에서 이미 reference에 정렬되어 있는 query conformer다.

---

## 2. 조건화 규칙 (엄격)

1. **네트워크가 보는 조건은 reference 리간드뿐이다.** query↔reference 메시지패싱으로 형상 정보를 주입한다.
2. **포켓(단백질)은 네트워크에 절대 넣지 않는다.** (DiffAlign 논문 2회 명시.)
   포켓은 **추론 시에만** `guidance` 모듈의 UFF 그래디언트로 ODE 궤적에 주입한다.
3. **reference 좌표를 절대값으로 넣지 않는다.** query↔ref 상대 거리/방향(등변량)으로만 넣는다.
   → 전체 joint roto-translation 등변성을 유지해야 한다 (§6 게이트로 강제).

---

## 3. 데이터 명세

- **출처**: GEOM-Drugs에서 뽑은 유사분자 정렬쌍.
  필터 = 2D Tanimoto 0.55–0.85 AND 3D shape overlap > 0.5.
- **한 샘플** = `(reference 분자, reference에 정렬된 query conformer)`.
- **규모 목표**: ~65k 쌍 (과거 라인에서 서버에 빌드 이력 있음, 이번엔 새로 검증).

### 3.1 샘플 텐서 계약 (per-sample)

| 필드 | 모양 | 설명 |
|---|---|---|
| `z` | `[Nq]` long | query 원자종 |
| `bonds` | `[2, 2Eq]` long | query 결합 (양방향 저장) |
| `bond_type` | `[2Eq]` long | 결합 차수 코드 (bonds와 정렬) |
| `pos` | `[Nq, 3]` float | query 좌표 = **학습 타깃 x1** (reference에 정렬된 conformer) |
| `ref_z` | `[Nr]` long | reference 원자종 |
| `ref_pos` | `[Nr, 3]` float | reference 좌표 (조건) |
| `mol_id` | str | 추적용 메타 |

배치는 PyG 스타일 `batch` 인덱스로 여러 샘플을 연결한다.
(코드에서 타깃 좌표의 키 이름은 `pos`이며, 개념상 x1과 동일하다.)

### 3.2 데이터 게이트 (Phase 1)
- 무작위 몇 쌍을 SDF로 내보내 **reference와 query(pos)가 실제로 형상 정렬되어 있는지 눈으로 확인**한다.
- 데이터가 틀리면 모델은 배울 수 없다 — 이 게이트를 통과하기 전엔 학습 코드로 넘어가지 않는다.

---

## 4. 생성 엔진 명세 (ET-Flow 계열)

### 4.1 Prior / Source
- **harmonic prior**: 분자 결합 그래프의 라플라시안 기반. 시작점부터 대략적 결합거리(~1.5Å) 유지.
- `x0 ~ harmonic(bonds)`. (게이트 A 이전엔 gaussian도 fallback으로 둔다.)

### 4.2 Path (보간 경로) + source 정렬
- 기본 = linear: `x_t = (1-t)·x0 + t·x1`, 목표 속도 `u = x1 - x0`.
- **source→target Kabsch 정렬 (비조건 모드 필수, `FlowConfig.kabsch_source_align`, 기본 on).**
  등변 백본은 무작위 방향 x0를 고정 방향 x1로 매핑할 수 없으므로(등변성 위반),
  x0를 x1에 회전정렬해 "모양 형성"만 학습시킨다. 생성 시 방향은 등변성이 처리
  (raw RMSD는 크게 나오는 게 정상, **aligned RMSD로 평가**). Gate A를 뚫은 핵심 장치.
- **조건부 모드에선 정렬 OFF** — reference가 프레임을 고정하므로 pose 자체를 학습.
  (과거 `pc@r` vs `pc@r.T` transpose 버그는 `kabsch_align_source_to_target` 자체검증으로 해결)

### 4.3 Loss
- `batchwise_l2`: 분자별 `||v_θ - u||` 평균 후 배치 평균. (ET-Flow와 동일.)
- **보조손실 없음** (bond/clash penalty 없음). DiffAlign도 auxiliary loss 없음.

### 4.4 Sampler
- 학습된 `v_θ`를 ODE로 적분 (Euler, `n_steps`). `t: 0 → 1`.
- eval-only 시 시작점 `x0`를 직접 주입 가능(진단). 포켓 UFF 가이던스 훅(추론)은 Phase 7.

### 4.5 학습 안정화 (train.py)
- **EMA** (가중치 지수이동평균, 기본 on): 생성엔 EMA 가중치 사용(FM 표준).
- **n_micro**: 스텝당 여러 `(x0,t)`를 평균해 gradient 분산↓(안정화/진단).
- **NaN 방어**: clip_grad_norm + nan_to_num, 연속 NaN 시 best 롤백.

---

## 5. 백본 명세 (등변 벡터-피처, self-contained)

ET-Flow의 torchmd-net은 로컬에서 0바이트 플레이스홀더 → **직접 재구현**한다.

### 5.1 원자당 상태
| 이름 | 모양 | 변환 | 역할 |
|---|---|---|---|
| `h` | `[N, F]` | 불변 | 스칼라 피처 |
| `vec` | `[N, 3, F]` | 등변 (`Rvec`) | 벡터 피처 |
| `pos` | `[N, 3]` | 등변 | 현재 `x_t` |

`F` = hidden channels (**기본 256**), `num_layers` = **기본 8**, `num_heads` = 8, `num_rbf` = 64, `cutoff` = 10.0Å.

### 5.2 등변 규칙 (모든 연산이 지켜야 함)
- 불변 스칼라끼리만 MLP/비선형에 자유롭게 넣는다.
- 벡터는 오직 (a) 불변 스칼라로 스케일, (b) 등변 단위방향 `û_ij`와 결합, (c) 두 벡터 내적으로 불변 생성 — 이 3가지로만 다룬다.

### 5.3 레이어 = 등변 어텐션 블록
1. 에지: `RBF(|r_ij|)`(expnorm) + 단위방향 `û_ij`, cosine cutoff 포락선.
2. 어텐션: `α_ij = SiLU(q_i·k_j ⊙ dk_edge)·cutoff`.
3. `Δh` = 값의 어텐션 집계 (불변).
4. `Δvec` = `gate ⊙ vec_j` + `scalar ⊙ û_ij` (방향 주입).
5. 잔차: `h += Δh`, `vec += Δvec`.
6. 업데이트 서브블록: `⟨U₁vec, U₂vec⟩ → 불변` 을 `h`에 피드백.

### 5.4 임베딩 / 출력
- 입력: `z → h`(Embedding), `vec = 0`.
- 시간 `t`: sinusoidal → MLP → `h`에 더함.
- reference: 추가 노드(`is_ref` 플래그, 불변) + query↔ref 메시지패싱(상대 기하만). 출력은 query 노드만.
- 출력 헤드: 벡터 피처 채널 가중합 → 원자당 속도 `v_θ ∈ R^{N×3}`.

### 5.5 안정화 3종 세트 (필수 — 과거 실증, 없으면 발산)
1. **출력 zero-init** — 마지막 사영 가중치 0 → 초기 속도장 ≈ 0.
2. **스칼라 LayerNorm** — 각 블록 어텐션 전 `h`.
3. **벡터 소프트정규화** — `vec ← vec / (1 + ‖vec‖)`.

---

## 6. 등변성 계약 (단위테스트로 강제)

임의 회전 `R`, 평행이동 `t`에 대해:
```
v_θ(R·x + t, R·ref + t) == R · v_θ(x, ref)   (오차 < 1e-5)
h(R·x, ...) == h(x, ...)                       (불변)
```
`tests/test_equivariance.py`가 이걸 검사한다. **조건화를 추가할 때마다 이 테스트를 다시 통과**해야 한다 (과거 조건화에서 등변성이 깨진 버그를 사전 차단).

---

## 7. 평가 명세

- **지표**: **heavy-atom Kabsch-aligned RMSD**(수소 제외 = 표준; GEOM 데이터는 H 없음),
  Top-1 RMSD @ 1/2/3Å, TanimotoCombo 랭킹. (raw RMSD는 방향 포함이라 참고용)
- **벤치마크**: DISCO식 (원본 DiffAlign 평가 프로토콜 참고), 샘플 다수(예 30) 생성 후 랭킹.
- **prior 대비**: 생성 결과가 시작 노이즈(prior)보다 나아야 의미 있음.

---

## 8. 마일스톤 & 게이트 (통과해야 다음으로)

| Phase | 산출물 | **게이트 (통과 기준)** |
|---|---|---|
| 0 | 이 SPEC.md | 정의 합의 |
| 1 | 데이터 파이프라인 | reference-x1 정렬 육안 확인 (§3.2) |
| 2 | 조건 없는 FM 코어 | **게이트 A**: 소수 분자에서 정상 결합길이의 3D 구조 생성 |
| 3 | 등변 백본 | **등변성 테스트** < 1e-5 (§6) |
| 4 | reference 조건화 | 등변성 테스트 재통과 |
| 5 | overfit-100 | **게이트 B**: median heavy-atom aligned RMSD < 2Å |
| 6 | 일반화 (65k + val) | **게이트 C**: val aligned RMSD < prior source RMSD |
| 7 | 평가 + 포켓 가이던스 | DISCO식 Top-1 RMSD 산출 |

**철학**: "loss 감소"가 아니라 "실제 생성 성공"을 게이트로 삼는다.
생성 모델은 loss가 내려가도 생성이 실패할 수 있다(과거의 핵심 교훈).

> ⚠️ **게이트 A 주의(실증됨):** 합성 소분자는 harmonic prior 분포와 거의 같아
> transport가 trivial 하므로, 게이트 A 통과는 **실분자 생성 능력의 검증이 아니다.**
> 실분자 생성의 진짜 시험은 Phase 5(overfit) 이후다.

---

## 9. 비목표 (Non-goals) — 과거 우회로 재발 방지

- ❌ rigid SE(3) placement (DiffAlign은 flexible 생성이다)
- ❌ 포켓을 denoiser/네트워크 입력으로 넣기 (추론 UFF 가이던스만 허용)
- ❌ PDBbind/CrossDocked 데이터 (DiffAlign은 GEOM-Drugs 정렬쌍)
- ❌ torsion-only 파라미터화
- ❌ bond/clash 보조손실

---

## 10. 디렉터리 구조

```
etflowalign/
├── SPEC.md                 # (이 문서) 설계 캐논
├── tree.txt                # Phase별 진행 추적
├── SERVER_RUN.md           # SLURM 실행 가이드
├── backbone/
│   ├── utils.py            # safe_norm/soft_normalize/scatter/radius_graph
│   ├── rbf.py              # expnorm RBF + cosine cutoff
│   ├── layers.py           # EquivariantAttention + GatedEquivariantBlock
│   ├── embedding.py        # 원자종/시간/reference(is_ref) 임베딩
│   └── network.py          # 블록 스택 → 속도장 헤드 (조건부 forward)
├── flow/
│   ├── prior.py            # harmonic/gaussian + PrecomputedHarmonicSampler
│   ├── path.py             # 보간 경로 + Kabsch 정렬/RMSD + 중심화
│   └── loss.py             # batchwise_l2
├── data/
│   └── pairs.py            # 정렬쌍 .sdf → 텐서 + collate
├── guidance.py             # 포켓 UFF 가이던스 (추론, Phase 7 예정)
├── sampler.py              # ODE 적분 (조건부/eval x0 주입)
├── train.py                # FlowConfig/EMA/flow_train_step/train_flow
├── evaluation.py           # (Phase 7 예정)
├── scripts/
│   ├── 01_build_aligned_pairs.py / 02_visualize_pairs.py
│   ├── 03_smoke_generate.py       # 게이트 A
│   ├── 04_overfit100.py           # 게이트 B + 진단(force-align/uncond/micro/load)
│   └── run_overfit.sh             # sbatch 러너
└── tests/
    ├── test_equivariance.py       # Phase3+4 등변성
    ├── test_smoke_run.py / test_data_pairs.py / test_conditional_flow.py
```

**격리 원칙**: `backbone/`은 태스크/조건을 모르는 순수 등변 네트워크 →
단독 검증 가능. 조건화는 `embedding.py`(is_ref)와 `network.py` forward 에 국한.
