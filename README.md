# ETFlowAlign

**DiffAlign의 확산(diffusion) 엔진을 ET-Flow의 flow matching으로 교체한 분자 정렬 생성 모델.**

DiffAlign의 **과제**(reference 리간드 형상에 정렬된 query 3D 구조를 노이즈에서 생성)를, ET-Flow의 **엔진**(flow matching + ODE 적분 + harmonic prior + 등변 백본) 위에서 다시 구현한 연구 저장소입니다.

---

## 개요

- **입력**: query 분자 그래프(원자종·결합) + reference 리간드의 3D 형상
- **출력**: reference에 정렬된 query의 3D 좌표
- **생성 방식**: harmonic prior에서 뽑은 노이즈 `x0` → 등변 백본이 예측한 속도장 `v_θ`를 ODE로 적분 → 정렬된 구조 `x1`
- **조건화**: reference 리간드만 네트워크에 주입(포켓은 추론 시 UFF 가이던스로만). 상대 기하로만 넣어 SE(3) 등변성 유지.

한 줄 요약: `harmonic prior → (reference 조건) 등변 속도장 → ODE → 정렬된 x1`

전체 설계 명세는 [`etflowalign/SPEC.md`](etflowalign/SPEC.md), 단계별 작업 분해와 진행 현황은 [`etflowalign/tree.txt`](etflowalign/tree.txt)에 있습니다.

---

## 저장소 구조

```
ETFlowAlign/
├── etflowalign/            # 새로 구현한 핵심 코드 (백지 재구축)
│   ├── SPEC.md             #   설계 캐논(태스크·조건·백본·게이트·비목표)
│   ├── tree.txt            #   Phase별 작업 분해 + 진행 추적
│   ├── SERVER_RUN.md       #   서버(SLURM) 실행 가이드
│   ├── backbone/           #   등변 벡터-피처 트랜스포머(자작)
│   ├── flow/               #   harmonic prior · 보간 경로 · 손실
│   ├── data/               #   정렬쌍 SDF → 텐서 변환
│   ├── scripts/            #   데이터 빌드·진단·학습 러너
│   └── tests/              #   등변성·데이터·조건화 단위 검증
├── external/
│   ├── diffalign/          # upstream DiffAlign (참조)
│   └── etflow/             # upstream ET-Flow (참조)
└── third_party_licenses/   # 원본 라이선스 원문
```

---

## `etflowalign` 패키지 구성

| 모듈 | 내용 |
|---|---|
| `backbone/` | self-contained 등변 벡터-피처 트랜스포머(TorchMD-Net/PaiNN 계열). 스칼라 `h`(불변) + 벡터 `vec`(등변) 동시 갱신. 등변성 오차 ~1e-15 검증 |
| `flow/` | harmonic prior(결합 라플라시안 기반), linear flow-matching 경로, batchwise-l2 손실, Kabsch source→target 정렬 |
| `data/` | GEOM-Drugs 유사분자 정렬쌍 SDF → per-sample 텐서. reference/query 분리 |
| `sampler.py` | ODE(Euler) 적분 생성. 포켓 UFF 가이던스 훅(예정) |
| `train.py` | flow-matching 학습(EMA, NaN 방어, source 정렬, 다중 샘플) |
| `tests/` | 등변성, 스모크 실행, 데이터 왕복, 조건부 학습/샘플 |

**설계 원칙**: `backbone/`은 태스크·조건을 모르는 순수 등변 네트워크로 격리 → 단독 검증 가능. 등변성은 코드가 아니라 **단위테스트로 강제**.

---

## 진행 현황

| Phase | 내용 | 상태 |
|---|---|---|
| 0 설계 | SPEC 고정 | ✅ 완료 |
| 3 백본 | 등변 벡터-피처 트랜스포머 | ✅ 등변성 1e-15 |
| 4 조건화 | reference 형상 주입 | ✅ joint 등변 검증 |
| 1 데이터 | GEOM-Drugs 정렬쌍 텐서화 | ✅ 65k 빌드 |
| 2 flow 엔진 | prior+FM+ODE | ✅ 합성 소분자 생성 통과 |
| 5 overfit | 실분자 암기(용량 게이트) | 🔬 진행 중 — 실분자 생성 정밀도 조사 |
| 6 일반화 | full 65k + held-out | ⬜ 예정 |
| 7 평가+포켓 | DISCO RMSD + UFF 가이던스 | ⬜ 예정 |

> 현재 초점: 합성 소분자 생성은 되지만 **실제 약물 분자(링·복잡한 기하)의 생성 정밀도**가 관건. 측정·적분·개수·용량·조건화·cutoff 요인을 진단으로 배제하며 원인을 좁히는 중.

---

## 실행

### 로컬 검증 (수초, 코드 무결성 확인)
```bash
PYTHONPATH=. python -m etflowalign.tests.test_equivariance      # 백본 등변성
PYTHONPATH=. python -m etflowalign.tests.test_smoke_run         # flow 엔진 실행
PYTHONPATH=. python -m etflowalign.tests.test_data_pairs        # 데이터 왕복
PYTHONPATH=. python -m etflowalign.tests.test_conditional_flow  # 조건부 학습/샘플
```

### 서버 학습/진단 (SLURM)
자세한 내용은 [`etflowalign/SERVER_RUN.md`](etflowalign/SERVER_RUN.md) 참조.
```bash
# 예: overfit 진단 (SLURM 배치 제출)
N=20 sbatch etflowalign/scripts/run_overfit.sh
```

---

## 서드파티 라이선스

이 저장소는 아래 오픈소스 코드를 참조·활용합니다 (원문은 `third_party_licenses/`):

- **DiffAlign** (MIT License) — 정렬 태스크·데이터 구성 참조
- **ET-Flow** (MIT License) — flow matching 엔진·harmonic prior·등변 백본 참조
