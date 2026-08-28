# ETFlowAlign 서버 실행 가이드 (GPU 클러스터, SLURM)

> **원칙: 모든 compute 는 SLURM(sbatch)으로만.** 로그인 노드 직접 실행·srun 잡 금지(감사 대상).
> 코드는 로컬에서 작성 → 서버로 **파일 통째 복사** 동기화(부분 편집 금지 — 유령버그 이력).
> 서버 파일은 본인 사용자 디렉터리 범위 안에서만 생성/수정.

```bash
# ── 채울 변수 ─────────────────────────────────────────────
export CODE=/path/to/ETFlowAlign   # etflowalign/ 패키지의 부모 디렉터리
cd $CODE
```

---

## 0) 환경 (최초 1회)

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate etflowalign
pip install -r etflowalign/requirements.txt        # torch 는 CUDA 맞는 빌드 먼저
python -c "import torch; print('cuda', torch.cuda.is_available())"
```

## 1) 설치 검증 (수초, 로그인노드 CPU 무방 — 학습 아님)

```bash
PYTHONPATH=. python -m etflowalign.tests.test_equivariance
PYTHONPATH=. python -m etflowalign.tests.test_smoke_run
PYTHONPATH=. python -m etflowalign.tests.test_data_pairs
PYTHONPATH=. python -m etflowalign.tests.test_conditional_flow
```
→ 모두 PASS 면 코드/의존성 정상.

## 2) 데이터 (Phase 1 — 이미 65k 빌드 완료, 재빌드 시에만)

```bash
export DATA=/path/to/GEOM-Drugs_AlignedPairs       # complex_*/{query,reference}.sdf
PYTHONPATH=. python etflowalign/scripts/01_build_aligned_pairs.py --root $DATA --out $CODE/geom_pt --limit 0
PYTHONPATH=. python etflowalign/scripts/02_visualize_pairs.py     --root $DATA --out $CODE/vis --n 20
```
→ `$CODE/geom_pt/*.pt` (per-sample 텐서). shape_overlap>0.5 면 정렬 양호.

---

## 3) 학습/진단 = sbatch 러너 (scripts/run_overfit.sh)

**환경변수로 파라미터 지정, sbatch로 제출.** 로그: `etfa_of_<jobid>.log` (제출 디렉터리).

```bash
# overfit 기본 (100개, 조건부)
sbatch etflowalign/scripts/run_overfit.sh

# 진단 예시 — 분자 1개, 분산↓(B1), 비조건
N=1 UNCOND=--unconditional MICRO=8 sbatch --time=00:40:00 etflowalign/scripts/run_overfit.sh

# eval-only (재학습 없이 저장 체크포인트로 샘플만; 샘플스텝·지표 재측정)
N=20 UNCOND=--unconditional LOAD=$CODE/overfit_20_uc_ckpt.pt SAMPLE=500 sbatch etflowalign/scripts/run_overfit.sh
```

**러너 환경변수 (run_overfit.sh):**

| 변수 | 기본 | 뜻 |
|---|---|---|
| `N` | 100 | 분자 수 |
| `STEPS` | 30000 | 학습 스텝 |
| `BATCH` | 8 | 미니배치 분자 수 (OOM 시 낮춤) |
| `HIDDEN`/`LAYERS` | 256/8 | 백본 크기 |
| `SAMPLE` | 100 | ODE 샘플 스텝 |
| `MICRO` | 1 | 스텝당 (x0,t) 샘플 수 (K>1=분산↓, B1) |
| `FORCE` | (없음) | `--force-align` 진단(오라클 회전제거) |
| `UNCOND` | (없음) | `--unconditional` 진단(reference 제거) |
| `LOAD` | (없음) | 체크포인트 경로 → eval-only |

**결과 읽기 (GATE B 블록):** `median heavy-atom RMSD`(표준 지표), `<1A/<2A 비율`,
`prior 기준선`, 원자조성(H비율), 크기별 진단.

## 4) 확인 / 큐 관리

```bash
squeue --me                              # 내 job 상태 (PD 대기 / R 실행)
squeue --me --start                      # 예상 시작 시각
tail -40 $(ls -t etfa_of_*.log | head -1)   # 최신 로그 끝
sinfo                                     # 파티션별 여유
```
- **큐가 붐벼 안 잡히면 짧은 `--time`으로 backfill 유도**(예 `--time=00:10:00`) — 짧은 job이
  큰 job들 사이 빈틈에 먼저 들어감. 필요 자원만 짧게 요청 = 빨리 실행.
- 덜 붐비는 파티션: `sbatch --partition=<partition> ...` (다른 GPU 파티션도 코드 그대로 동작).

---

## 성능 메모 (Phase 6 full 65k 전 벡터화 필요)
- HarmonicSampler eigh/파이썬 루프, center_pos 루프 → scatter 기반 벡터화.
  (overfit 소규모는 PrecomputedHarmonicSampler 캐시로 완화됨)
- build_radius_graph O(N^2) — drug-sized/미니배치8 OK, 대형 배치서 메모리↑.
- 대규모 학습 시 `torch.set_float32_matmul_precision("high")` (tf32 지원 GPU).
