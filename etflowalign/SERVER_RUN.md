# ETFlowAlign 서버 실행 가이드 (A10 x4, CUDA)

> 상단 **변수 3개만** 본인 환경에 맞게 채우면 아래 명령을 그대로 복붙 실행 가능.
> 서버 파일은 본인 사용자 디렉터리 범위 안에서만 생성/수정할 것.

```bash
# ── 채울 변수 ─────────────────────────────────────────────
export CODE=/path/to/ETFlowAlign          # etflowalign/ 패키지의 부모 디렉터리
export DATA=/path/to/GEOM-Drugs_AlignedPairs   # complex_*/{query,reference}.sdf 루트
export GPU=1                                # 빈 GPU 번호 (nvidia-smi 로 확인, 0은 자주 점유)
# ────────────────────────────────────────────────────────
cd $CODE
```

모든 명령은 `$CODE`(패키지 부모)에서 실행하며 `PYTHONPATH=.` 를 붙인다.
스크립트 파일명이 숫자로 시작해 `-m` import 는 안 되므로 **파일 경로로 실행**한다.

---

## 0) 환경 설치 (최초 1회)

```bash
# 서버 CUDA 에 맞는 torch 를 먼저 설치(예시 — 실제 CUDA 버전에 맞출 것)
# pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r etflowalign/requirements.txt
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 1) 설치 검증 — 로컬 4종 테스트를 서버에서 (수초, CPU 무방)

```bash
PYTHONPATH=. python -m etflowalign.tests.test_equivariance
PYTHONPATH=. python -m etflowalign.tests.test_smoke_run
PYTHONPATH=. python -m etflowalign.tests.test_data_pairs
PYTHONPATH=. python -m etflowalign.tests.test_conditional_flow
```
→ 모두 PASS 면 코드/의존성 정상.

## 2) 게이트 A — GPU에서 엔진 학습+생성 (실데이터 불필요, 첫 GPU 실전)

합성 분자로 harmonic prior + 8000스텝 학습 후 생성. **A10에서 엔진이 도는지 + 게이트 A 판정.**

```bash
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. PYTHONIOENCODING=utf-8 \
  python etflowalign/scripts/03_smoke_generate.py \
    --prior harmonic --mols 4 --steps 8000 --device cuda
```
→ 기대: `generated aligned RMSD` 가 baseline 보다 크게 낮고 결합거리 1.3~1.7A → `[GATE A] PASS`.
   (안 넘으면 steps 를 늘리거나 로컬에서 진단 — 서버 시간 낭비 전 알려주세요.)

## 3) Phase 1 — 실데이터 정렬쌍을 텐서(.pt)로 빌드

```bash
# 먼저 소수만 (--limit 50) 로 형식 확인
PYTHONPATH=. python etflowalign/scripts/01_build_aligned_pairs.py \
  --root $DATA --out $CODE/geom_pt --limit 50

# 정렬 육안/통계 게이트 (SPEC §3.2)
PYTHONPATH=. python etflowalign/scripts/02_visualize_pairs.py \
  --root $DATA --out $CODE/vis --n 20
# -> shape_overlap>0.5, centroid_dist 작음 이면 정렬 양호. vis/*.sdf 를 뷰어로 확인.

# 문제 없으면 전체 빌드 (--limit 0)
PYTHONPATH=. python etflowalign/scripts/01_build_aligned_pairs.py \
  --root $DATA --out $CODE/geom_pt --limit 0
```

## 4) Phase 5 — overfit-100 (다음 단계, 러너 scripts/04 작성 예정)

> `scripts/04_overfit100.py` 는 아직 미작성. harmonic 고유분해 사전계산 최적화와 함께
> 곧 추가 예정. 추가되면 여기 명령을 채운다.

---

## 성능 메모
- 대규모 학습 전: `torch.set_float32_matmul_precision("high")` 권장 (A10 tf32).
- Phase 6(full 65k) 전 벡터화 필수 부채: HarmonicSampler eigh 루프, center_pos 루프
  (tree.txt "성능 부채" 참조).
