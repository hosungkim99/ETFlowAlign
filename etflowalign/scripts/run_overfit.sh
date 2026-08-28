#!/bin/bash
#SBATCH --job-name=etfa_of
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.log
#SBATCH --time=02:00:00
# 필요 시 파티션 지정: #SBATCH --partition=<partition>

# overfit-100 진단 러너 (SLURM 배치 제출용).
# 사용 예:
#   sbatch etflowalign/scripts/run_overfit.sbatch                 # 기본(100개)
#   N=4  FORCE=--force-align sbatch etflowalign/scripts/run_overfit.sbatch
#   N=20 STEPS=30000 FORCE=--force-align sbatch etflowalign/scripts/run_overfit.sbatch
# 로그: etfa_of_<jobid>.log  (제출 디렉터리에 생성)
# 확인: squeue --me   |   tail -f etfa_of_<jobid>.log

set -e

# ── 경로/환경 ─────────────────────────────────────────────
CODE=${CODE:?export CODE=<ETFlowAlign 패키지의 부모 디렉터리>}
cd "$CODE"

# batch 셸은 rc 를 안 읽으므로 conda 를 명시적으로 활성화
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate etflowalign

export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── 파라미터 (환경변수로 덮어쓰기 가능) ───────────────────
N=${N:-100}                 # 분자 수
STEPS=${STEPS:-30000}       # 학습 스텝
BATCH=${BATCH:-8}           # 미니배치 분자 수
HIDDEN=${HIDDEN:-256}       # 백본 hidden 채널
LAYERS=${LAYERS:-8}         # 백본 레이어 수
SAMPLE=${SAMPLE:-100}       # ODE 샘플 스텝
MICRO=${MICRO:-1}           # 스텝당 (x0,t) 샘플 수(K>1=분산↓, B1)
FORCE=${FORCE:-}            # "--force-align" 진단(조건부 회전 제거)
UNCOND=${UNCOND:-}          # "--unconditional" 진단(reference 제거)
LOAD=${LOAD:-}              # 체크포인트 경로 주면 eval-only(학습 생략, 샘플만)
SAVE=${SAVE:-$CODE/overfit_${N}${FORCE:+_fa}${UNCOND:+_uc}_ckpt.pt}

LOAD_ARG=""; [ -n "$LOAD" ] && LOAD_ARG="--load $LOAD"
SAVE_ARG="--save $SAVE"; [ -n "$LOAD" ] && SAVE_ARG=""   # eval-only 는 저장 안 함

echo "[sbatch] node=$(hostname) N=$N STEPS=$STEPS BATCH=$BATCH HIDDEN=$HIDDEN LAYERS=$LAYERS SAMPLE=$SAMPLE MICRO=$MICRO FORCE=${FORCE:-none} UNCOND=${UNCOND:-none} LOAD=${LOAD:-none}"
nvidia-smi -L

python etflowalign/scripts/04_overfit100.py \
  --pt "$CODE/geom_pt" --n "$N" --steps "$STEPS" --batch-size "$BATCH" \
  --hidden "$HIDDEN" --layers "$LAYERS" --sample-steps "$SAMPLE" --micro "$MICRO" \
  --device cuda $FORCE $UNCOND $LOAD_ARG $SAVE_ARG
