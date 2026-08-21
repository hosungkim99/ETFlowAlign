#!/bin/bash
#SBATCH -J t2383_pipe
#SBATCH -p normal
#SBATCH -c 4
#SBATCH -t 12:00:00
#SBATCH -o /gpfs/deepfold/casp/casp17-ligand/users/hosungkim/targets/T2383/casp17_t2383/pipe.%j.out
#SBATCH -e /gpfs/deepfold/casp/casp17-ligand/users/hosungkim/targets/T2383/casp17_t2383/pipe.%j.err
# ※ 이 클러스터는 RealMemory=1MB 로 잡혀 있어 --mem 을 주면 거부됨 → 절대 넣지 말 것.
# GPU로 gnina 가속하려면 아래 한 줄 켜기(공유 5노드라 시간 짧게 권장):
# #SBATCH --gres=gpu:1
# #SBATCH --exclude=ada-002

source /gpfs/deepfold/casp/casp17-ligand/scripts/env_setup.sh
set -e

SC=/gpfs/deepfold/casp/casp17-ligand/users/hosungkim/targets/T2383/casp17_t2383/pipeline
CONF=$SC/config/T2383.conf

echo "[sbatch] node=$(hostname) start=$(date)"
python3 $SC/run_pipeline.py $CONF
echo "[sbatch] done=$(date)"

# 재실행(이어하기): idempotent → 끝난 스텝 skip. 강제 재실행:
#   python3 $SC/run_pipeline.py $CONF --from pocket_validate --force
