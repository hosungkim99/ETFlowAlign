#!/usr/bin/env bash
set -euo pipefail

# ETFlowAlign environment setup script.
# References:
# - external/diffalign/diffalign/env.yml
# - external/etflow/env.yml

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/env.yml"

ENV_NAME="etflowalign"
MODE="cuda"       # cpu | cuda
CUDA_VERSION="12.1"
INSTALL_UFF="true" # true | false
UFF_REF="main"     # override with commit hash/tag for reproducibility
RUN_SMOKE="false"

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Options:
  --name <env_name>         Conda environment name (default: etflowalign)
  --mode <cpu|cuda>         Torch installation mode (default: cuda)
  --cuda-version <version>  CUDA version for pytorch-cuda (default: 12.1)
  --no-uff                  Skip UFF_PyTorch installation
  --uff-ref <ref>           UFF_PyTorch git ref (commit/tag/branch; default: main)
  --smoke-test              Run a post-install import smoke test
  -h, --help                Show this message
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      ENV_NAME="$2"; shift 2 ;;
    --mode)
      MODE="$2"; shift 2 ;;
    --cuda-version)
      CUDA_VERSION="$2"; shift 2 ;;
    --no-uff)
      INSTALL_UFF="false"; shift ;;
    --uff-ref)
      UFF_REF="$2"; shift 2 ;;
    --smoke-test)
      RUN_SMOKE="true"; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1 ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  echo "[error] conda command not found. Install Miniconda/Anaconda first." >&2
  exit 1
fi

if [[ "$MODE" != "cpu" && "$MODE" != "cuda" ]]; then
  echo "[error] --mode must be 'cpu' or 'cuda'." >&2
  exit 1
fi

echo "[setup] creating conda env '${ENV_NAME}' from ${ENV_FILE}"
conda env create -f "${ENV_FILE}" -n "${ENV_NAME}" || {
  echo "[setup] env creation failed (already exists?). trying update..."
  conda env update -f "${ENV_FILE}" -n "${ENV_NAME}" --prune
}

if [[ "$MODE" == "cuda" ]]; then
  echo "[setup] installing PyTorch + CUDA (${CUDA_VERSION})"
  conda install -n "${ENV_NAME}" -y pytorch pytorch-cuda="${CUDA_VERSION}" -c pytorch -c nvidia
else
  echo "[setup] installing CPU PyTorch"
  conda install -n "${ENV_NAME}" -y pytorch cpuonly -c pytorch
fi

echo "[setup] installing PyG packages"
if ! conda install -n "${ENV_NAME}" -y pyg pytorch-cluster -c pyg; then
  echo "[setup] conda PyG install failed; trying pip wheels fallback"
  TORCH_VER="$(conda run -n "${ENV_NAME}" python -c 'import torch; print(torch.__version__.split(\"+\")[0])')"
  if [[ "$MODE" == "cuda" ]]; then
    WHEEL_INDEX="https://data.pyg.org/whl/torch-${TORCH_VER}+cu121.html"
  else
    WHEEL_INDEX="https://data.pyg.org/whl/torch-${TORCH_VER}+cpu.html"
  fi
  conda run -n "${ENV_NAME}" pip install --find-links="${WHEEL_INDEX}" torch-geometric pyg_lib torch_scatter torch_sparse torch_cluster
fi

if [[ "$INSTALL_UFF" == "true" ]]; then
  echo "[setup] installing UFF_PyTorch (DiffAlign guidance dependency) @ ${UFF_REF}"
  conda run -n "${ENV_NAME}" pip install "git+https://github.com/kim-iljung/UFF_PyTorch.git@${UFF_REF}"
fi

echo "[setup] done"
echo "[setup] activate with: conda activate ${ENV_NAME}"
echo "[setup] smoke test: python -m etflowalign.train --help"

if [[ "$RUN_SMOKE" == "true" ]]; then
  echo "[setup] running smoke test imports"
  conda run -n "${ENV_NAME}" python - <<'PY'
import torch
import etflowalign
print("torch:", torch.__version__)
print("etflowalign exports:", etflowalign.__all__)
PY
fi
