#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SCRATCH:-}" || -z "${WORK:-}" ]]; then
  echo "SCRATCH and WORK must be set before sourcing jobs/environment.sh" >&2
  exit 1
fi

if command -v module >/dev/null 2>&1; then
  module load cuda/12.1.1 || true
fi

unset SLURM_EXPORT_ENV || true

if [[ -f .env ]]; then
  set -o allexport
  # shellcheck disable=SC1091
  source .env
  set +o allexport
fi

REPO_DIR=$(pwd)
REPO_NAME=${REPO_NAME:-GateSkip}
APPTAINER_BIN=${APPTAINER:-apptainer}
IMAGE_PATH="${WORK}/${REPO_NAME}/images/pytorch.sif"

export APPTAINER_CACHEDIR="${SCRATCH}/apptainer-cache"
export PIP_CACHE_DIR="${SCRATCH}/pip-cache"
export HF_HOME="${SCRATCH}/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export WANDB_DIR="${SCRATCH}/wandb"
export CHECKPOINT_ROOT="${WORK}/${REPO_NAME}/checkpoints"
export PYTHONNOUSERSITE=1
export PYTHONPATH=

mkdir -p \
  "${APPTAINER_CACHEDIR}" \
  "${PIP_CACHE_DIR}" \
  "${HF_DATASETS_CACHE}" \
  "${HF_HUB_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${WANDB_DIR}" \
  "${CHECKPOINT_ROOT}"

if [[ ! -f "${IMAGE_PATH}" ]]; then
  echo "Missing apptainer image at ${IMAGE_PATH}. Run setup.sh before launching jobs." >&2
  exit 1
fi

APPTAINER_RUN=(
  "${APPTAINER_BIN}" exec --nv --cleanenv
  --bind "${SCRATCH}:${SCRATCH}","${WORK}:${WORK}","${REPO_DIR}:${REPO_DIR}"
  --pwd "${REPO_DIR}"
  --env APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR}"
  --env PIP_CACHE_DIR="${PIP_CACHE_DIR}"
  --env HF_HOME="${HF_HOME}"
  --env HF_DATASETS_CACHE="${HF_DATASETS_CACHE}"
  --env HF_HUB_CACHE="${HF_HUB_CACHE}"
  --env TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}"
  --env WANDB_DIR="${WANDB_DIR}"
  --env CHECKPOINT_ROOT="${CHECKPOINT_ROOT}"
  --env HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-}"
  --env WANDB_API_KEY="${WANDB_API_KEY:-}"
  --env CUDA_HOME="${CUDA_HOME:-}"
  --env PYTHONNOUSERSITE="${PYTHONNOUSERSITE}"
  --env PYTHONPATH="${PYTHONPATH}"
  "${IMAGE_PATH}"
)

PYTHON_BIN=${PYTHON_BIN:-python3.10}

run_in_apptainer() {
  "${APPTAINER_RUN[@]}" "$@"
}
