#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SCRATCH:-}" || -z "${WORK:-}" ]]; then
  echo "SCRATCH and WORK must be set in the environment before running setup.sh" >&2
  exit 1
fi

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_NAME="GateSkip"
APPTAINER_BIN=${APPTAINER:-apptainer}
IMAGE_DIR="${WORK}/${REPO_NAME}/images"
IMAGE_PATH="${IMAGE_DIR}/pytorch.sif"
APPTAINER_BUILD_OPTS=${APPTAINER_BUILD_OPTS:-}

mkdir -p "${IMAGE_DIR}"
if [[ ! -f "${IMAGE_PATH}" ]]; then
  echo "Building Apptainer image at ${IMAGE_PATH}"
  "${APPTAINER_BIN}" build ${APPTAINER_BUILD_OPTS} "${IMAGE_PATH}" "${REPO_DIR}/pytorch.def"
fi

export APPTAINER_CACHEDIR="${SCRATCH}/apptainer-cache"
export PIP_CACHE_DIR="${SCRATCH}/pip-cache"
export HF_HOME="${SCRATCH}/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export WANDB_DIR="${SCRATCH}/wandb"

mkdir -p \
  "${APPTAINER_CACHEDIR}" \
  "${PIP_CACHE_DIR}" \
  "${HF_DATASETS_CACHE}" \
  "${HF_HUB_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${WANDB_DIR}"

cat <<MSG
Setup complete.
Image: ${IMAGE_PATH}
Caches: ${HF_HOME}
MSG
