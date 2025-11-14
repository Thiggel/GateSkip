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
OVERLAY_DIR="${WORK}/${REPO_NAME}/overlays"
OVERLAY_PATH="${OVERLAY_DIR}/pytorch.ext3"
OVERLAY_SIZE_GB=${OVERLAY_SIZE_GB:-32}
APPTAINER_BUILD_OPTS=${APPTAINER_BUILD_OPTS:-}

mkdir -p "${IMAGE_DIR}"
if [[ ! -f "${IMAGE_PATH}" ]]; then
  echo "Building Apptainer image at ${IMAGE_PATH}"
  "${APPTAINER_BIN}" build ${APPTAINER_BUILD_OPTS} "${IMAGE_PATH}" "${REPO_DIR}/pytorch.def"
fi

mkdir -p "${OVERLAY_DIR}"
if [[ ! -f "${OVERLAY_PATH}" ]]; then
  echo "Creating ${OVERLAY_SIZE_GB}G overlay at ${OVERLAY_PATH}"
  "${APPTAINER_BIN}" overlay create --size $((OVERLAY_SIZE_GB * 1024)) "${OVERLAY_PATH}"
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

"${APPTAINER_BIN}" exec --nv --cleanenv \
  --overlay "${OVERLAY_PATH}" \
  --bind "${SCRATCH}:${SCRATCH}","${WORK}:${WORK}","${REPO_DIR}:${REPO_DIR}" \
  --pwd "${REPO_DIR}" \
  --env APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR}" \
  --env PIP_CACHE_DIR="${PIP_CACHE_DIR}" \
  --env HF_HOME="${HF_HOME}" \
  --env HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
  --env HF_HUB_CACHE="${HF_HUB_CACHE}" \
  --env TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" \
  --env WANDB_DIR="${WANDB_DIR}" \
  "${IMAGE_PATH}" bash -lc "python3.10 -m pip install --upgrade pip && python3.10 -m pip install -r requirements.txt"

cat <<MSG
Setup complete.
Image: ${IMAGE_PATH}
Overlay: ${OVERLAY_PATH}
Caches: ${HF_HOME}
MSG
