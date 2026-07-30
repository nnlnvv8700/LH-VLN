# 用途：创建 OmniNav slow-fast 推理用轻量环境，训练依赖和 Habitat 源码编译分开处理。
#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${OMNINAV_ENV_NAME:-omninav-infer}"
MICROMAMBA_BIN="${MICROMAMBA_BIN:-/file_system/vepfs/algorithm/intern03/.local/bin/micromamba}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/file_system/vepfs/algorithm/intern03/.micromamba}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OMNINAV_ROOT="${REPO_ROOT}/third_party/OmniNav"
TRANSFORMERS_MAIN="${OMNINAV_ROOT}/train_code/transformers-main"

if [[ ! -x "${MICROMAMBA_BIN}" ]]; then
  echo "Missing micromamba: ${MICROMAMBA_BIN}" >&2
  echo "Run: bash scripts/omninav/bootstrap_micromamba.sh" >&2
  exit 1
fi
if [[ ! -d "${OMNINAV_ROOT}" ]]; then
  echo "Missing OmniNav repo: ${OMNINAV_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${TRANSFORMERS_MAIN}" ]]; then
  echo "Missing OmniNav transformers-main: ${TRANSFORMERS_MAIN}" >&2
  exit 1
fi

mkdir -p "${MAMBA_ROOT_PREFIX}"

echo "[OmniNav infer] env: ${ENV_NAME}"
echo "[OmniNav infer] MAMBA_ROOT_PREFIX: ${MAMBA_ROOT_PREFIX}"
if [[ -x "${MAMBA_ROOT_PREFIX}/envs/${ENV_NAME}/bin/python" ]]; then
  echo "[OmniNav infer] env already exists, skip create"
else
  "${MICROMAMBA_BIN}" create -y -n "${ENV_NAME}" -c conda-forge python=3.10 pip
fi

PYTHON=("${MICROMAMBA_BIN}" run -n "${ENV_NAME}" python)

"${PYTHON[@]}" -m pip install --upgrade pip setuptools wheel
"${PYTHON[@]}" -m pip install \
  --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
"${PYTHON[@]}" -m pip install -e "${TRANSFORMERS_MAIN}"
"${PYTHON[@]}" -m pip install \
  qwen-vl-utils==0.0.10 \
  accelerate \
  safetensors \
  pillow \
  numpy==1.26.4 \
  opencv-python-headless \
  scipy \
  numba \
  omegaconf \
  tqdm \
  numpy-quaternion \
  jsonlines \
  modelscope

cat <<EOF

Inference Python dependencies installed.

Still required before full OmniNav inference:
1. Install Habitat-Sim v0.2.3 into env ${ENV_NAME}.
2. Install Habitat-Lab v0.2.3_waypoint into env ${ENV_NAME}.
3. Re-run:
   ${MICROMAMBA_BIN} run -n ${ENV_NAME} python tools/omninav/check_omninav_setup.py
EOF
