# 用途：在 OmniNav 推理环境中按官方要求源码安装 Habitat-Sim 0.2.3 和 Habitat-Lab waypoint 分支。
#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${OMNINAV_ENV_NAME:-omninav-infer}"
MICROMAMBA_BIN="${MICROMAMBA_BIN:-/file_system/vepfs/algorithm/intern03/.local/bin/micromamba}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/file_system/vepfs/algorithm/intern03/.micromamba}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
THIRD_PARTY_DIR="${OMNINAV_THIRD_PARTY_DIR:-${REPO_ROOT}/third_party}"
HAB_SIM_DIR="${HABITAT_SIM_SRC:-${THIRD_PARTY_DIR}/habitat-sim-v0.2.3}"
HAB_LAB_DIR="${HABITAT_LAB_SRC:-${THIRD_PARTY_DIR}/habitat-lab-v0.2.3_waypoint}"

PYTHON=("${MICROMAMBA_BIN}" run -n "${ENV_NAME}" python)

if [[ ! -x "${MICROMAMBA_BIN}" ]]; then
  echo "Missing micromamba: ${MICROMAMBA_BIN}" >&2
  exit 1
fi

mkdir -p "${THIRD_PARTY_DIR}"

echo "[Habitat] env: ${ENV_NAME}"
echo "[Habitat] third_party: ${THIRD_PARTY_DIR}"

"${PYTHON[@]}" -m pip install ninja
"${PYTHON[@]}" -m pip uninstall -y cmake || true

if [[ "${SKIP_HABITAT_SIM:-0}" != "1" ]]; then
  if [[ ! -d "${HAB_SIM_DIR}/.git" ]]; then
    git clone --recursive https://github.com/facebookresearch/habitat-sim.git "${HAB_SIM_DIR}"
  fi
  git -C "${HAB_SIM_DIR}" fetch --tags
  git -C "${HAB_SIM_DIR}" checkout v0.2.3
  git -C "${HAB_SIM_DIR}" submodule update --init --recursive

  echo "[Habitat-Sim] installing requirements"
  "${PYTHON[@]}" -m pip install -r "${HAB_SIM_DIR}/requirements.txt"

  echo "[Habitat-Sim] building headless v0.2.3"
  (
    cd "${HAB_SIM_DIR}"
    if [[ "${HABITAT_CLEAN_BUILD:-0}" == "1" ]]; then
      rm -rf build
    fi
    export CMAKE_ARGS="${CMAKE_ARGS:-} -DCMAKE_POLICY_VERSION_MINIMUM=3.5"
    "${PYTHON[@]}" setup.py install --headless
  )
else
  echo "[Habitat-Sim] skip build because SKIP_HABITAT_SIM=1"
fi

if [[ ! -d "${HAB_LAB_DIR}/.git" ]]; then
  git clone https://github.com/chongchong2025/habitat-lab "${HAB_LAB_DIR}"
fi
git -C "${HAB_LAB_DIR}" fetch
git -C "${HAB_LAB_DIR}" checkout v0.2.3_waypoint

echo "[Habitat-Lab] installing waypoint fork requirements"
"${PYTHON[@]}" -m pip install -r "${HAB_LAB_DIR}/habitat-baselines/habitat_baselines/rl/requirements.txt"
"${PYTHON[@]}" -m pip install -r "${HAB_LAB_DIR}/habitat-baselines/habitat_baselines/rl/ddppo/requirements.txt"
"${PYTHON[@]}" -m pip install -e "${HAB_LAB_DIR}/habitat-lab"
"${PYTHON[@]}" -m pip install --force-reinstall \
  "numpy==1.26.4" \
  "opencv-python==4.11.0.86" \
  "opencv-python-headless==4.11.0.86"
"${PYTHON[@]}" -m pip install -e "${HAB_LAB_DIR}/habitat-baselines"
"${PYTHON[@]}" -m pip install --force-reinstall \
  "numpy==1.26.4" \
  "opencv-python==4.11.0.86" \
  "opencv-python-headless==4.11.0.86"

echo "[Habitat] install finished"
"${PYTHON[@]}" - <<'PY'
import habitat
import habitat_sim
print("habitat", getattr(habitat, "__version__", "unknown"))
print("habitat_sim", getattr(habitat_sim, "__version__", "unknown"))
PY
