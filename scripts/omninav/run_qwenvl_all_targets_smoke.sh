# 用途：运行 QwenVL-only 版本，QwenVL 同时看全部目标、模糊时间压力并选择导航 frontier。
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

GPU_ID="${GPU_ID:-7}"
LIMIT="${LIMIT:-5}"
BUDGET_RATIOS="${BUDGET_RATIOS:-0.5,1.0}"
MAX_DECISIONS="${MAX_DECISIONS:-5}"
MAX_GOTO_STEPS="${MAX_GOTO_STEPS:-60}"
OUTPUT_DIR="${OUTPUT_DIR:-output/time_aware_scene/omninav_qwenvl_all_targets_limit${LIMIT}}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/file_system/vepfs/algorithm/intern03/.micromamba}"
MICROMAMBA="${MICROMAMBA:-/file_system/vepfs/algorithm/intern03/.local/bin/micromamba}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet EGL_PLATFORM=surfaceless \
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX}" \
"${MICROMAMBA}" run -n omninav-infer \
  python tools/omninav/run_time_aware_omninav_slowfast.py \
  --planner omninav \
  --target-scheduler qwenvl-all-targets \
  --limit "${LIMIT}" \
  --budget-ratios "${BUDGET_RATIOS}" \
  --max-decisions "${MAX_DECISIONS}" \
  --max-goto-steps "${MAX_GOTO_STEPS}" \
  --output-dir "${OUTPUT_DIR}" \
  --save-planner-outputs
