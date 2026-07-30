# 用途：运行两层版本，外部 LLM 选择 active target，QwenVL/OmniNav 只负责视觉导航。
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

GPU_ID="${GPU_ID:-7}"
LIMIT="${LIMIT:-5}"
BUDGET_RATIOS="${BUDGET_RATIOS:-0.5,1.0}"
MAX_DECISIONS="${MAX_DECISIONS:-5}"
MAX_GOTO_STEPS="${MAX_GOTO_STEPS:-60}"
OUTPUT_DIR="${OUTPUT_DIR:-output/time_aware_scene/omninav_llm_scheduler_qwen_nav_limit${LIMIT}}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/file_system/vepfs/algorithm/intern03/.micromamba}"
MICROMAMBA="${MICROMAMBA:-/file_system/vepfs/algorithm/intern03/.local/bin/micromamba}"

# Replace TARGET_LLM_COMMAND with any LLM command that reads a prompt from stdin
# and prints exactly one remaining target index. The default is the DeepSeek
# wrapper. For plumbing tests, set:
#   TARGET_LLM_COMMAND="python tools/omninav/deepseek_time_aware_target_scheduler.py --mock-first-index"
TARGET_LLM_COMMAND="${TARGET_LLM_COMMAND:-python tools/omninav/deepseek_time_aware_target_scheduler.py}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet EGL_PLATFORM=surfaceless \
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX}" \
"${MICROMAMBA}" run -n omninav-infer \
  python tools/omninav/run_time_aware_omninav_slowfast.py \
  --planner omninav \
  --target-scheduler time-aware-llm-target \
  --target-llm-command "${TARGET_LLM_COMMAND}" \
  --limit "${LIMIT}" \
  --budget-ratios "${BUDGET_RATIOS}" \
  --max-decisions "${MAX_DECISIONS}" \
  --max-goto-steps "${MAX_GOTO_STEPS}" \
  --output-dir "${OUTPUT_DIR}" \
  --save-planner-outputs \
  --save-scheduler-prompts
