#!/usr/bin/env bash
set -euo pipefail

cd /file_system/vepfs/algorithm/intern03/mhw/LH-VLN

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "Please export DEEPSEEK_API_KEY before running this script." >&2
  exit 1
fi

EGL_PLATFORM=surfaceless \
__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLdispatch.so.0 \
HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet HABITAT_GPU_DEVICE_ID="${HABITAT_GPU_DEVICE_ID:-0}" \
/file_system/vepfs/algorithm/intern03/.conda/envs/lhvln/bin/python \
  tools/run_time_aware_navgpt.py \
  --episodes /file_system/nas/algorithm/Intern03/data/time_aware_scene/all_episodes_spatial_start_floor_oracle_time.jsonl \
  --split all \
  --limit "${LIMIT:-2}" \
  --budget-ratios "${BUDGET_RATIOS:-0.5}" \
  --max-decisions "${MAX_DECISIONS:-40}" \
  --max-action-repeat "${MAX_ACTION_REPEAT:-4}" \
  --observation-mode semantic_text \
  --render \
  --time-prompt-mode dynamic_fuzzy \
  --navgpt-command "/file_system/vepfs/algorithm/intern03/.conda/envs/lhvln/bin/python tools/deepseek_navgpt_action_selector.py --model ${DEEPSEEK_MODEL:-deepseek-chat}" \
  --output-dir "${OUTPUT_DIR:-output/time_aware_scene/navgpt_visual_action_smoke}" \
  --summary-csv "${SUMMARY_CSV:-output/time_aware_scene/navgpt_visual_action_smoke/summary.csv}" \
  --save-prompts \
  --quiet
