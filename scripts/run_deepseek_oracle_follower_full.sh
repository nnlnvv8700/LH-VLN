#!/usr/bin/env bash
# 用途：运行 Time-Aware VLN 历史实验。
set -euo pipefail

cd /file_system/vepfs/algorithm/intern03/mhw/LH-VLN

export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:?Please export DEEPSEEK_API_KEY before running.}"
export PYTHONUNBUFFERED=1

HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet EGL_PLATFORM=surfaceless \
/file_system/vepfs/algorithm/intern03/.conda/envs/lhvln/bin/python -u \
  tools/run_time_aware_llm_planner.py \
  --episodes /file_system/nas/algorithm/Intern03/data/time_aware_scene/all_episodes_spatial_start_floor_oracle_time.jsonl \
  --split all \
  --limit 0 \
  --budget-ratios 0.5,1.0,1.5 \
  --planner llm \
  --time-prompt fuzzy \
  --fuzzy-time auto \
  --planner-observation text_only \
  --llm-command "/file_system/vepfs/algorithm/intern03/.conda/envs/lhvln/bin/python tools/deepseek_target_selector.py --model deepseek-chat" \
  --invalid-llm-choice end_episode \
  --output-dir output/time_aware_scene/navgpt_oracle_follower_deepseek_full \
  --summary-csv output/time_aware_scene/navgpt_oracle_follower_deepseek_full/summary.csv \
  --quiet
