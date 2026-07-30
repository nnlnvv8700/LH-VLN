# 用途：NavGPT-style Time-Aware VLN 脚本。
#!/usr/bin/env python3
"""Run NaVid/Uni-NaVid inside the Time-Aware VLN multi-target evaluator.

Unlike the DeepSeek/NavGPT-style action selector, this runner lets a VLN/VLM
navigation model consume RGB observations and choose navigation actions itself.
The Habitat execution and unordered multi-target completion metrics stay in the
LH-VLN time-aware simulator.
"""

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from habitat_base.time_aware_simulation import TimeAwareSceneSimulator
from tools.run_time_aware_greedy import (
    load_records,
    parse_budget_ratios,
    record_id,
    record_instruction,
    record_to_config,
    summarize,
    write_json,
    write_summary_csv,
)


ACTION_ID_TO_NAME = {
    0: "stop",
    1: "move_forward",
    2: "turn_left",
    3: "turn_right",
}

FUZZY_BY_RATIO = {
    "0.5": "Time condition: time is very limited. You are in a hurry. Prioritize quick useful progress.",
    "1.0": "Time condition: time is moderate. Balance completing targets with avoiding wasted exploration.",
    "1.5": "Time condition: time is sufficient. Try to complete all targets carefully.",
}


def ratio_name(ratio):
    return str(ratio).replace(".", "p")


def make_sim(args, record, time_budget):
    sim_args = SimpleNamespace(
        scene=args.scene,
        scene_dataset=args.scene_dataset,
        success_dis=args.success_dis,
        max_step=time_budget,
        no_render=args.no_render,
    )
    return TimeAwareSceneSimulator(
        sim_args,
        record_to_config(record),
        time_budget=time_budget,
        target_values=[target.get("value", 1.0) for target in record["targets"]],
    )


def dynamic_time_text(args, sim):
    base = FUZZY_BY_RATIO.get(str(args.budget_ratio), "Time condition: time is limited. Navigate efficiently.")
    if args.time_prompt_mode == "ratio_fuzzy":
        return base
    if sim.time_budget <= 0:
        urgency = "Current urgency: time is exhausted."
    else:
        fraction = sim.time_remaining / sim.time_budget
        if fraction <= 0.25:
            urgency = "Current urgency: time is almost exhausted."
        elif fraction <= 0.5:
            urgency = "Current urgency: time is tight."
        elif fraction <= 0.8:
            urgency = "Current urgency: time is moderate."
        else:
            urgency = "Current urgency: time is relatively sufficient."
    return f"{base} {urgency}"


def target_text(record, sim):
    lines = []
    for index in sorted(sim.remaining_targets):
        target = record["targets"][index]
        name = target.get("name", f"target_{index}")
        region = target.get("region_name") or target.get("region") or "unknown area"
        lines.append(f"target {index}: {name} in {region}")
    if not lines:
        return "All targets have been completed."
    return "\n".join(lines)


def build_instruction(args, record, sim):
    completed = [
        record["targets"][index].get("name", f"target_{index}")
        for index in sim.completed_targets
    ]
    return (
        f"{dynamic_time_text(args, sim)}\n"
        "Complete as many unordered targets as possible. The targets can be completed in any order.\n"
        f"Completed targets: {', '.join(completed) if completed else 'none'}.\n"
        "Remaining targets:\n"
        f"{target_text(record, sim)}\n"
        "Original task context:\n"
        f"{record_instruction(record)}"
    )


def rgb_observation(sim):
    observations = sim.observations or {}
    rgb = observations.get("color_sensor_f")
    if rgb is None:
        raise RuntimeError("Missing color_sensor_f observation. Run with --render.")
    rgb = np.asarray(rgb)
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    return rgb.astype(np.uint8)


def load_agent(args):
    navid_root = Path(args.navid_root).resolve()
    if str(navid_root) not in sys.path:
        sys.path.insert(0, str(navid_root))

    if args.agent == "uninavid":
        from agent_uninavid import UniNaVid_Agent

        return UniNaVid_Agent(args.model_path, args.agent_result_path, args.exp_save)
    if args.agent == "navid":
        from agent_navid import NaVid_Agent

        return NaVid_Agent(args.model_path, args.agent_result_path, args.exp_save)
    raise ValueError(f"Unsupported agent: {args.agent}")


def action_from_agent_output(output):
    if isinstance(output, dict):
        action_id = output.get("action")
    else:
        action_id = output
    try:
        action_id = int(action_id)
    except Exception as exc:
        raise RuntimeError(f"Invalid agent action output: {output!r}") from exc
    if action_id not in ACTION_ID_TO_NAME:
        raise RuntimeError(f"Unknown action id from agent: {action_id}")
    return ACTION_ID_TO_NAME[action_id], action_id


def run_episode(args, agent, record):
    ratio_key = str(args.budget_ratio)
    time_budget = record["time_budgets"].get(ratio_key)
    if time_budget is None:
        raise ValueError(f"Missing budget ratio {ratio_key} for {record_id(record)}")

    sim = make_sim(args, record, time_budget)
    trace = []
    try:
        agent.reset()
        sim.actor("stop")
        while not sim.episode_over and sim.time_used < time_budget:
            obs = {
                "rgb": rgb_observation(sim),
                "instruction": {"text": build_instruction(args, record, sim)},
            }
            info = {}
            output = agent.act(obs, info, record_id(record))
            action, action_id = action_from_agent_output(output)
            sim.actor(action)
            trace.append(
                {
                    "time_used": sim.time_used,
                    "time_remaining": sim.time_remaining,
                    "action": action,
                    "action_id": action_id,
                    "completed_targets": list(sim.completed_targets),
                }
            )
            if len(trace) >= args.max_steps:
                break
        result = sim.return_results()
    finally:
        sim.close()

    result["task_id"] = record_id(record)
    result["split"] = record["split"]
    result["scene"] = record["scene"]
    result["robot"] = record["robot"]
    result["baseline"] = args.agent
    result["time_prompt_mode"] = args.time_prompt_mode
    result["trace"] = trace
    return result


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes",
        default="/file_system/nas/algorithm/Intern03/data/time_aware_scene/all_episodes_spatial_start_floor_oracle_time.jsonl",
    )
    parser.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--budget-ratio", type=float, default=None)
    parser.add_argument("--budget-ratios", default="0.5,1.0,1.5")
    parser.add_argument("--agent", choices=["navid", "uninavid"], default="uninavid")
    parser.add_argument("--navid-root", default="/file_system/vepfs/algorithm/intern03/mhw/NaVid-VLN-CE")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--agent-result-path", default="output/time_aware_scene/uninavid_agent_artifacts")
    parser.add_argument("--exp-save", default="data")
    parser.add_argument("--time-prompt-mode", choices=["dynamic_fuzzy", "ratio_fuzzy"], default="dynamic_fuzzy")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--scene", default="data/hm3d/")
    parser.add_argument("--scene-dataset", default="data/hm3d/hm3d_annotated_basis.scene_dataset_config.json")
    parser.add_argument("--success-dis", type=float, default=1.0)
    parser.add_argument("--no-render", action="store_true", default=False)
    parser.add_argument("--render", dest="no_render", action="store_false")
    parser.add_argument("--output-dir", default="output/time_aware_scene/uninavid")
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.no_render:
        raise ValueError("NaVid/Uni-NaVid requires RGB observations. Do not use --no-render.")

    records = load_records(args.episodes, args.split, args.limit)
    budget_ratios = [args.budget_ratio] if args.budget_ratio is not None else parse_budget_ratios(args.budget_ratios)
    agent = load_agent(args)
    summary_rows = []
    for ratio in budget_ratios:
        args.budget_ratio = ratio
        results = []
        print(f"\n===== {args.agent} budget_ratio={ratio} split={args.split} episodes={len(records)} =====")
        for index, record in enumerate(records, start=1):
            print(f"===== [{index}/{len(records)}] {record_id(record)} =====")
            if args.quiet:
                with contextlib.redirect_stdout(io.StringIO()):
                    results.append(run_episode(args, agent, record))
            else:
                results.append(run_episode(args, agent, record))
        summary = summarize(results)
        summary["split"] = args.split
        summary["budget_ratio"] = ratio
        summary["limit"] = args.limit
        summary["baseline"] = args.agent
        summary["time_prompt_mode"] = args.time_prompt_mode
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        output = Path(args.output_dir) / f"{args.agent}_{args.split}_budget_{ratio_name(ratio)}.json"
        write_json(output, summary, results)
        summary_rows.append(summary)

    summary_csv = Path(args.summary_csv) if args.summary_csv else Path(args.output_dir) / f"{args.agent}_{args.split}_summary.csv"
    write_summary_csv(summary_csv, summary_rows)


if __name__ == "__main__":
    main()
