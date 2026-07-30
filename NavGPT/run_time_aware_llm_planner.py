#!/usr/bin/env python3
"""Run budget-conditioned unordered long-horizon planning with an oracle follower.

The LLM only chooses the next unordered target. Habitat-Sim's
GreedyGeodesicFollower executes the low-level shortest-path actions. This
separates time-aware high-level scheduling from visual recognition, local
avoidance, and motor control.
"""

import argparse
import contextlib
import csv
import io
import json
import random
import re
import sys
import subprocess
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from habitat_base.time_aware_simulation import TimeAwareSceneSimulator
from tools.run_time_aware_greedy import (
    load_records,
    output_path_for_ratio,
    parse_budget_ratios,
    record_id,
    record_instruction,
    record_to_config,
    summarize,
    write_json,
    write_summary_csv,
)


FUZZY_TIME_PROMPTS = {
    "sufficient": "The time is sufficient. Try to complete the whole instruction.",
    "tight": "The time is tight. Prioritize useful progress.",
    "insufficient": "The time is relatively insufficient. Complete as many targets as possible.",
}

FUZZY_BY_RATIO = {
    "0.5": "The time is very limited. You are in a hurry. Prioritize quick useful progress.",
    "1.0": "The time is moderate. Balance completing targets with avoiding wasted exploration.",
    "1.5": "The time is sufficient. Try to complete all targets carefully.",
}


def make_sim(args, record, time_budget):
    config = record_to_config(record)
    target_values = [target.get("value", 1.0) for target in record["targets"]]
    sim_args = SimpleNamespace(
        scene=args.scene,
        scene_dataset=args.scene_dataset,
        success_dis=args.success_dis,
        max_step=time_budget,
        no_render=args.no_render,
    )
    return TimeAwareSceneSimulator(
        sim_args,
        config,
        time_budget=time_budget,
        target_values=target_values,
    )


def fuzzy_time_text(args):
    if args.fuzzy_time != "auto":
        return FUZZY_TIME_PROMPTS[args.fuzzy_time]
    return FUZZY_BY_RATIO.get(
        str(args.budget_ratio),
        "The time is limited. Complete as many targets as possible.",
    )


def format_targets(args, record, sim):
    lines = []
    for target_index in sorted(sim.remaining_targets):
        target = record["targets"][target_index]
        line = "- index={index}, name={name}, region={region}".format(
            index=target_index,
            name=target["name"],
            region=target.get("region_name") or target.get("region") or "unknown",
        )
        if args.planner_observation == "oracle_distance":
            info = sim.get_target_info(target_index)
            line += ", estimated_distance={distance:.2f}".format(distance=info["geo dis"])
        lines.append(line)
    return "\n".join(lines)


def build_prompt(args, record, sim):
    completed_names = [
        record["targets"][target_index]["name"]
        for target_index in sim.completed_targets
    ]
    completed_text = ", ".join(completed_names) if completed_names else "none"

    if args.time_prompt == "explicit":
        time_text = (
            f"The total time budget for this episode is {sim.time_budget} steps.\n"
            f"The agent has used {sim.time_used} steps and has {sim.time_remaining} steps remaining."
        )
    elif args.time_prompt == "fuzzy":
        time_text = fuzzy_time_text(args)
    else:
        time_text = "No time hint is provided."

    distance_hint = ""
    if args.planner_observation == "oracle_distance":
        distance_hint = "The estimated_distance value is the shortest-path distance from the current position; smaller is usually better.\n"
    else:
        distance_hint = ""

    return f"""You are a high-level target scheduler for a budget-conditioned unordered long-horizon navigation task.
Choose the next target for the navigation system.
The targets are unordered: you do not need to follow the order in the instruction.
Your goal is to maximize the number of completed targets before time runs out.
When time is limited, choose a target that is likely to create useful progress soon.
{distance_hint}

{time_text}

Instruction:
{record_instruction(record)}

Completed targets: {completed_text}

Remaining targets:
{format_targets(args, record, sim)}

Only output one target index from the remaining targets. Output the integer only.
"""


def choose_target(args, record, sim, prompt):
    remaining = sorted(sim.remaining_targets)
    if not remaining:
        return None, ""

    if args.planner == "llm":
        return choose_target_with_llm(args, sim, prompt)
    if args.planner == "first":
        return remaining[0], str(remaining[0])
    if args.planner == "random":
        target_index = random.choice(remaining)
        return target_index, str(target_index)
    if args.planner == "nearest":
        nearest = sim.select_nearest_target()
        target_index = nearest["target_index"] if nearest else remaining[0]
        return target_index, str(target_index)
    if args.planner == "oracle_order":
        for target_index in record.get("oracle_optimal_order") or []:
            if target_index in sim.remaining_targets:
                return target_index, str(target_index)
        return remaining[0], str(remaining[0])

    raise ValueError(f"Unsupported planner: {args.planner}")


def parse_target_index(raw_response, remaining):
    """Parse the first valid target index from an LLM response."""
    remaining = set(remaining)
    for match in re.finditer(r"-?\d+", raw_response):
        candidate = int(match.group(0))
        if candidate in remaining:
            return candidate
    return None


def choose_target_with_llm(args, sim, prompt):
    if not args.llm_command:
        raise ValueError("--planner llm requires --llm-command")

    remaining = sorted(sim.remaining_targets)
    try:
        completed = subprocess.run(
            args.llm_command,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=args.llm_timeout,
            check=False,
            shell=True,
        )
    except subprocess.TimeoutExpired as exc:
        raw_response = f"[stderr]\nLLM command timed out after {args.llm_timeout} seconds"
        if exc.stdout:
            raw_response = f"{exc.stdout.strip()}\n{raw_response}".strip()
        if exc.stderr:
            raw_response = f"{raw_response}\n{exc.stderr.strip()}".strip()
        return invalid_llm_fallback(args, sim, remaining, raw_response)

    stdout = completed.stdout.strip()
    raw_response = stdout
    if completed.stderr.strip():
        raw_response = f"{raw_response}\n[stderr]\n{completed.stderr.strip()}".strip()
    if completed.returncode != 0:
        raw_response = f"{raw_response}\n[returncode]\n{completed.returncode}".strip()
    target_index = parse_target_index(stdout, remaining)

    if target_index is not None:
        return target_index, raw_response
    return invalid_llm_fallback(args, sim, remaining, raw_response)


def invalid_llm_fallback(args, sim, remaining, raw_response):
    if args.invalid_llm_choice == "first":
        return remaining[0], raw_response
    if args.invalid_llm_choice == "nearest":
        nearest = sim.select_nearest_target()
        return nearest["target_index"] if nearest else remaining[0], raw_response
    if args.invalid_llm_choice == "end_episode":
        sim.episode_over = True
        return None, raw_response
    raise ValueError(f"Unsupported invalid choice policy: {args.invalid_llm_choice}")


def parse_budget(record, ratio):
    ratio_key = str(ratio)
    time_budget = record["time_budgets"].get(ratio_key)
    if time_budget is None:
        raise ValueError(f"Missing budget ratio {ratio_key} for {record_id(record)}")
    return time_budget


def navigate_to_target(sim, target_index):
    while not sim.episode_over and target_index in sim.remaining_targets:
        action = sim.get_next_action_to_target(target_index)
        if action == "stop":
            sim.actor("stop", stop_target_index=target_index)
            if target_index in sim.remaining_targets:
                sim._abandon_target(target_index, "planner_failed_stop")
            return
        sim.actor(action)


def run_episode(args, record):
    time_budget = parse_budget(record, args.budget_ratio)
    sim = make_sim(args, record, time_budget)
    planner_trace = []

    try:
        sim.actor("stop")
        while not sim.episode_over and sim.remaining_targets:
            prompt = build_prompt(args, record, sim)
            target_index, raw_response = choose_target(args, record, sim, prompt)
            if target_index is None:
                planner_trace.append(
                    {
                        "time_used": sim.time_used,
                        "time_remaining": sim.time_remaining,
                        "prompt": prompt if args.save_prompts else None,
                        "raw_response": raw_response,
                        "target_index": None,
                        "target": None,
                        "event": "planner_returned_no_valid_target",
                    }
                )
                break
            if target_index not in sim.remaining_targets:
                planner_trace.append(
                    {
                        "time_used": sim.time_used,
                        "time_remaining": sim.time_remaining,
                        "prompt": prompt if args.save_prompts else None,
                        "raw_response": raw_response,
                        "target_index": target_index,
                        "target": None,
                        "event": "planner_selected_completed_or_invalid_target",
                    }
                )
                sim._abandon_target(target_index, "planner_selected_completed_or_invalid_target")
                continue

            planner_trace.append(
                {
                    "time_used": sim.time_used,
                    "time_remaining": sim.time_remaining,
                    "prompt": prompt if args.save_prompts else None,
                    "raw_response": raw_response,
                    "target_index": target_index,
                    "target": record["targets"][target_index]["name"],
                    "event": "planner_selected_target",
                }
            )
            navigate_to_target(sim, target_index)

        result = sim.return_results()
    finally:
        sim.close()

    result["planner"] = args.planner
    result["time_prompt"] = args.time_prompt
    result["fuzzy_time"] = args.fuzzy_time if args.time_prompt == "fuzzy" else None
    result["planner_trace"] = planner_trace
    result["task_id"] = record_id(record)
    result["split"] = record["split"]
    result["scene"] = record["scene"]
    result["robot"] = record["robot"]
    return result


def write_summary_csv_with_planner(path, rows, args):
    for row in rows:
        row["baseline"] = f"planner_{args.planner}_{args.time_prompt}"
        if args.time_prompt == "fuzzy":
            row["fuzzy_time"] = args.fuzzy_time
    write_summary_csv(path, rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes",
        default="/file_system/nas/algorithm/Intern03/data/time_aware_scene/all_episodes_spatial_start_floor_oracle_time.jsonl",
    )
    parser.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--budget-ratio", type=float, default=None)
    parser.add_argument(
        "--budget-ratios",
        default="0.5,1.0,1.5",
        help="Comma-separated budget ratios. Ignored when --budget-ratio is set.",
    )
    parser.add_argument(
        "--planner",
        default="nearest",
        choices=["nearest", "first", "random", "oracle_order", "llm"],
    )
    parser.add_argument("--time-prompt", default="fuzzy", choices=["explicit", "fuzzy", "none"])
    parser.add_argument("--fuzzy-time", default="auto", choices=["auto"] + sorted(FUZZY_TIME_PROMPTS))
    parser.add_argument(
        "--planner-observation",
        default="text_only",
        choices=["text_only", "oracle_distance"],
        help="What target information the LLM sees. text_only hides oracle distances.",
    )
    parser.add_argument(
        "--llm-command",
        default=None,
        help="External command for --planner llm. It receives the prompt on stdin and prints a target index.",
    )
    parser.add_argument("--llm-timeout", type=float, default=60.0)
    parser.add_argument(
        "--invalid-llm-choice",
        default="end_episode",
        choices=["end_episode", "nearest", "first"],
        help="Policy when the LLM output does not contain a valid remaining target index.",
    )
    parser.add_argument("--save-prompts", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scene", default="data/hm3d/")
    parser.add_argument(
        "--scene-dataset",
        default="data/hm3d/hm3d_annotated_basis.scene_dataset_config.json",
    )
    parser.add_argument("--success-dis", type=float, default=1.0)
    parser.add_argument("--no-render", action="store_true", default=True)
    parser.add_argument("--render", dest="no_render", action="store_false")
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-dir", default="output/time_aware_scene/llm_oracle_follower")
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    records = load_records(args.episodes, args.split, args.limit)
    budget_ratios = [args.budget_ratio] if args.budget_ratio is not None else parse_budget_ratios(args.budget_ratios)
    if args.output and len(budget_ratios) != 1:
        raise ValueError("--output can only be used with one budget ratio. Use --output-dir for sweeps.")

    summary_rows = []
    for ratio in budget_ratios:
        args.budget_ratio = ratio
        results = []
        print(
            f"\n===== planner={args.planner} time_prompt={args.time_prompt} "
            f"budget_ratio={ratio} split={args.split} episodes={len(records)} ====="
        )
        for index, record in enumerate(records):
            print(f"===== [{index + 1}/{len(records)}] {record_id(record)} =====")
            if args.quiet:
                with contextlib.redirect_stdout(io.StringIO()):
                    results.append(run_episode(args, record))
            else:
                results.append(run_episode(args, record))

        summary = summarize(results)
        summary["split"] = args.split
        summary["budget_ratio"] = ratio
        summary["limit"] = args.limit
        summary["planner"] = args.planner
        summary["time_prompt"] = args.time_prompt
        summary["planner_observation"] = args.planner_observation
        if args.time_prompt == "fuzzy":
            summary["fuzzy_time"] = args.fuzzy_time
        print("\nsummary:")
        print(json.dumps(summary, ensure_ascii=False, indent=2))

        output = output_path_for_ratio(args.output, args.output_dir, args.split, ratio)
        if output:
            name = output.name.replace("greedy", f"planner_{args.planner}_{args.time_prompt}")
            output = output.with_name(name)
            write_json(output, summary, results)
        summary_rows.append(summary)

    summary_csv = Path(args.summary_csv) if args.summary_csv else Path(args.output_dir) / f"planner_{args.planner}_{args.split}_summary.csv"
    write_summary_csv_with_planner(summary_csv, summary_rows, args)


if __name__ == "__main__":
    main()
