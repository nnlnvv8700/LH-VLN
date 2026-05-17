#!/usr/bin/env python3
"""Run a NavGPT-style high-level target planner for time-aware LH-VLN tasks.

The first version keeps low-level navigation in Habitat-Sim and only swaps the
target-selection layer. This makes it easy to compare a future LLM planner with
nearest-target greedy and oracle target ordering.
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


def format_targets(record, sim):
    lines = []
    for target_index in sorted(sim.remaining_targets):
        target = record["targets"][target_index]
        info = sim.get_target_info(target_index)
        lines.append(
            "- index={index}, name={name}, region={region}, estimated_distance={distance:.2f}".format(
                index=target_index,
                name=target["name"],
                region=target.get("region_name") or target.get("region") or "unknown",
                distance=info["geo dis"],
            )
        )
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
        time_text = FUZZY_TIME_PROMPTS[args.fuzzy_time]
    else:
        time_text = "No time hint is provided."

    return f"""You are a high-level planner for a time-aware VLN task.
Choose the next target for the navigation system.
The targets are unordered: you do not need to follow the order in the instruction.
Your goal is to maximize the number of completed targets before the budget runs out.
When time is limited, prefer a target that is likely reachable soon.
The estimated_distance value is the shortest-path distance from the current position; smaller is usually better.

{time_text}

Instruction:
{record["instruction"]}

Completed targets: {completed_text}

Remaining targets:
{format_targets(record, sim)}

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
        raise ValueError(f"Missing budget ratio {ratio_key} for {record['task_id']}")
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
    result["task_id"] = record["task_id"]
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
    parser.add_argument("--episodes", default="data/time_aware/episodes.jsonl")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--budget-ratio", type=float, default=None)
    parser.add_argument(
        "--budget-ratios",
        default="0.5,0.75,1.0,1.25",
        help="Comma-separated budget ratios. Ignored when --budget-ratio is set.",
    )
    parser.add_argument("--planner", default="nearest", choices=["nearest", "first", "random", "llm"])
    parser.add_argument("--time-prompt", default="explicit", choices=["explicit", "fuzzy", "none"])
    parser.add_argument("--fuzzy-time", default="tight", choices=sorted(FUZZY_TIME_PROMPTS))
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
    parser.add_argument("--output-dir", default="output/time_aware")
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
            print(f"===== [{index + 1}/{len(records)}] {record['task_id']} =====")
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
