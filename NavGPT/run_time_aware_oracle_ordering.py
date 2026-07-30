# 用途：NavGPT-style Time-Aware VLN 脚本。
#!/usr/bin/env python3
"""Run an oracle target-ordering baseline for time-aware LH-VLN tasks."""

import argparse
import contextlib
import csv
import io
import itertools
import json
import sys
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
    record_to_config,
    summarize,
    write_json,
    write_summary_csv,
)


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


def run_order(args, record, order, time_budget):
    sim = make_sim(args, record, time_budget)
    try:
        sim.actor("stop")
        for target_index in order:
            if sim.episode_over:
                break
            if target_index not in sim.remaining_targets:
                continue

            while not sim.episode_over and target_index in sim.remaining_targets:
                action = sim.get_next_action_to_target(target_index)
                if action == "stop":
                    sim.actor("stop", stop_target_index=target_index)
                    if target_index in sim.remaining_targets:
                        sim._abandon_target(target_index, "failed_stop_in_order")
                    break
                sim.actor(action)

        result = sim.return_results()
    finally:
        sim.close()

    result["target_order"] = list(order)
    result["target_order_names"] = [record["targets"][index]["name"] for index in order]
    return result


def result_key(result):
    return (
        result["reward"],
        sum(result["successes"]),
        int(result["success_at_budget"]),
        -result["time_used"],
        -len(result.get("abandoned_targets", [])),
    )


def run_episode(args, record):
    ratio_key = str(args.budget_ratio)
    time_budget = record["time_budgets"].get(ratio_key)
    if time_budget is None:
        raise ValueError(f"Missing budget ratio {ratio_key} for {record_id(record)}")

    target_indices = list(range(len(record["targets"])))
    candidates = []
    for order in itertools.permutations(target_indices):
        candidates.append(run_order(args, record, order, time_budget))

    best = max(candidates, key=result_key)
    best["candidate_orders"] = len(candidates)
    best["task_id"] = record_id(record)
    best["split"] = record["split"]
    best["scene"] = record["scene"]
    best["robot"] = record["robot"]
    return best


def write_summary_csv_with_baseline(path, rows):
    for row in rows:
        row["baseline"] = "oracle_ordering"
    write_summary_csv(path, rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default="data/time_aware/episodes.jsonl")
    parser.add_argument("--split", default="val", choices=["train", "val", "test", "all"])
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--budget-ratio", type=float, default=None)
    parser.add_argument(
        "--budget-ratios",
        default="0.5,0.75,1.0,1.25",
        help="Comma-separated budget ratios. Ignored when --budget-ratio is set.",
    )
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

    records = load_records(args.episodes, args.split, args.limit)
    budget_ratios = [args.budget_ratio] if args.budget_ratio is not None else parse_budget_ratios(args.budget_ratios)
    if args.output and len(budget_ratios) != 1:
        raise ValueError("--output can only be used with one budget ratio. Use --output-dir for sweeps.")

    summary_rows = []
    for ratio in budget_ratios:
        args.budget_ratio = ratio
        results = []
        print(f"\n===== oracle_ordering budget_ratio={ratio} split={args.split} episodes={len(records)} =====")
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
        summary["baseline"] = "oracle_ordering"
        print("\nsummary:")
        print(json.dumps(summary, ensure_ascii=False, indent=2))

        output = output_path_for_ratio(args.output, args.output_dir, args.split, ratio)
        if output:
            output = output.with_name(output.name.replace("greedy", "oracle_ordering"))
            write_json(output, summary, results)
        summary_rows.append(summary)

    summary_csv = Path(args.summary_csv) if args.summary_csv else Path(args.output_dir) / f"oracle_ordering_{args.split}_summary.csv"
    write_summary_csv_with_baseline(summary_csv, summary_rows)


if __name__ == "__main__":
    main()
