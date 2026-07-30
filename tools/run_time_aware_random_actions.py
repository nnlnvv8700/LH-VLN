# 用途：在 Habitat 中按指定概率执行随机前进/转向原子动作，作为无视觉决策的时间预算基线。
#!/usr/bin/env python3
"""Run a primitive random-action baseline for scene-level Time-Aware VLN.

Unlike the random waypoint baseline, this baseline does not use candidate
waypoints or a Habitat follower. It samples low-level actions directly from
move_forward / turn_left / turn_right, then lets the runner perform the same
environment-side auto-stop check used by the waypoint experiments.
"""

import argparse
import contextlib
import io
import json
import random
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
    record_to_config,
    summarize,
    write_json,
    write_summary_csv,
)


PRIMITIVE_ACTIONS = ("move_forward", "turn_left", "turn_right")


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


def get_time_budget(record, ratio):
    ratio_key = str(ratio)
    time_budget = record.get("time_budgets", {}).get(ratio_key)
    if time_budget is not None:
        return time_budget
    oracle_time = record.get("oracle_optimal_time")
    if oracle_time is None:
        raise ValueError(f"Missing budget ratio {ratio_key} for {record_id(record)}")
    return max(1, int(round(float(ratio) * float(oracle_time))))


def nearby_target_indices(args, sim):
    nearby = []
    for index in sorted(sim.remaining_targets):
        info = sim.get_target_info(index)
        distance = info["geo dis"]
        if np.isfinite(distance) and distance < args.success_dis:
            nearby.append(index)
    return nearby


def auto_stop_if_near_target(args, sim):
    completed = []
    while not sim.episode_over:
        if not nearby_target_indices(args, sim):
            break
        before = set(sim.completed_targets)
        sim.actor("stop")
        after = set(sim.completed_targets)
        new_completed = sorted(after - before)
        if not new_completed:
            break
        completed.extend(new_completed)
    return completed


def position_distance(before, after):
    before_arr = np.asarray(before, dtype=np.float32)
    after_arr = np.asarray(after, dtype=np.float32)
    return float(np.linalg.norm(after_arr - before_arr))


def choose_action(args, rng):
    weights = [args.forward_weight, args.left_weight, args.right_weight]
    return rng.choices(PRIMITIVE_ACTIONS, weights=weights, k=1)[0]


def run_episode(args, record):
    time_budget = get_time_budget(record, args.budget_ratio)
    rng = random.Random(args.seed + abs(hash(record_id(record))) % 1_000_000_007 + int(float(args.budget_ratio) * 1000))
    sim = make_sim(args, record, time_budget)
    trace = []
    try:
        sim.actor("stop")
        while not sim.episode_over and sim.time_used < time_budget and len(trace) < args.max_decisions:
            action = choose_action(args, rng)
            before_pos, _ = sim.return_state()
            sim.actor(action)
            after_pos, _ = sim.return_state()
            auto_completed = []
            if args.auto_stop and not sim.episode_over:
                auto_completed = auto_stop_if_near_target(args, sim)
            event = "random_action"
            if auto_completed:
                event += "_auto_stop_completed"
            trace.append(
                {
                    "decision": len(trace) + 1,
                    "time_used": sim.time_used,
                    "time_remaining": sim.time_remaining,
                    "action": action,
                    "movement_distance": position_distance(before_pos, after_pos),
                    "auto_completed_targets": auto_completed,
                    "completed_targets": sorted(sim.completed_targets),
                    "event": event,
                }
            )
        result = sim.return_results()
    finally:
        sim.close()

    result["task_id"] = record_id(record)
    result["split"] = record["split"]
    result["scene"] = record["scene"]
    result["robot"] = record["robot"]
    result["baseline"] = args.baseline_name
    result["observation_mode"] = "primitive_random_actions"
    result["auto_stop"] = bool(args.auto_stop)
    result["random_action_trace"] = trace
    return result


def write_summary_csv_with_baseline(path, rows, args):
    for row in rows:
        row["baseline"] = args.baseline_name
        row["observation_mode"] = "primitive_random_actions"
        row["auto_stop"] = bool(args.auto_stop)
        row["seed"] = args.seed
    write_summary_csv(path, rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes",
        default="/file_system/nas/algorithm/Intern03/data/time_aware_scene/all_episodes_spatial_start_floor_oracle_time.jsonl",
    )
    parser.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--budget-ratio", type=float, default=None)
    parser.add_argument("--budget-ratios", default="0.5,1.0,1.5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--forward-weight", type=float, default=0.5)
    parser.add_argument("--left-weight", type=float, default=0.25)
    parser.add_argument("--right-weight", type=float, default=0.25)
    parser.add_argument("--max-decisions", type=int, default=10000)
    parser.add_argument("--auto-stop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--baseline-name", default="random_primitive_actions_seed0")
    parser.add_argument("--scene", default="data/hm3d/")
    parser.add_argument(
        "--scene-dataset",
        default="data/hm3d/hm3d_annotated_basis.scene_dataset_config.json",
    )
    parser.add_argument("--success-dis", type=float, default=1.0)
    parser.add_argument("--no-render", action="store_true", default=True)
    parser.add_argument("--render", dest="no_render", action="store_false")
    parser.add_argument("--output-dir", default="output/time_aware_scene/random_primitive_actions")
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    records = load_records(args.episodes, args.split, args.limit)
    budget_ratios = [args.budget_ratio] if args.budget_ratio is not None else parse_budget_ratios(args.budget_ratios)
    summary_rows = []
    for ratio in budget_ratios:
        args.budget_ratio = ratio
        results = []
        print(f"\n===== RandomPrimitive budget_ratio={ratio} split={args.split} episodes={len(records)} =====")
        for index, record in enumerate(records, start=1):
            print(f"===== [{index}/{len(records)}] {record_id(record)} =====")
            if args.quiet:
                with contextlib.redirect_stdout(io.StringIO()):
                    results.append(run_episode(args, record))
            else:
                results.append(run_episode(args, record))

        summary = summarize(results)
        summary["split"] = args.split
        summary["budget_ratio"] = ratio
        summary["limit"] = args.limit
        summary["baseline"] = args.baseline_name
        summary["observation_mode"] = "primitive_random_actions"
        summary["auto_stop"] = bool(args.auto_stop)
        summary["seed"] = args.seed
        print("\nsummary:")
        print(json.dumps(summary, ensure_ascii=False, indent=2))

        output = Path(args.output_dir) / f"random_primitive_{args.split}_budget_{ratio_name(ratio)}.json"
        write_json(output, summary, results)
        jsonl_output = Path(args.output_dir) / f"random_primitive_{args.split}_budget_{ratio_name(ratio)}.jsonl"
        jsonl_output.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_output.open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"wrote: {output}")
        print(f"wrote: {jsonl_output}")
        summary_rows.append(summary)

    summary_csv = Path(args.summary_csv) if args.summary_csv else Path(args.output_dir) / f"random_primitive_{args.split}_summary.csv"
    write_summary_csv_with_baseline(summary_csv, summary_rows, args)
    print(f"wrote: {summary_csv}")


if __name__ == "__main__":
    main()
