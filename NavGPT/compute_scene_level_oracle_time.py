# 用途：NavGPT-style Time-Aware VLN 脚本。
#!/usr/bin/env python3
"""Compute scene-level oracle time budgets by enumerating target orders.

The input is a scene-level time-aware JSONL. For each episode, this script first
estimates pairwise navigation step costs between the start point and all target
waypoints with Habitat's GreedyGeodesicFollower. It then enumerates all target
orders and keeps the minimum-cost order that visits every target.

The resulting oracle time is used to rewrite:
  time_budgets = ratio * oracle_optimal_time
"""

import argparse
import contextlib
import csv
import io
import itertools
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import habitat_sim
import numpy as np
import quaternion

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from habitat_base.time_aware_simulation import TimeAwareSceneSimulator
from tools.run_time_aware_greedy import record_id, record_to_config


def load_jsonl(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(path, records):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output


def write_summary_csv(path, rows):
    if not rows:
        return None
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output


def set_agent_state(sim, position, yaw_degrees):
    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array(sim.pathfinder.snap_point(position), dtype=np.float32)
    yaw = math.radians(float(yaw_degrees or 0.0))
    agent_state.rotation = quaternion.from_rotation_vector([0.0, yaw, 0.0])
    sim.agent.set_state(agent_state)


def geodesic_distance(sim, goal):
    start, _ = sim.return_state()
    path = habitat_sim.nav.ShortestPath()
    path.requested_start = np.array(start, dtype=np.float32)
    path.requested_end = np.array(goal, dtype=np.float32)
    if not sim.pathfinder.find_path(path):
        return math.inf
    return path.geodesic_distance


def follower_step_cost(sim, start_position, goal_position, yaw_degrees, max_pair_steps):
    goal = sim.pathfinder.snap_point(goal_position)
    set_agent_state(sim, start_position, yaw_degrees)

    steps = 0
    while steps < max_pair_steps:
        distance = geodesic_distance(sim, goal)
        if math.isinf(distance):
            return None
        if distance < sim.args.success_dis:
            return steps + 1  # final stop action

        try:
            action = sim.get_next_action(goal)
        except habitat_sim.errors.GreedyFollowerError:
            return None
        if action is None or action == "stop":
            return steps + 1 if distance < sim.args.success_dis else None

        sim.sim.step(action)
        steps += 1
    return None


def make_sim(args, record):
    sim_args = SimpleNamespace(
        scene=args.scene,
        scene_dataset=args.scene_dataset,
        success_dis=args.success_dis,
        max_step=args.max_pair_steps,
        no_render=True,
    )
    return TimeAwareSceneSimulator(
        sim_args,
        record_to_config(record),
        time_budget=args.max_pair_steps,
        target_values=[target.get("value", 1.0) for target in record["targets"]],
    )


def compute_pairwise_costs(args, record):
    positions = [record["start_position"]] + [
        target["target_position"] for target in record["targets"]
    ]
    size = len(positions)
    costs = [[None for _ in range(size)] for _ in range(size)]
    if args.quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            sim = make_sim(args, record)
    else:
        sim = make_sim(args, record)
    try:
        for source in range(size):
            for target in range(1, size):
                if source == target:
                    continue
                costs[source][target] = follower_step_cost(
                    sim,
                    positions[source],
                    positions[target],
                    record.get("start_yaw") or 0,
                    args.max_pair_steps,
                )
    finally:
        sim.close()
    return costs


def order_cost(order, costs):
    total = 0
    current = 0
    for target_index in order:
        node_index = target_index + 1
        edge_cost = costs[current][node_index]
        if edge_cost is None:
            return None
        total += edge_cost
        current = node_index
    return total


def enumerate_optimal_order(record, costs):
    best_order = None
    best_cost = None
    target_indices = list(range(len(record["targets"])))
    for order in itertools.permutations(target_indices):
        cost = order_cost(order, costs)
        if cost is None:
            continue
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_order = list(order)
    return best_cost, best_order


def update_record(record, oracle_time, oracle_order, costs, budget_ratios):
    updated = dict(record)
    updated.pop("oracle_time_proxy_ordered_sum", None)
    updated.pop("pairwise_step_costs", None)
    updated["oracle_optimal_time"] = int(oracle_time) if oracle_time is not None else None
    updated["oracle_optimal_order"] = oracle_order
    updated["oracle_time_source"] = "pairwise_follower_step_enum"
    if oracle_time is not None:
        updated["time_budgets"] = {
            str(ratio): max(1, int(round(float(ratio) * oracle_time)))
            for ratio in budget_ratios
        }
    for target in updated.get("targets", []):
        target.pop("source_ordered_gt_step", None)
        for occurrence in target.get("source_occurrences", []):
            occurrence.pop("source_ordered_gt_step", None)
    return updated


def parse_budget_ratios(value):
    return [float(item) for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--budget-ratios", default="0.5,1.0,1.5")
    parser.add_argument("--scene", default="data/hm3d/")
    parser.add_argument(
        "--scene-dataset",
        default="data/hm3d/hm3d_annotated_basis.scene_dataset_config.json",
    )
    parser.add_argument("--success-dis", type=float, default=1.0)
    parser.add_argument("--max-pair-steps", type=int, default=2000)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    budget_ratios = parse_budget_ratios(args.budget_ratios)
    records = load_jsonl(args.input)
    if args.limit:
        records = records[:args.limit]

    updated_records = []
    summary_rows = []
    for index, record in enumerate(records, start=1):
        if not args.quiet:
            print(f"===== [{index}/{len(records)}] {record_id(record)} =====")
        costs = compute_pairwise_costs(args, record)
        oracle_time, oracle_order = enumerate_optimal_order(record, costs)
        updated = update_record(record, oracle_time, oracle_order, costs, budget_ratios)
        updated_records.append(updated)

        row = {
            "episode_id": record_id(record),
            "scene": record["scene"],
            "num_targets": len(record["targets"]),
            "oracle_optimal_time": oracle_time,
            "oracle_optimal_order": " ".join(map(str, oracle_order or [])),
            "unreachable_full_order": oracle_time is None,
        }
        for ratio in budget_ratios:
            row[f"budget_{ratio}"] = updated["time_budgets"].get(str(ratio))
        summary_rows.append(row)

    output = write_jsonl(args.output, updated_records)
    summary = write_summary_csv(args.summary_csv, summary_rows) if args.summary_csv else None
    failed = sum(row["unreachable_full_order"] for row in summary_rows)
    print(f"episodes: {len(updated_records)}")
    print(f"unreachable full-order episodes: {failed}")
    print(f"wrote: {output}")
    if summary:
        print(f"wrote: {summary}")


if __name__ == "__main__":
    main()
