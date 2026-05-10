#!/usr/bin/env python3
"""Run a nearest-target greedy baseline for time-aware LH-VLN tasks."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from habitat_base.time_aware_simulation import TimeAwareSceneSimulator


def region_number(region_id):
    if region_id is None:
        return None
    return str(region_id).replace("Region", "").strip()


def load_records(path, split, limit):
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if split and record["split"] != split:
                continue
            records.append(record)
            if limit and len(records) >= limit:
                break
    return records


def record_to_config(record):
    targets = record["targets"]
    return {
        "Task instruction": record["instruction"],
        "Scene": record["scene"],
        "Robot": record["robot"],
        "Object": [target["name"] for target in targets],
        "Region": [region_number(target["region_id"]) for target in targets],
        "Batch": "/" + record["batch"],
    }


def run_episode(args, record):
    ratio_key = str(args.budget_ratio)
    time_budget = record["time_budgets"].get(ratio_key)
    if time_budget is None:
        raise ValueError(f"Missing budget ratio {ratio_key} for {record['task_id']}")

    config = record_to_config(record)
    target_values = [target.get("value", 1.0) for target in record["targets"]]
    sim_args = SimpleNamespace(
        scene=args.scene,
        scene_dataset=args.scene_dataset,
        success_dis=args.success_dis,
        max_step=time_budget,
        no_render=args.no_render,
    )
    sim = TimeAwareSceneSimulator(
        sim_args,
        config,
        time_budget=time_budget,
        target_values=target_values,
    )
    try:
        action = "stop"
        sim.actor(action)
        while not sim.episode_over:
            action = sim.get_next_action_to_nearest_target()
            sim.actor(action)
        result = sim.return_results()
    finally:
        sim.close()
    result["task_id"] = record["task_id"]
    result["split"] = record["split"]
    result["scene"] = record["scene"]
    result["robot"] = record["robot"]
    return result


def summarize(results):
    if not results:
        return {}
    return {
        "episodes": len(results),
        "success_at_budget": sum(item["success_at_budget"] for item in results) / len(results),
        "completion_rate": sum(item["completion_rate"] for item in results) / len(results),
        "reward_rate": sum(item["reward_rate"] for item in results) / len(results),
        "avg_time_used": sum(item["time_used"] for item in results) / len(results),
        "avg_failed_stops": sum(item["failed_stops"] for item in results) / len(results),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default="data/time_aware/episodes.jsonl")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--budget-ratio", type=float, default=1.0)
    parser.add_argument("--scene", default="data/hm3d/")
    parser.add_argument(
        "--scene-dataset",
        default="data/hm3d/hm3d_annotated_basis.scene_dataset_config.json",
    )
    parser.add_argument("--success-dis", type=float, default=1.0)
    parser.add_argument("--no-render", action="store_true", default=True)
    parser.add_argument("--render", dest="no_render", action="store_false")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    records = load_records(args.episodes, args.split, args.limit)
    results = []
    for index, record in enumerate(records):
        print(f"===== [{index + 1}/{len(records)}] {record['task_id']} =====")
        results.append(run_episode(args, record))

    summary = summarize(results)
    print("\nsummary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump({"summary": summary, "results": results}, handle, ensure_ascii=False, indent=2)
        print(f"wrote: {output}")


if __name__ == "__main__":
    main()
