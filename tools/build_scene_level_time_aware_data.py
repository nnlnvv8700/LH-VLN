#!/usr/bin/env python3
"""Build scene-level time-aware tasks by stitching LH-VLN episodes.

This is the V2 dataset constructor. It groups multiple original LH-VLN episodes
from the same HM3D scene into a longer scene-level task. The first version is
data-only: it creates 4-8 task groups and uses the sum of ordered oracle steps as
a proxy time reference. A later oracle pass should replace the proxy with an
enumerated/simulated optimal scene-level time.
"""

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def load_records(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def task_sort_key(record):
    batch = record.get("batch") or record.get("task_id", "").split("/", 1)[0]
    episode = record.get("task_id", "").split("/")[-1]
    try:
        episode_index = int(episode.replace("episode_", ""))
    except ValueError:
        episode_index = episode
    return batch, episode_index, record.get("task_id")


def choose_scene_tasks(records, min_tasks, max_tasks, coverage):
    """Choose one stitched group for a scene.

    The goal is to cover about `coverage` of available tasks while staying in the
    requested 4-8 task range. Scenes with fewer than min_tasks are skipped.
    """
    if len(records) < min_tasks:
        return []
    target_count = int(math.ceil(len(records) * coverage))
    task_count = min(max_tasks, max(min_tasks, target_count))
    return sorted(records, key=task_sort_key)[:task_count]


def flatten_targets(tasks):
    targets = []
    for task_index, task in enumerate(tasks):
        for target in task.get("targets", []):
            merged = dict(target)
            merged["global_index"] = len(targets)
            merged["source_task_index"] = task_index
            merged["source_task_id"] = task["task_id"]
            merged["source_target_index"] = target.get("index")
            targets.append(merged)
    return targets


def build_scene_episode(
    split,
    scene,
    tasks,
    scene_task_count,
    budget_ratios,
    coverage,
    oracle_time_source,
):
    ordered_sum = sum(task.get("oracle_total_steps_ordered") or 0 for task in tasks)
    oracle_time_proxy = max(1, int(ordered_sum))
    budgets = {
        str(ratio): max(1, int(round(oracle_time_proxy * ratio)))
        for ratio in budget_ratios
    }
    scene_episode = {
        "scene_episode_id": f"{split}/{scene}/stitched_0",
        "split": split,
        "scene": scene,
        "robot": tasks[0].get("robot"),
        "start_position": tasks[0].get("start_position"),
        "start_yaw": tasks[0].get("start_yaw"),
        "scene_level": True,
        "unordered_tasks": True,
        "unordered_targets": True,
        "time_unit": "step",
        "num_source_tasks": len(tasks),
        "source_task_ids": [task["task_id"] for task in tasks],
        "instructions": [task.get("instruction") for task in tasks],
        "stitched_instruction": "\n".join(
            f"Task {index + 1}: {task.get('instruction')}"
            for index, task in enumerate(tasks)
        ),
        "targets": flatten_targets(tasks),
        "oracle_time_proxy_ordered_sum": oracle_time_proxy,
        "oracle_time_source": oracle_time_source,
        "time_budgets": budgets,
        "budget_ratios": budget_ratios,
        "coverage_target": coverage,
        "source_total_tasks_in_scene": scene_task_count,
        "source_task_coverage_in_scene": len(tasks) / scene_task_count if scene_task_count else 0.0,
    }
    return scene_episode


def describe(values):
    if not values:
        return "n=0"
    values = sorted(values)
    return (
        f"n={len(values)} min={values[0]} median={statistics.median(values):.1f} "
        f"mean={statistics.mean(values):.1f} max={values[-1]}"
    )


def write_jsonl(path, records):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/time_aware/episodes.jsonl")
    parser.add_argument("--output", default="data/time_aware_scene/test_episodes.jsonl")
    parser.add_argument("--split", default="test")
    parser.add_argument("--min-tasks", type=int, default=4)
    parser.add_argument("--max-tasks", type=int, default=8)
    parser.add_argument("--coverage", type=float, default=0.8)
    parser.add_argument("--budget-ratios", default="0.5,1.0,1.5")
    parser.add_argument(
        "--oracle-time-source",
        default="ordered_sum_proxy",
        help="Metadata only. Later replace with enumerated_simulated_optimal.",
    )
    args = parser.parse_args()

    budget_ratios = [float(item) for item in args.budget_ratios.split(",") if item]
    records = [
        record for record in load_records(args.input)
        if args.split == "all" or record.get("split") == args.split
    ]
    by_scene = defaultdict(list)
    for record in records:
        by_scene[record["scene"]].append(record)

    scene_episodes = []
    skipped = {}
    for scene, scene_records in sorted(by_scene.items()):
        selected = choose_scene_tasks(
            scene_records,
            min_tasks=args.min_tasks,
            max_tasks=args.max_tasks,
            coverage=args.coverage,
        )
        if not selected:
            skipped[scene] = len(scene_records)
            continue
        scene_episodes.append(
            build_scene_episode(
                args.split,
                scene,
                selected,
                len(scene_records),
                budget_ratios,
                args.coverage,
                args.oracle_time_source,
            )
        )

    output = write_jsonl(args.output, scene_episodes)

    task_counts = [record["num_source_tasks"] for record in scene_episodes]
    target_counts = [len(record["targets"]) for record in scene_episodes]
    oracle_times = [record["oracle_time_proxy_ordered_sum"] for record in scene_episodes]
    covered_tasks = sum(task_counts)
    available_tasks = sum(len(items) for items in by_scene.values())
    eligible_tasks = sum(len(items) for items in by_scene.values() if len(items) >= args.min_tasks)
    source_scene_counts = [len(items) for items in by_scene.values()]

    print("Scene-level time-aware data")
    print(f"input: {args.input}")
    print(f"split: {args.split}")
    print(f"source scenes: {len(by_scene)}")
    print(f"source tasks: {available_tasks}")
    print(f"source tasks per scene: {describe(source_scene_counts)}")
    print(f"stitched episodes: {len(scene_episodes)}")
    print(f"skipped scenes (<{args.min_tasks} tasks): {len(skipped)}")
    print(f"covered source tasks among all source tasks: {covered_tasks}/{available_tasks} ({covered_tasks / available_tasks:.2%})")
    print(f"covered source tasks among eligible scenes: {covered_tasks}/{eligible_tasks} ({covered_tasks / eligible_tasks:.2%})")
    print(f"tasks per stitched episode: {describe(task_counts)}")
    print(f"targets per stitched episode: {describe(target_counts)}")
    print(f"oracle_time_proxy_ordered_sum: {describe(oracle_times)}")
    for ratio in budget_ratios:
        key = str(ratio)
        values = [record["time_budgets"][key] for record in scene_episodes]
        print(f"budget_ratio_{key}: {describe(values)}")
    print("selected task count distribution:", dict(sorted(Counter(task_counts).items())))
    print(f"wrote: {output}")

    if scene_episodes:
        print("\nsample:")
        print(json.dumps(scene_episodes[0], ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
