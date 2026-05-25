#!/usr/bin/env python3
"""Build scene-level time-aware tasks by stitching LH-VLN target points.

This is the V2 dataset constructor. It groups multiple original LH-VLN episodes
from the same HM3D scene into a longer scene-level task. The primary unit is a
target point/event, not a whole original task: each new scene-level episode
contains 4-8 target points. The first version is data-only and uses the sum of
the selected targets' ordered oracle steps as a proxy time reference. A later
oracle pass should replace the proxy with an enumerated/simulated optimal
scene-level time.
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


def flatten_scene_targets(records):
    scene_targets = []
    for task in sorted(records, key=task_sort_key):
        gt_steps = task.get("gt_steps_ordered") or []
        for target in task.get("targets", []):
            source_target_index = target.get("index", 0)
            item = dict(target)
            item["source_task_id"] = task["task_id"]
            item["source_task_instruction"] = task.get("instruction")
            item["source_target_index"] = source_target_index
            item["source_ordered_gt_step"] = (
                gt_steps[source_target_index]
                if isinstance(source_target_index, int) and source_target_index < len(gt_steps)
                else None
            )
            scene_targets.append(item)
    return scene_targets


def choose_scene_targets(records, min_targets, max_targets, coverage):
    """Choose 4-8 target points for a scene.

    The goal is to cover about `coverage` of available targets while staying in the
    requested 4-8 target range. Scenes with fewer than min_targets are skipped.
    """
    scene_targets = flatten_scene_targets(records)
    if len(scene_targets) < min_targets:
        return []
    target_count = int(math.ceil(len(scene_targets) * coverage))
    target_count = min(max_targets, max(min_targets, target_count))
    selected = scene_targets[:target_count]
    for index, target in enumerate(selected):
        target["global_index"] = index
    return selected


def build_scene_episode(
    split,
    scene,
    scene_records,
    selected_targets,
    scene_task_count,
    budget_ratios,
    coverage,
    oracle_time_source,
):
    ordered_sum = sum(
        target.get("source_ordered_gt_step") or 0
        for target in selected_targets
    )
    oracle_time_proxy = max(1, int(ordered_sum))
    budgets = {
        str(ratio): max(1, int(round(oracle_time_proxy * ratio)))
        for ratio in budget_ratios
    }
    source_tasks = []
    seen_task_ids = set()
    task_lookup = {task["task_id"]: task for task in scene_records}
    for target in selected_targets:
        task_id = target["source_task_id"]
        if task_id not in seen_task_ids:
            seen_task_ids.add(task_id)
            source_tasks.append(task_lookup[task_id])

    scene_episode = {
        "scene_episode_id": f"{split}/{scene}/stitched_0",
        "split": split,
        "scene": scene,
        "robot": source_tasks[0].get("robot"),
        "start_position": source_tasks[0].get("start_position"),
        "start_yaw": source_tasks[0].get("start_yaw"),
        "scene_level": True,
        "unordered_tasks": True,
        "unordered_targets": True,
        "time_unit": "step",
        "num_source_tasks": len(source_tasks),
        "num_targets": len(selected_targets),
        "source_task_ids": [task["task_id"] for task in source_tasks],
        "instructions": [task.get("instruction") for task in source_tasks],
        "stitched_instruction": "\n".join(
            f"Task {index + 1}: {task.get('instruction')}"
            for index, task in enumerate(source_tasks)
        ),
        "targets": selected_targets,
        "oracle_time_proxy_ordered_sum": oracle_time_proxy,
        "oracle_time_source": oracle_time_source,
        "time_budgets": budgets,
        "budget_ratios": budget_ratios,
        "coverage_target": coverage,
        "source_total_tasks_in_scene": scene_task_count,
        "source_total_targets_in_scene": sum(len(task.get("targets", [])) for task in scene_records),
        "source_target_coverage_in_scene": (
            len(selected_targets) / sum(len(task.get("targets", [])) for task in scene_records)
            if scene_records else 0.0
        ),
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
    parser.add_argument("--min-targets", type=int, default=4)
    parser.add_argument("--max-targets", type=int, default=8)
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
        selected_targets = choose_scene_targets(
            scene_records,
            min_targets=args.min_targets,
            max_targets=args.max_targets,
            coverage=args.coverage,
        )
        if not selected_targets:
            skipped[scene] = len(scene_records)
            continue
        scene_episodes.append(
            build_scene_episode(
                args.split,
                scene,
                scene_records,
                selected_targets,
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
    available_targets = sum(
        len(task.get("targets", []))
        for items in by_scene.values()
        for task in items
    )
    eligible_targets = sum(
        sum(len(task.get("targets", [])) for task in items)
        for items in by_scene.values()
        if sum(len(task.get("targets", [])) for task in items) >= args.min_targets
    )
    covered_targets = sum(target_counts)
    source_scene_counts = [len(items) for items in by_scene.values()]
    source_target_counts = [
        sum(len(task.get("targets", [])) for task in items)
        for items in by_scene.values()
    ]

    print("Scene-level time-aware data")
    print(f"input: {args.input}")
    print(f"split: {args.split}")
    print(f"source scenes: {len(by_scene)}")
    print(f"source tasks: {available_tasks}")
    print(f"source tasks per scene: {describe(source_scene_counts)}")
    print(f"source targets: {available_targets}")
    print(f"source targets per scene: {describe(source_target_counts)}")
    print(f"stitched episodes: {len(scene_episodes)}")
    print(f"skipped scenes (<{args.min_targets} targets): {len(skipped)}")
    print(f"covered source targets among all source targets: {covered_targets}/{available_targets} ({covered_targets / available_targets:.2%})")
    print(f"covered source targets among eligible scenes: {covered_targets}/{eligible_targets} ({covered_targets / eligible_targets:.2%})")
    print(f"source tasks per stitched episode: {describe(task_counts)}")
    print(f"targets per stitched episode: {describe(target_counts)}")
    print(f"oracle_time_proxy_ordered_sum: {describe(oracle_times)}")
    for ratio in budget_ratios:
        key = str(ratio)
        values = [record["time_budgets"][key] for record in scene_episodes]
        print(f"budget_ratio_{key}: {describe(values)}")
    print("selected target count distribution:", dict(sorted(Counter(target_counts).items())))
    print(f"wrote: {output}")

    if scene_episodes:
        print("\nsample:")
        print(json.dumps(scene_episodes[0], ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
