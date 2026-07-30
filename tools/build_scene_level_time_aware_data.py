# 用途：从原始导航任务构建同楼层、空间去重后的 scene-level 多目标 Time-Aware VLN episode 数据集。
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
import csv
import json
import math
import statistics
from copy import deepcopy
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


def round_position(position, precision):
    if position is None:
        return None
    return tuple(round(float(value) / precision) for value in position)


def normalize_text(value):
    return str(value or "").strip().lower()


def target_dedup_key(target, position_precision):
    rounded_position = round_position(target.get("target_position"), position_precision)
    if rounded_position is None:
        # Unknown-position targets are not safe to merge: two objects with the
        # same text label may live on different floors or in different places.
        return (
            "missing-position",
            target.get("source_task_id"),
            target.get("source_target_index"),
        )
    return (
        normalize_text(target.get("name")),
        normalize_text(target.get("region_name") or target.get("region")),
        rounded_position,
    )


def deduplicate_scene_targets(scene_targets, position_precision):
    """Merge repeated target points within the same scene.

    A target is considered duplicated only when it has the same object name,
    same region label, and nearly identical 3D target position, including the
    y coordinate. This means targets are merged only on the same floor. The first
    occurrence is kept, and all source occurrences are recorded for traceability.
    Targets without positions are kept separate.
    """
    unique_targets = []
    by_key = {}
    for target in scene_targets:
        key = target_dedup_key(target, position_precision)
        occurrence = {
            "source_task_id": target.get("source_task_id"),
            "source_target_index": target.get("source_target_index"),
            "source_task_instruction": target.get("source_task_instruction"),
            "source_ordered_gt_step": target.get("source_ordered_gt_step"),
        }
        if key not in by_key:
            item = deepcopy(target)
            item["dedup_key"] = {
                "name": normalize_text(target.get("name")),
                "region": normalize_text(target.get("region_name") or target.get("region")),
                "rounded_position": round_position(target.get("target_position"), position_precision),
                "position_precision": position_precision,
            }
            item["source_occurrences"] = [occurrence]
            item["duplicate_count"] = 1
            by_key[key] = item
            unique_targets.append(item)
        else:
            kept = by_key[key]
            kept["source_occurrences"].append(occurrence)
            kept["duplicate_count"] += 1
    return unique_targets


def source_occurrence(target):
    return {
        "source_task_id": target.get("source_task_id"),
        "source_target_index": target.get("source_target_index"),
        "source_task_instruction": target.get("source_task_instruction"),
        "source_ordered_gt_step": target.get("source_ordered_gt_step"),
        "name": target.get("name"),
        "region_name": target.get("region_name") or target.get("region"),
        "target_position": target.get("target_position"),
    }


def xz_distance(first, second):
    return math.sqrt(
        (float(first[0]) - float(second[0])) ** 2
        + (float(first[2]) - float(second[2])) ** 2
    )


def min_non_null(values, default=None):
    filtered = [value for value in values if value is not None]
    if not filtered:
        return default
    return min(filtered)


def deduplicate_spatial_targets(scene_targets, merge_distance, floor_threshold):
    """Merge nearby/overlapping target points into one spatial waypoint.

    The representative target_position is kept from a real source target instead
    of using the centroid, because centroids can fall outside the navmesh. The
    centroid is still recorded as metadata for visualization/debugging.
    """
    clusters = []
    for target in scene_targets:
        position = target.get("target_position")
        if position is None:
            item = deepcopy(target)
            item["target_type"] = "spatial_waypoint"
            item["source_occurrences"] = [source_occurrence(target)]
            item["duplicate_count"] = 1
            clusters.append(item)
            continue

        matched = None
        for cluster in clusters:
            cluster_position = cluster.get("target_position")
            if cluster_position is None:
                continue
            if abs(float(position[1]) - float(cluster_position[1])) > floor_threshold:
                continue
            if xz_distance(position, cluster_position) <= merge_distance:
                matched = cluster
                break

        occurrence = source_occurrence(target)
        if matched is None:
            item = deepcopy(target)
            item["target_type"] = "spatial_waypoint"
            item["spatial_merge_distance"] = merge_distance
            item["source_occurrences"] = [occurrence]
            item["duplicate_count"] = 1
            item["cluster_center"] = list(position)
            clusters.append(item)
        else:
            matched["source_occurrences"].append(occurrence)
            matched["duplicate_count"] += 1
            positions = [
                occurrence["target_position"]
                for occurrence in matched["source_occurrences"]
                if occurrence.get("target_position") is not None
            ]
            matched["cluster_center"] = [
                sum(float(position[index]) for position in positions) / len(positions)
                for index in range(3)
            ]
            matched["source_ordered_gt_step"] = min_non_null(
                occurrence.get("source_ordered_gt_step")
                for occurrence in matched["source_occurrences"]
            )
    for index, cluster in enumerate(clusters):
        cluster["spatial_cluster_id"] = index
        cluster["dedup_key"] = {
            "mode": "spatial",
            "representative_position": cluster.get("target_position"),
            "cluster_center": cluster.get("cluster_center"),
            "merge_distance": merge_distance,
            "floor_threshold": floor_threshold,
        }
    return clusters


def cluster_floor_heights(points, threshold):
    heights = sorted(float(point[1]) for point in points if point is not None)
    floors = []
    for height in heights:
        if not floors or abs(height - floors[-1][-1]) > threshold:
            floors.append([height])
        else:
            floors[-1].append(height)
    return [sum(group) / len(group) for group in floors]


def nearest_floor_index(point, floors):
    if point is None or not floors:
        return None
    height = float(point[1])
    return min(range(len(floors)), key=lambda index: abs(height - floors[index]))


def count_start_floor_targets(start_position, targets, floor_threshold):
    points = []
    if start_position is not None:
        points.append(start_position)
    points.extend(
        target["target_position"]
        for target in targets
        if target.get("target_position") is not None
    )
    floors = cluster_floor_heights(points, floor_threshold)
    start_floor = nearest_floor_index(start_position, floors)
    if start_floor is None:
        return 0
    return sum(
        nearest_floor_index(target.get("target_position"), floors) == start_floor
        for target in targets
        if target.get("target_position") is not None
    )


def floor_groups(scene_records, targets, floor_threshold):
    points = []
    for record in scene_records:
        if record.get("start_position") is not None:
            points.append(record["start_position"])
    points.extend(
        target["target_position"]
        for target in targets
        if target.get("target_position") is not None
    )
    floors = cluster_floor_heights(points, floor_threshold)
    targets_by_floor = defaultdict(list)
    for target in targets:
        floor_id = nearest_floor_index(target.get("target_position"), floors)
        targets_by_floor[floor_id].append(target)
    starts_by_floor = defaultdict(list)
    for record in sorted(scene_records, key=task_sort_key):
        floor_id = nearest_floor_index(record.get("start_position"), floors)
        starts_by_floor[floor_id].append(record)
    return floors, targets_by_floor, starts_by_floor


def deduplicate_targets(targets, mode, position_precision, spatial_merge_distance, floor_threshold):
    if mode == "none":
        scene_targets = [deepcopy(target) for target in targets]
        for target in scene_targets:
            target["source_occurrences"] = [source_occurrence(target)]
            target["duplicate_count"] = 1
        return scene_targets
    if mode == "semantic":
        return deduplicate_scene_targets(targets, position_precision)
    if mode == "spatial":
        return deduplicate_spatial_targets(targets, spatial_merge_distance, floor_threshold)
    raise ValueError(f"Unknown dedup mode: {mode}")


def choose_floor_id(
    floors,
    targets_by_floor,
    starts_by_floor,
    floor_selection,
    dedup_mode,
    position_precision,
    spatial_merge_distance,
    floor_threshold,
):
    candidate_floor_ids = sorted(
        floor_id for floor_id in targets_by_floor
        if floor_id is not None
    )
    if not candidate_floor_ids:
        return None
    if floor_selection in {"start", "best_with_start"}:
        candidate_floor_ids = [
            floor_id for floor_id in candidate_floor_ids
            if starts_by_floor.get(floor_id)
        ]
    if not candidate_floor_ids:
        return None
    if floor_selection == "start":
        return candidate_floor_ids[0]

    def score(floor_id):
        unique_targets = deduplicate_targets(
            targets_by_floor[floor_id],
            dedup_mode,
            position_precision,
            spatial_merge_distance,
            floor_threshold,
        )
        has_start = 1 if starts_by_floor.get(floor_id) else 0
        return len(unique_targets), has_start, -floor_id

    return max(candidate_floor_ids, key=score)


def scene_split(scene):
    return "train" if int(scene[2]) < 8 else "val"


def scene_asset_paths(scene, scene_root):
    split = scene_split(scene)
    scene_name = scene.split("-")[-1]
    scene_dir = Path(scene_root) / split / scene
    return {
        "scene_split": split,
        "scene_path": str(scene_dir / f"{scene_name}.basis.glb"),
        "navmesh_path": str(scene_dir / f"{scene_name}.basis.navmesh"),
    }


def annotate_floor_metadata(start_position, targets, floor_threshold):
    points = []
    if start_position is not None:
        points.append(start_position)
    points.extend(
        target["target_position"]
        for target in targets
        if target.get("target_position") is not None
    )
    floor_heights = cluster_floor_heights(points, floor_threshold)
    start_floor_id = nearest_floor_index(start_position, floor_heights)
    targets_on_start_floor = []
    targets_off_start_floor = []
    for target in targets:
        target["target_id"] = target.get("global_index")
        if target.get("target_position") is not None:
            target["position"] = target["target_position"]
            target_floor_id = nearest_floor_index(target["target_position"], floor_heights)
            target["floor_id"] = target_floor_id
            if target_floor_id == start_floor_id:
                targets_on_start_floor.append(target["global_index"])
            else:
                targets_off_start_floor.append(target["global_index"])
        else:
            target["floor_id"] = None
            targets_off_start_floor.append(target["global_index"])
    return {
        "floor_threshold": floor_threshold,
        "floor_heights": floor_heights,
        "start_floor_id": start_floor_id,
        "targets_on_start_floor": targets_on_start_floor,
        "targets_off_start_floor": targets_off_start_floor,
        "num_start_floor_targets": len(targets_on_start_floor),
        "num_off_start_floor_targets": len(targets_off_start_floor),
    }


def choose_scene_targets(
    records,
    min_targets,
    max_targets,
    coverage,
    dedup_mode,
    position_precision,
    include_missing_targets,
    spatial_merge_distance,
    floor_threshold,
    floor_selection,
):
    """Choose 4-8 target points for a scene.

    The goal is to cover about `coverage` of available targets while staying in the
    requested 4-8 target range. Scenes with fewer than min_targets are skipped.
    """
    raw_scene_targets = flatten_scene_targets(records)
    if not include_missing_targets:
        raw_scene_targets = [
            target for target in raw_scene_targets
            if target.get("target_position") is not None
        ]
    floors, targets_by_floor, starts_by_floor = floor_groups(
        records,
        raw_scene_targets,
        floor_threshold,
    )
    selected_floor_id = choose_floor_id(
        floors,
        targets_by_floor,
        starts_by_floor,
        floor_selection,
        dedup_mode,
        position_precision,
        spatial_merge_distance,
        floor_threshold,
    )
    if selected_floor_id is None:
        return [], None, None, floors
    floor_targets = targets_by_floor[selected_floor_id]
    scene_targets = deduplicate_targets(
        floor_targets,
        dedup_mode,
        position_precision,
        spatial_merge_distance,
        floor_threshold,
    )
    floor_start_records = starts_by_floor.get(selected_floor_id, [])
    default_start_task = floor_start_records[0] if floor_start_records else None
    if default_start_task is not None and default_start_task.get("start_position") is not None:
        start_y = float(default_start_task["start_position"][1])
        scene_targets = [
            target for target in scene_targets
            if target.get("target_position") is not None
            and abs(float(target["target_position"][1]) - start_y) <= floor_threshold
        ]
    if len(scene_targets) < min_targets:
        return [], selected_floor_id, None, floors
    target_count = int(math.ceil(len(scene_targets) * coverage))
    target_count = min(max_targets, max(min_targets, target_count))
    selected = scene_targets[:target_count]
    for index, target in enumerate(selected):
        target["global_index"] = index
    start_task = None
    selected_source_task_ids = set()
    for target in selected:
        for occurrence in target.get("source_occurrences", []):
            if occurrence.get("source_task_id"):
                selected_source_task_ids.add(occurrence["source_task_id"])
        if target.get("source_task_id"):
            selected_source_task_ids.add(target["source_task_id"])
    for record in floor_start_records:
        if record.get("task_id") in selected_source_task_ids:
            start_task = record
            break
    if start_task is None:
        start_task = default_start_task
    return selected, selected_floor_id, start_task, floors


def build_scene_episode(
    split,
    scene,
    scene_records,
    selected_targets,
    selected_floor_id,
    selected_floor_heights,
    start_task,
    scene_task_count,
    budget_ratios,
    coverage,
    oracle_time_source,
    scene_root,
    floor_threshold,
    success_distance,
    benchmark_setting,
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
        occurrence_task_ids = [
            occurrence.get("source_task_id")
            for occurrence in target.get("source_occurrences", [])
            if occurrence.get("source_task_id")
        ] or [target.get("source_task_id")]
        for task_id in occurrence_task_ids:
            if task_id in task_lookup and task_id not in seen_task_ids:
                seen_task_ids.add(task_id)
                source_tasks.append(task_lookup[task_id])
    if start_task is not None and start_task["task_id"] not in seen_task_ids:
        source_tasks.insert(0, start_task)
        seen_task_ids.add(start_task["task_id"])
    elif start_task is not None:
        source_tasks = [start_task] + [
            task for task in source_tasks
            if task["task_id"] != start_task["task_id"]
        ]
    start_position = source_tasks[0].get("start_position")
    floor_metadata = annotate_floor_metadata(
        start_position,
        selected_targets,
        floor_threshold,
    )
    asset_paths = scene_asset_paths(scene, scene_root)

    scene_episode = {
        "episode_id": f"{split}/{scene}/stitched_0",
        "scene_episode_id": f"{split}/{scene}/stitched_0",
        "split": split,
        "scene": scene,
        **asset_paths,
        "robot": source_tasks[0].get("robot"),
        "start_position": start_position,
        "start_yaw": source_tasks[0].get("start_yaw"),
        "scene_level": True,
        "unordered_tasks": True,
        "unordered_targets": True,
        "time_unit": "step",
        "num_source_tasks": len(source_tasks),
        "source_splits": sorted({task.get("split") for task in source_tasks if task.get("split")}),
        "num_targets": len(selected_targets),
        "source_task_ids": [task["task_id"] for task in source_tasks],
        "instructions": [task.get("instruction") for task in source_tasks],
        "stitched_instruction": "\n".join(
            f"Task {index + 1}: {task.get('instruction')}"
            for index, task in enumerate(source_tasks)
        ),
        "targets": selected_targets,
        **floor_metadata,
        "selected_floor_id": selected_floor_id,
        "selected_floor_heights": selected_floor_heights,
        "oracle_time_proxy_ordered_sum": oracle_time_proxy,
        "oracle_optimal_time": None,
        "oracle_optimal_order": None,
        "oracle_time_source": oracle_time_source,
        "time_budgets": budgets,
        "budget_ratios": budget_ratios,
        "budget_unit": "step",
        "success_distance": success_distance,
        "target_value_type": "uniform",
        "benchmark_setting": benchmark_setting,
        "dedup_mode": selected_targets[0].get("dedup_key", {}).get("mode", "semantic") if selected_targets else None,
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


def write_summary_csv(path, records, budget_ratios):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode_id",
        "split",
        "scene",
        "scene_split",
        "robot",
        "num_targets",
        "num_start_floor_targets",
        "num_off_start_floor_targets",
        "num_source_tasks",
        "source_splits",
        "oracle_time_source",
        "oracle_time_proxy_ordered_sum",
        "benchmark_setting",
        "dedup_mode",
    ]
    fieldnames.extend(f"budget_{ratio}" for ratio in budget_ratios)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {
                "episode_id": record["episode_id"],
                "split": record["split"],
                "scene": record["scene"],
                "scene_split": record["scene_split"],
                "robot": record["robot"],
                "num_targets": record["num_targets"],
                "num_start_floor_targets": record["num_start_floor_targets"],
                "num_off_start_floor_targets": len(record["targets_off_start_floor"]),
                "num_source_tasks": record["num_source_tasks"],
                "source_splits": ";".join(record.get("source_splits", [])),
                "oracle_time_source": record["oracle_time_source"],
                "oracle_time_proxy_ordered_sum": record["oracle_time_proxy_ordered_sum"],
                "benchmark_setting": record.get("benchmark_setting"),
                "dedup_mode": record.get("dedup_mode"),
            }
            for ratio in budget_ratios:
                row[f"budget_{ratio}"] = record["time_budgets"][str(ratio)]
            writer.writerow(row)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/time_aware/episodes.jsonl")
    parser.add_argument("--output", default="data/time_aware_scene/test_episodes.jsonl")
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--scene-root", default="data/hm3d")
    parser.add_argument("--min-targets", type=int, default=4)
    parser.add_argument("--max-targets", type=int, default=8)
    parser.add_argument("--coverage", type=float, default=0.8)
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable target-point deduplication inside each scene.",
    )
    parser.add_argument(
        "--dedup-mode",
        choices=["semantic", "spatial", "none"],
        default="semantic",
        help="How to merge repeated targets after floor filtering.",
    )
    parser.add_argument(
        "--include-missing-targets",
        action="store_true",
        help="Keep targets without target_position. By default they are filtered out.",
    )
    parser.add_argument(
        "--dedup-position-precision",
        type=float,
        default=0.25,
        help="Position grid size in meters for duplicate target matching.",
    )
    parser.add_argument(
        "--spatial-merge-distance",
        type=float,
        default=0.5,
        help="XZ distance in meters for merging nearby target points in spatial mode.",
    )
    parser.add_argument(
        "--floor-selection",
        choices=["start", "best_with_start", "best"],
        default="best_with_start",
        help="Which floor to keep before selecting targets.",
    )
    parser.add_argument(
        "--min-start-floor-targets",
        type=int,
        default=0,
        help="Skip scene episodes with fewer than this many targets on the start floor.",
    )
    parser.add_argument(
        "--floor-threshold",
        type=float,
        default=0.75,
        help="Y-distance threshold in meters for grouping floors.",
    )
    parser.add_argument("--budget-ratios", default="0.5,1.0,1.5")
    parser.add_argument("--success-distance", type=float, default=1.0)
    parser.add_argument("--benchmark-setting", default="start_floor")
    parser.add_argument(
        "--oracle-time-source",
        default="ordered_sum_proxy",
        help="Metadata only. Later replace with enumerated_simulated_optimal.",
    )
    args = parser.parse_args()
    dedup_mode = "none" if args.no_dedup else args.dedup_mode

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
    skipped_no_start_floor_targets = {}
    unique_targets_by_scene = {}
    raw_targets_by_scene = {}
    for scene, scene_records in sorted(by_scene.items()):
        raw_targets = flatten_scene_targets(scene_records)
        if not args.include_missing_targets:
            raw_targets = [
                target for target in raw_targets
                if target.get("target_position") is not None
            ]
        unique_targets = deduplicate_targets(
            raw_targets,
            dedup_mode,
            args.dedup_position_precision,
            args.spatial_merge_distance,
            args.floor_threshold,
        )
        raw_targets_by_scene[scene] = len(raw_targets)
        unique_targets_by_scene[scene] = len(unique_targets)
        selected_targets, selected_floor_id, start_task, floor_heights = choose_scene_targets(
            scene_records,
            min_targets=args.min_targets,
            max_targets=args.max_targets,
            coverage=args.coverage,
            dedup_mode=dedup_mode,
            position_precision=args.dedup_position_precision,
            include_missing_targets=args.include_missing_targets,
            spatial_merge_distance=args.spatial_merge_distance,
            floor_threshold=args.floor_threshold,
            floor_selection=args.floor_selection,
        )
        if not selected_targets:
            skipped[scene] = len(scene_records)
            continue
        start_floor_target_count = count_start_floor_targets(
            scene_records[0].get("start_position"),
            selected_targets,
            args.floor_threshold,
        )
        if start_floor_target_count < args.min_start_floor_targets:
            skipped_no_start_floor_targets[scene] = start_floor_target_count
            continue
        scene_episodes.append(
            build_scene_episode(
                args.split,
                scene,
                scene_records,
                selected_targets,
                selected_floor_id,
                floor_heights,
                start_task,
                len(scene_records),
                budget_ratios,
                args.coverage,
                args.oracle_time_source,
                args.scene_root,
                args.floor_threshold,
                args.success_distance,
                args.benchmark_setting,
            )
        )

    output = write_jsonl(args.output, scene_episodes)
    summary_output = None
    if args.summary_csv:
        summary_output = write_summary_csv(args.summary_csv, scene_episodes, budget_ratios)

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
    available_positioned_targets = sum(
        sum(target.get("target_position") is not None for target in task.get("targets", []))
        for items in by_scene.values()
        for task in items
    )
    available_unique_targets = sum(unique_targets_by_scene.values())
    eligible_targets = sum(
        unique_targets_by_scene[scene]
        for scene in by_scene
        if unique_targets_by_scene[scene] >= args.min_targets
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
    print(f"source targets with positions: {available_positioned_targets}")
    print(f"unique target points after dedup: {available_unique_targets}")
    print(f"include missing targets: {args.include_missing_targets}")
    dedup_source_targets = available_targets if args.include_missing_targets else available_positioned_targets
    print(f"duplicate target points removed: {dedup_source_targets - available_unique_targets}")
    print(f"dedup mode: {dedup_mode}")
    print(f"dedup position precision: {args.dedup_position_precision} m")
    print(f"spatial merge distance: {args.spatial_merge_distance} m")
    print(f"floor selection: {args.floor_selection}")
    print(f"source targets per scene: {describe(source_target_counts)}")
    print(f"stitched episodes: {len(scene_episodes)}")
    print(f"skipped scenes (<{args.min_targets} targets): {len(skipped)}")
    print(f"skipped scenes (<{args.min_start_floor_targets} start-floor targets): {len(skipped_no_start_floor_targets)}")
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
    if summary_output:
        print(f"wrote: {summary_output}")

    if scene_episodes:
        print("\nsample:")
        print(json.dumps(scene_episodes[0], ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
