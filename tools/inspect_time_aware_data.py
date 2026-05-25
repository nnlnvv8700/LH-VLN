#!/usr/bin/env python3
"""Inspect LHPR-VLN episodes and optionally export time-aware task records."""

import argparse
import gzip
import json
import statistics
from pathlib import Path


SPLITS = {
    "train": ["batch_1", "batch_2", "batch_3", "batch_4", "batch_5"],
    "val": ["batch_6"],
    "test": ["batch_7", "batch_8"],
}


def read_json_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def parse_region(region_text):
    if not isinstance(region_text, str):
        return None, None
    if ":" not in region_text:
        return None, region_text
    region_id, region_name = region_text.split(":", 1)
    return region_id.strip(), region_name.strip()


def normalize_targets(raw_targets, default_value):
    targets = []
    for index, item in enumerate(raw_targets or []):
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            name = item[0]
            region_text = item[1]
        else:
            name = item
            region_text = None
        region_id, region_name = parse_region(region_text)
        targets.append(
            {
                "index": index,
                "name": name,
                "region": region_text,
                "region_id": region_id,
                "region_name": region_name,
                "value": default_value,
            }
        )
    return targets


def get_episode_start_yaw(episode):
    st_tasks = episode.get("st_task") or []
    if not st_tasks:
        return None
    first_task = min(st_tasks, key=lambda item: item.get("start", 0))
    return first_task.get("start_yaw")


def load_trial_positions(task_dir, trajectory_path):
    task_path = task_dir / trajectory_path / "task.json"
    if not task_path.exists():
        return None
    with task_path.open("r", encoding="utf-8") as handle:
        task_data = json.load(handle)
    trial = task_data.get("trial", {}).get("trial_1")
    if trial is None:
        trial = next(iter(task_data.get("trial", {}).values()), None)
    if not trial:
        return None
    return trial.get("pos")


def load_all_trial_endpoints(task_dir, trajectory_path):
    task_path = task_dir / trajectory_path / "task.json"
    if not task_path.exists():
        return {}
    with task_path.open("r", encoding="utf-8") as handle:
        task_data = json.load(handle)
    endpoints = {}
    for trial_key, trial in (task_data.get("trial") or {}).items():
        if not trial_key.startswith("trial_"):
            continue
        try:
            trial_index = int(trial_key.split("_", 1)[1])
        except ValueError:
            continue
        positions = trial.get("pos") or []
        if positions:
            endpoints[trial_index] = positions[-1]
    return endpoints


def ordered_trial_position_lookup(episode, task_dir):
    st_tasks = episode.get("st_task") or []
    trajectory_path = None
    for step_task in st_tasks:
        if step_task.get("trajectory path"):
            trajectory_path = step_task.get("trajectory path")
            break
    if trajectory_path is None:
        return {}

    endpoints = load_all_trial_endpoints(task_dir, trajectory_path)
    if not endpoints:
        return {}

    lookup = {}
    for target_index, item in enumerate(episode.get("lh_task", {}).get("Object") or []):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        target_name = item[0]
        region_id, _ = parse_region(item[1])
        if region_id is None:
            continue
        region_number = region_id.replace("Region", "").strip()
        if target_index in endpoints:
            lookup[(target_name, str(region_number))] = endpoints[target_index]
    return lookup


def target_position_lookup(episode, task_dir):
    lookup = {}
    for step_task in episode.get("st_task") or []:
        trajectory_path = step_task.get("trajectory path")
        positions = load_trial_positions(task_dir, trajectory_path) if trajectory_path else None
        if not positions:
            continue

        end = step_task.get("end")
        if end is None:
            continue
        index = max(0, min(int(end) - 1, len(positions) - 1))
        end_position = positions[index]

        for target, region in zip(step_task.get("target", []), step_task.get("Region", [])):
            lookup[(target, str(region))] = end_position
    for key, position in ordered_trial_position_lookup(episode, task_dir).items():
        lookup.setdefault(key, position)
    return lookup


def make_record(split, batch, episode_key, episode, budget_ratios, default_value):
    task = episode["lh_task"]
    task_dir = Path("data")
    positions_by_target = target_position_lookup(episode, task_dir)
    gt_steps = task.get("gt_step") or []
    oracle_total_steps = int(sum(gt_steps)) if gt_steps else None
    budgets = {}
    if oracle_total_steps is not None:
        for ratio in budget_ratios:
            budgets[str(ratio)] = max(1, int(round(oracle_total_steps * ratio)))

    record = {
        "task_id": f"{batch}/{episode_key}",
        "split": split,
        "batch": batch,
        "scene": task.get("Scene"),
        "robot": task.get("Robot"),
        "instruction": task.get("Task instruction"),
        "start_position": task.get("Start pos"),
        "start_yaw": get_episode_start_yaw(episode),
        "unordered_targets": True,
        "time_unit": "step",
        "targets": normalize_targets(task.get("Object"), default_value),
        "gt_steps_ordered": gt_steps,
        "oracle_total_steps_ordered": oracle_total_steps,
        "time_budgets": budgets,
        "source_subtask_list": task.get("Subtask list"),
    }
    for target in record["targets"]:
        region_number = None
        if target["region_id"] is not None:
            region_number = target["region_id"].replace("Region", "").strip()
        target["target_position"] = positions_by_target.get((target["name"], str(region_number)))
    return record


def describe(values):
    if not values:
        return "n=0"
    values = sorted(values)
    return (
        f"n={len(values)} min={values[0]} "
        f"median={statistics.median(values):.1f} "
        f"mean={statistics.mean(values):.1f} max={values[-1]}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-dir", default="data/episode_task")
    parser.add_argument("--output", default=None, help="Optional JSONL export path.")
    parser.add_argument(
        "--budget-ratios",
        default="0.5,0.75,1.0,1.25",
        help="Comma-separated ratios of ordered oracle total steps.",
    )
    parser.add_argument("--default-target-value", type=float, default=1.0)
    args = parser.parse_args()

    episode_dir = Path(args.episode_dir)
    budget_ratios = [float(item) for item in args.budget_ratios.split(",") if item]

    records = []
    missing = []
    for split, batches in SPLITS.items():
        for batch in batches:
            path = episode_dir / f"{batch}.json.gz"
            if not path.exists():
                missing.append(str(path))
                continue
            data = read_json_gz(path)
            for episode_key, episode in data.items():
                records.append(
                    make_record(
                        split,
                        batch,
                        episode_key,
                        episode,
                        budget_ratios,
                        args.default_target_value,
                    )
                )

    target_counts = [len(record["targets"]) for record in records]
    target_position_counts = [
        sum(target.get("target_position") is not None for target in record["targets"])
        for record in records
    ]
    total_steps = [
        record["oracle_total_steps_ordered"]
        for record in records
        if record["oracle_total_steps_ordered"] is not None
    ]
    by_split = {}
    by_robot = {}
    for record in records:
        by_split[record["split"]] = by_split.get(record["split"], 0) + 1
        by_robot[record["robot"]] = by_robot.get(record["robot"], 0) + 1

    print("Time-aware data inspection")
    print(f"episode_dir: {episode_dir}")
    print(f"records: {len(records)}")
    print(f"missing_files: {missing if missing else 'none'}")
    print(f"splits: {by_split}")
    print(f"robots: {by_robot}")
    print(f"targets_per_task: {describe(target_counts)}")
    print(f"target_positions_per_task: {describe(target_position_counts)}")
    print(f"ordered_oracle_total_steps: {describe(total_steps)}")
    for ratio in budget_ratios:
        key = str(ratio)
        values = [record["time_budgets"][key] for record in records if key in record["time_budgets"]]
        print(f"budget_ratio_{key}: {describe(values)}")

    if records:
        sample = records[0]
        print("\nsample:")
        print(json.dumps(sample, ensure_ascii=False, indent=2)[:3000])

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"\nwrote: {output}")


if __name__ == "__main__":
    main()
