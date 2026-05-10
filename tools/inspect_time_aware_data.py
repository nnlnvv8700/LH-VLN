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


def make_record(split, batch, episode_key, episode, budget_ratios, default_value):
    task = episode["lh_task"]
    gt_steps = task.get("gt_step") or []
    oracle_total_steps = int(sum(gt_steps)) if gt_steps else None
    budgets = {}
    if oracle_total_steps is not None:
        for ratio in budget_ratios:
            budgets[str(ratio)] = max(1, int(round(oracle_total_steps * ratio)))

    return {
        "task_id": f"{batch}/{episode_key}",
        "split": split,
        "batch": batch,
        "scene": task.get("Scene"),
        "robot": task.get("Robot"),
        "instruction": task.get("Task instruction"),
        "unordered_targets": True,
        "time_unit": "step",
        "targets": normalize_targets(task.get("Object"), default_value),
        "gt_steps_ordered": gt_steps,
        "oracle_total_steps_ordered": oracle_total_steps,
        "time_budgets": budgets,
        "source_subtask_list": task.get("Subtask list"),
    }


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
