#!/usr/bin/env python3
"""Export scene-level Time-Aware VLN episodes to a VLN-CE/NaVid-style file.

This exporter is only an interface bridge. The final benchmark metrics should
still be computed with the time-aware multi-target evaluator, because standard
VLN-CE episodes contain one goal while this project has unordered multi-target
completion.
"""

import argparse
import gzip
import json
import math
from pathlib import Path


TIME_TEXT = {
    "0.5": "Time condition: time is very limited. You are in a hurry. Prioritize quick useful progress.",
    "1.0": "Time condition: time is moderate. Balance completing targets with avoiding wasted exploration.",
    "1.5": "Time condition: time is sufficient. Try to complete all targets carefully.",
}


def read_jsonl(path, limit=0):
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit and len(records) >= limit:
                break
    return records


def yaw_to_quat_xyzw(yaw_degrees):
    yaw = math.radians(float(yaw_degrees or 0.0))
    return [0.0, math.sin(yaw / 2.0), 0.0, math.cos(yaw / 2.0)]


def ratio_name(ratio):
    return str(ratio).replace(".", "p")


def scene_path(record, repo_root):
    path = Path(record["scene_path"])
    if path.is_absolute():
        return str(path)
    return str((repo_root / path).resolve())


def target_summary(display_index, target):
    names = [target.get("name", f"target_{target.get('index', '?')}")]
    for occurrence in target.get("source_occurrences", []):
        name = occurrence.get("name")
        if name and name not in names:
            names.append(name)
    region = target.get("region_name") or target.get("region") or "unknown area"
    return f"target {display_index}: {' / '.join(names)} in {region}"


def build_instruction(record, ratio):
    ratio_key = str(ratio)
    time_text = TIME_TEXT.get(ratio_key, "Time condition: time is limited. Navigate efficiently.")
    target_lines = "\n".join(
        target_summary(index, target)
        for index, target in enumerate(record["targets"])
    )
    return (
        f"{time_text}\n"
        "Complete as many of the following unordered targets as possible. "
        "The targets can be completed in any order.\n"
        f"{target_lines}\n"
        "Original task context:\n"
        f"{record.get('stitched_instruction') or record.get('instruction') or ''}"
    )


def final_goal_position(record):
    order = record.get("oracle_optimal_order") or list(range(len(record["targets"])))
    if not order:
        return record["start_position"]
    target = record["targets"][order[-1]]
    return target.get("target_position") or target.get("position") or record["start_position"]


def reference_path(record):
    order = record.get("oracle_optimal_order") or list(range(len(record["targets"])))
    path = [record["start_position"]]
    for index in order:
        target = record["targets"][index]
        position = target.get("target_position") or target.get("position")
        if position is not None:
            path.append(position)
    return path


def make_episode(record, ratio, index, repo_root):
    ratio_key = str(ratio)
    episode_id = f"{index:06d}_{ratio_name(ratio)}"
    goal_position = final_goal_position(record)
    instruction = build_instruction(record, ratio)
    return {
        "episode_id": episode_id,
        "scene_id": scene_path(record, repo_root),
        "start_position": record["start_position"],
        "start_rotation": yaw_to_quat_xyzw(record.get("start_yaw", 0.0)),
        "goals": [{"position": goal_position, "radius": record.get("success_distance", 1.0)}],
        "reference_path": reference_path(record),
        "instruction": {
            "instruction_text": instruction,
            "instruction_id": episode_id,
            "language": "en",
            "split": record.get("split", "all"),
        },
        "trajectory_id": episode_id,
        "source_time_aware_episode_id": record["episode_id"],
        "source_scene": record["scene"],
        "budget_ratio": float(ratio),
        "time_budget": record["time_budgets"][ratio_key],
        "oracle_optimal_time": record.get("oracle_optimal_time"),
        "target_positions": [
            target.get("target_position") or target.get("position")
            for target in record["targets"]
        ],
        "target_names": [target.get("name", f"target_{i}") for i, target in enumerate(record["targets"])],
        "target_values": [target.get("value", 1.0) for target in record["targets"]],
    }


def write_json_gz(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


def write_navid_configs(output_dir, ratio, data_path, gt_path):
    ratio_slug = ratio_name(ratio)
    task_config = output_dir / f"vlnce_task_timeaware_{ratio_slug}.yaml"
    exp_config = output_dir / f"uninavid_timeaware_{ratio_slug}.yaml"
    task_config.write_text(
        f"""ENVIRONMENT:
  MAX_EPISODE_STEPS: 500
SIMULATOR:
  ACTION_SPACE_CONFIG: v0
  AGENT_0:
    SENSORS: [RGB_SENSOR]
  FORWARD_STEP_SIZE: 0.25
  TURN_ANGLE: 30
  HABITAT_SIM_V0:
    GPU_DEVICE_ID: 0
    ALLOW_SLIDING: True
  RGB_SENSOR:
    WIDTH: 640
    HEIGHT: 480
    HFOV: 120
    TYPE: HabitatSimRGBSensor
    POSITION: [0, 1.25, 0]

TASK:
  TYPE: VLN-v0
  SUCCESS_DISTANCE: 3.0
  SENSORS: [
    INSTRUCTION_SENSOR,
    SHORTEST_PATH_SENSOR,
    VLN_ORACLE_PROGRESS_SENSOR
  ]
  INSTRUCTION_SENSOR_UUID: instruction
  POSSIBLE_ACTIONS: [STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT]
  MEASUREMENTS: [
    TOP_DOWN_MAP_VLNCE,
    DISTANCE_TO_GOAL,
    SUCCESS,
    SPL,
    NDTW,
    PATH_LENGTH,
    ORACLE_SUCCESS,
    STEPS_TAKEN
  ]
  SUCCESS:
    SUCCESS_DISTANCE: 3.0
  SPL:
    SUCCESS_DISTANCE: 3.0
  NDTW:
    GT_PATH: {gt_path}
DATASET:
  TYPE: VLN-CE-v1
  SPLIT: all
  DATA_PATH: {data_path}
  SCENES_DIR: /
""",
        encoding="utf-8",
    )
    exp_config.write_text(
        f"""BASE_TASK_CONFIG_PATH: {task_config}
EVAL:
    IDENTIFICATION: timeaware_{ratio_slug}
    SPLIT: all
    EARLY_STOP_ROTATION: 25
    EARLY_STOP_STEPS: 500
""",
        encoding="utf-8",
    )
    return task_config, exp_config


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes",
        default="/file_system/nas/algorithm/Intern03/data/time_aware_scene/all_episodes_spatial_start_floor_oracle_time.jsonl",
    )
    parser.add_argument("--output-dir", default="/file_system/nas/algorithm/Intern03/data/time_aware_scene/vlnce_navid")
    parser.add_argument("--budget-ratios", default="0.5,1.0,1.5")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    return parser


def main():
    args = build_parser().parse_args()
    repo_root = Path(args.repo_root).resolve()
    records = read_jsonl(args.episodes, args.limit)
    output_dir = Path(args.output_dir)
    vocab = {"word_list": ["<PAD>", "<UNK>"]}
    summary = []
    for ratio_text in args.budget_ratios.split(","):
        ratio_text = ratio_text.strip()
        if not ratio_text:
            continue
        ratio = float(ratio_text)
        episodes = [
            make_episode(record, ratio, index, repo_root)
            for index, record in enumerate(records)
            if str(ratio) in record.get("time_budgets", {})
        ]
        payload = {
            "instruction_vocab": vocab,
            "episodes": episodes,
        }
        name = f"all_timeaware_budget_{ratio_name(ratio)}.json.gz"
        out_path = output_dir / name
        write_json_gz(out_path, payload)

        gt = {
            episode["episode_id"]: {"locations": episode["reference_path"]}
            for episode in episodes
        }
        gt_path = output_dir / f"all_timeaware_budget_{ratio_name(ratio)}_gt.json.gz"
        write_json_gz(gt_path, gt)
        task_config, exp_config = write_navid_configs(output_dir, ratio, out_path, gt_path)
        summary.append(
            {
                "budget_ratio": ratio,
                "episodes": len(episodes),
                "data_path": str(out_path),
                "gt_path": str(gt_path),
                "task_config": str(task_config),
                "exp_config": str(exp_config),
            }
        )

    summary_path = output_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
