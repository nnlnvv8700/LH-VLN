#!/usr/bin/env python3
"""Run a NavGPT-style action-level agent on scene-level Time-Aware VLN.

This is an adapter rather than the unmodified official NavGPT runner: official
NavGPT is written for R2R/Matterport viewpoint graphs, while this benchmark runs
in Habitat/HM3D continuous-control scenes. The adapter preserves the NavGPT
interaction style: history + observation + action_maker output format.
"""

import argparse
import contextlib
import csv
import io
import json
import math
import re
import subprocess
import sys
from collections import Counter
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
    record_instruction,
    record_to_config,
    summarize,
    write_json,
    write_summary_csv,
)


VALID_ACTIONS = {"move_forward", "turn_left", "turn_right", "stop"}
REPEATABLE_ACTIONS = {"move_forward", "turn_left", "turn_right"}
SEMANTIC_IGNORE = {
    "",
    "unknown",
    "wall",
    "floor",
    "ceiling",
    "window",
    "door",
    "frame",
    "beam",
    "curtain",
    "pillow",
    "decoration",
    "sheet",
    "stairs",
}

STOP_HINT_TEMPLATE = (
    "Target completion hint: {hint} If this seems like the intended target, "
    "choose stop / Final Answer: Finished so the system can verify it."
)
VISUAL_MATCH_TEMPLATE = (
    "Visual target cue: {hint}. This only means the object/category is visible; "
    "it may still be too far away or in the wrong room. Move closer or align the view "
    "before stopping unless it is clearly adjacent."
)


FUZZY_BY_RATIO = {
    "0.5": "Overall time condition: very limited.",
    "1.0": "Overall time condition: moderate.",
    "1.5": "Overall time condition: sufficient.",
}


def dynamic_time_condition(sim):
    if sim.time_budget <= 0:
        return "Time is exhausted. Stop only if you can complete a target."
    fraction = sim.time_remaining / sim.time_budget
    if fraction <= 0.25:
        return "Time is almost exhausted. You are in a hurry. Stop only near a likely target and avoid detours."
    if fraction <= 0.50:
        return "Time is tight. Prioritize quick useful progress and avoid unnecessary turning."
    if fraction <= 0.80:
        return "Time is moderate. Continue making progress while keeping the remaining targets in mind."
    return "Time is relatively sufficient. Explore efficiently and try to complete all targets."


def time_condition(args, sim):
    overall = FUZZY_BY_RATIO.get(str(args.budget_ratio), "The overall time budget is limited.")
    if args.time_prompt_mode == "ratio_fuzzy":
        return overall
    return f"Overall condition: {overall} Current urgency: {dynamic_time_condition(sim)}"


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


def clamp_repeat(repeat, max_repeat):
    return max(1, min(int(repeat), int(max_repeat)))


def parse_action(raw_response, max_repeat):
    text = raw_response.strip()
    if re.search(r"Final Answer\s*:\s*Finished", text, flags=re.IGNORECASE):
        return "stop", 1
    match = re.search(r"Action\s*Input\s*:\s*\"?([^\"\n]+)\"?", text)
    if match:
        action_text = match.group(1).strip()
        action_match = re.search(r"\b(move_forward|turn_left|turn_right|stop)\b", action_text)
        action = action_match.group(1) if action_match else action_text
        if action in VALID_ACTIONS:
            repeat_match = re.search(r"(?:x|repeat|times|:)\s*(\d+)", action_text)
            repeat = int(repeat_match.group(1)) if repeat_match else 1
            if action not in REPEATABLE_ACTIONS:
                repeat = 1
            return action, clamp_repeat(repeat, max_repeat)
    direct_match = re.search(
        r"Action\s*:\s*\"?\s*(move_forward|turn_left|turn_right|stop)\s*\"?",
        text,
        flags=re.IGNORECASE,
    )
    if direct_match:
        action = direct_match.group(1).lower()
        return action, 1
    return None, 1


def target_line(record, sim, target_index, include_oracle_text):
    target = record["targets"][target_index]
    name = target.get("name", f"target_{target_index}")
    region = target.get("region_name") or target.get("region") or "unknown"
    if not include_oracle_text:
        return f"- target {target_index}: {name} in {region}"
    info = sim.get_target_info(target_index)
    distance = info["geo dis"]
    distance_text = "unreachable" if math.isinf(distance) else f"{distance:.2f}m shortest-path distance"
    return f"- target {target_index}: {name} in {region}, {distance_text}"


def semantic_id_to_name(sim):
    mapping = {}
    for obj in sim.sim.semantic_scene.objects:
        if obj is None or obj.category is None:
            continue
        name = obj.category.name().strip().lower()
        if not name:
            continue
        semantic_id = getattr(obj, "semantic_id", None)
        if semantic_id is not None:
            mapping[int(semantic_id)] = name
        match = re.search(r"(\d+)$", str(getattr(obj, "id", "")))
        if match:
            mapping[int(match.group(1))] = name
    return mapping


def normalize_name(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def names_overlap(left, right):
    left_norm = normalize_name(left)
    right_norm = normalize_name(right)
    if not left_norm or not right_norm:
        return False
    left_tokens = set(left_norm.split())
    right_tokens = set(right_norm.split())
    if left_norm in right_norm or right_norm in left_norm:
        return True
    return bool(left_tokens & right_tokens)


def visible_semantic_objects(sim, top_k):
    observations = sim.observations or {}
    semantic = observations.get("semantic_sensor")
    if semantic is None:
        return []
    semantic = np.asarray(semantic)
    ids, counts = np.unique(semantic.astype(np.int64), return_counts=True)
    id_counts = dict(zip(ids.tolist(), counts.tolist()))
    name_map = semantic_id_to_name(sim)
    name_counts = Counter()
    for semantic_id, count in id_counts.items():
        if semantic_id < 0:
            continue
        name = name_map.get(int(semantic_id))
        if name is None or name in SEMANTIC_IGNORE:
            continue
        if any(token in name for token in SEMANTIC_IGNORE if token):
            continue
        name_counts[name] += int(count)
    total = float(semantic.size) if semantic.size else 1.0
    return [
        {
            "name": name,
            "pixel_fraction": count / total,
        }
        for name, count in name_counts.most_common(top_k)
    ]


def visible_target_matches(record, sim, visible):
    matches = []
    for index in sorted(sim.remaining_targets):
        target = record["targets"][index]
        target_name = target.get("name", f"target_{index}")
        for item in visible:
            if names_overlap(target_name, item["name"]):
                matches.append(
                    {
                        "target_index": index,
                        "target_name": target_name,
                        "visible_name": item["name"],
                    }
                )
                break
    return matches


def proximity_target_matches(args, record, sim):
    if not args.proximity_hint:
        return []
    matches = []
    threshold = max(float(args.success_dis), float(args.proximity_hint_distance))
    for index in sorted(sim.remaining_targets):
        info = sim.get_target_info(index)
        distance = info["geo dis"]
        if not math.isinf(distance) and distance <= threshold:
            target = record["targets"][index]
            matches.append(
                {
                    "target_index": index,
                    "target_name": target.get("name", f"target_{index}"),
                }
            )
    return matches


def visual_observation_text(args, record, sim):
    visible = visible_semantic_objects(sim, args.semantic_top_k)
    lines = []
    if not visible:
        lines.append("Current visual observation: no recognizable semantic objects are visible from the front camera.")
    else:
        lines.append("Current visual observation from the front camera:")
        for item in visible:
            lines.append(f"- visible object/category: {item['name']} (approx area {item['pixel_fraction']:.3f})")
    visible_matches = visible_target_matches(record, sim, visible)
    if visible_matches:
        match_text = "; ".join(
            f"target {item['target_index']} ({item['target_name']}) may match visible {item['visible_name']}"
            for item in visible_matches
        )
        lines.append(VISUAL_MATCH_TEMPLATE.format(hint=match_text))
    return "\n".join(lines)


def build_observation(args, record, sim, last_feedback=None):
    completed = [
        record["targets"][index].get("name", str(index))
        for index in sim.completed_targets
    ]
    remaining_lines = [
        target_line(record, sim, index, args.observation_mode == "oracle_text")
        for index in sorted(sim.remaining_targets)
    ]
    position, _ = sim.return_state()
    include_oracle_text = args.observation_mode == "oracle_text"
    lines = [
        f"Current progress: completed {len(sim.completed_targets)}/{len(record['targets'])} targets.",
        f"Completed targets: {', '.join(completed) if completed else 'none'}.",
        "Remaining target descriptions:",
    ]
    if args.observation_mode != "blind":
        lines.extend(remaining_lines)
    else:
        lines.append("Target details are hidden in this setting.")
    proximity_matches = proximity_target_matches(args, record, sim)
    if proximity_matches:
        match_text = "; ".join(
            f"target {item['target_index']} ({item['target_name']}) may be nearby"
            for item in proximity_matches
        )
        lines.append(STOP_HINT_TEMPLATE.format(hint=match_text))
    if last_feedback:
        lines.append(f"Last action feedback: {last_feedback}")
    if args.observation_mode == "oracle_text":
        lines.insert(0, f"Current agent position: [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}].")
    elif args.observation_mode == "semantic_text":
        lines.insert(0, visual_observation_text(args, record, sim))
    else:
        lines.append("No target coordinates or distances are provided in this setting.")
    return "\n".join(lines)


def build_prompt(args, record, sim, history, last_feedback=None):
    time_text = time_condition(args, sim)
    history_text = "\n".join(history[-args.history_window:]) if history else "No previous actions."
    observation = build_observation(args, record, sim, last_feedback)
    return f"""You are NavGPT, an intelligent embodied agent navigating in an indoor environment.
At each step, you receive a task instruction, navigation history, current observation, and valid actions.
Think about progress, choose the next action, and use the exact NavGPT action format.

Time condition: {time_text}

Goal:
Complete as many unordered targets as possible before time runs out. The system will verify success when you choose stop.
The targets can be completed in any order. If a target is only visible, first move closer or align your view. Prefer stop only when the observation says a target may be nearby or after you are clearly adjacent to the object.
If the last action made little or no progress, avoid repeating the same movement blindly.
If a stop just failed, do not stop again for the same visual cue; move or turn to obtain a different view first.

Instruction:
{record_instruction(record)}

History:
{history_text}

Current Observation:
{observation}

Valid actions:
- move_forward: move ahead by one small step. You may repeat it as "move_forward x N".
- turn_left: rotate left. You may repeat it as "turn_left x N".
- turn_right: rotate right. You may repeat it as "turn_right x N".
- stop: attempt to complete a nearby target. If no target is nearby, this is a failed stop.

Repeat limit:
Use N between 1 and {args.max_action_repeat}. Prefer small N when uncertain.

Format:
Thought: your reasoning
Action: action_maker
Action Input: "move_forward x 1"

If you believe you are already at a target, use:
Thought: your reasoning
Final Answer: Finished!

Thought:"""


def choose_action(args, prompt):
    try:
        completed = subprocess.run(
            args.navgpt_command,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=args.llm_timeout,
            check=False,
            shell=True,
        )
    except subprocess.TimeoutExpired:
        return None, 1, f"NavGPT command timed out after {args.llm_timeout}s", True

    raw = completed.stdout.strip()
    if completed.stderr.strip():
        raw = f"{raw}\n[stderr]\n{completed.stderr.strip()}".strip()
    command_failed = completed.returncode != 0
    if command_failed:
        raw = f"{raw}\n[returncode]\n{completed.returncode}".strip()
        return None, 1, raw, True
    action, repeat = parse_action(completed.stdout, args.max_action_repeat)
    return action, repeat, raw, False


def execute_action(sim, action, repeat):
    executed = 0
    if action == "stop":
        sim.actor("stop")
        return 1
    for _ in range(repeat):
        if sim.episode_over:
            break
        sim.actor(action)
        executed += 1
    return executed


def position_distance(before, after):
    before_arr = np.asarray(before, dtype=np.float32)
    after_arr = np.asarray(after, dtype=np.float32)
    return float(np.linalg.norm(after_arr - before_arr))


def movement_feedback(action, requested_repeat, executed, before_pos, after_pos, completed_before, completed_after):
    moved = position_distance(before_pos, after_pos)
    completed_delta = completed_after - completed_before
    if action == "stop":
        if completed_delta > 0:
            return "stop succeeded and completed a target."
        return "stop did not complete a target."
    if action in {"turn_left", "turn_right"}:
        if executed <= 0:
            return "the requested turn could not be executed."
        if requested_repeat > 1:
            return f"{action} was repeated {executed} time(s) and changed viewing direction."
        return f"{action} changed viewing direction."
    if executed <= 0:
        return "the requested action could not be executed."
    if moved < 0.03:
        return (
            f"{action} was executed but made little or no position change; "
            "you may be blocked or facing an obstacle."
        )
    if requested_repeat > 1:
        return f"{action} was repeated {executed} time(s) and changed position."
    return f"{action} changed position."


def run_episode(args, record):
    ratio_key = str(args.budget_ratio)
    time_budget = record["time_budgets"].get(ratio_key)
    if time_budget is None:
        raise ValueError(f"Missing budget ratio {ratio_key} for {record_id(record)}")

    sim = make_sim(args, record, time_budget)
    trace = []
    history = []
    last_feedback = None
    try:
        sim.actor("stop")
        while not sim.episode_over and sim.time_used < time_budget:
            prompt = build_prompt(args, record, sim, history, last_feedback)
            action, repeat, raw_response, command_failed = choose_action(args, prompt)
            if command_failed and args.invalid_action_policy == "end_episode":
                trace.append(
                    {
                        "time_used": sim.time_used,
                        "time_remaining": sim.time_remaining,
                        "action": None,
                        "requested_repeat": 0,
                        "executed_steps": 0,
                        "raw_response": raw_response,
                        "prompt": prompt if args.save_prompts else None,
                        "event": "navgpt_command_failed",
                    }
                )
                break
            if action is None:
                action = args.invalid_action_fallback
                repeat = 1
            before_pos, _ = sim.return_state()
            completed_before = len(sim.completed_targets)
            executed = execute_action(sim, action, repeat)
            after_pos, _ = sim.return_state()
            completed_after = len(sim.completed_targets)
            last_feedback = movement_feedback(
                action,
                repeat,
                executed,
                before_pos,
                after_pos,
                completed_before,
                completed_after,
            )
            history.append(
                f"Decision {len(history) + 1}: action={action}, requested_repeat={repeat}, executed_steps={executed}, "
                f"completed={len(sim.completed_targets)}/{len(record['targets'])}, feedback={last_feedback}"
            )
            trace.append(
                {
                    "time_used": sim.time_used,
                    "time_remaining": sim.time_remaining,
                    "action": action,
                    "requested_repeat": repeat,
                    "executed_steps": executed,
                    "movement_distance": position_distance(before_pos, after_pos),
                    "feedback": last_feedback,
                    "raw_response": raw_response,
                    "prompt": prompt if args.save_prompts else None,
                }
            )
            if len(trace) >= args.max_decisions:
                break
        result = sim.return_results()
    finally:
        sim.close()

    result["task_id"] = record_id(record)
    result["split"] = record["split"]
    result["scene"] = record["scene"]
    result["robot"] = record["robot"]
    result["baseline"] = "navgpt_habitat_adapter"
    result["observation_mode"] = args.observation_mode
    result["navgpt_trace"] = trace
    return result


def ratio_name(ratio):
    return str(ratio).replace(".", "p")


def write_summary_csv_with_name(path, rows):
    for row in rows:
        row["baseline"] = "navgpt_habitat_adapter"
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
    parser.add_argument(
        "--navgpt-command",
        required=True,
        help="External NavGPT-style command. It receives the prompt on stdin.",
    )
    parser.add_argument("--llm-timeout", type=float, default=60.0)
    parser.add_argument("--history-window", type=int, default=12)
    parser.add_argument("--max-decisions", type=int, default=400)
    parser.add_argument("--max-action-repeat", type=int, default=4)
    parser.add_argument("--time-prompt-mode", choices=["dynamic_fuzzy", "ratio_fuzzy"], default="dynamic_fuzzy")
    parser.add_argument("--observation-mode", choices=["oracle_text", "semantic_text", "blind"], default="semantic_text")
    parser.add_argument("--semantic-top-k", type=int, default=12)
    parser.add_argument(
        "--proximity-hint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add a non-numeric hint when the system believes a remaining target may be nearby.",
    )
    parser.add_argument(
        "--proximity-hint-distance",
        type=float,
        default=1.25,
        help="Geodesic threshold for the non-numeric nearby-target hint. The number is not exposed to the LLM.",
    )
    parser.add_argument("--invalid-action-fallback", choices=sorted(VALID_ACTIONS), default="turn_left")
    parser.add_argument(
        "--invalid-action-policy",
        choices=["fallback", "end_episode"],
        default="end_episode",
        help="What to do when the NavGPT command fails or returns no valid action.",
    )
    parser.add_argument("--scene", default="data/hm3d/")
    parser.add_argument(
        "--scene-dataset",
        default="data/hm3d/hm3d_annotated_basis.scene_dataset_config.json",
    )
    parser.add_argument("--success-dis", type=float, default=1.0)
    parser.add_argument("--no-render", action="store_true", default=True)
    parser.add_argument("--render", dest="no_render", action="store_false")
    parser.add_argument("--output-dir", default="output/time_aware_scene/navgpt")
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--save-prompts", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.observation_mode == "semantic_text" and args.no_render:
        raise ValueError("--observation-mode semantic_text requires --render so Habitat returns semantic observations.")

    records = load_records(args.episodes, args.split, args.limit)
    budget_ratios = [args.budget_ratio] if args.budget_ratio is not None else parse_budget_ratios(args.budget_ratios)
    summary_rows = []
    for ratio in budget_ratios:
        args.budget_ratio = ratio
        results = []
        print(f"\n===== NavGPT-Habitat budget_ratio={ratio} split={args.split} episodes={len(records)} =====")
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
        summary["observation_mode"] = args.observation_mode
        summary["time_prompt_mode"] = args.time_prompt_mode
        summary["baseline"] = "navgpt_habitat_adapter"
        print("\nsummary:")
        print(json.dumps(summary, ensure_ascii=False, indent=2))

        output = Path(args.output_dir) / f"navgpt_{args.split}_budget_{ratio_name(ratio)}.json"
        write_json(output, summary, results)
        summary_rows.append(summary)

    summary_csv = Path(args.summary_csv) if args.summary_csv else Path(args.output_dir) / f"navgpt_{args.split}_summary.csv"
    write_summary_csv_with_name(summary_csv, summary_rows)


if __name__ == "__main__":
    main()
