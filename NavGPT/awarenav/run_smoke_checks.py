# 用途：独立验证 Habitat RGB、策略动作解析、环境侧自动目标完成和 step budget 截断，不运行真实模型。
"""Evaluator-only checks for the visual-policy protocol.

The completion probe intentionally places the agent at one hidden evaluation
target to prove that completion is automatic.  This is a test fixture only and
is explicitly reported as such; no policy payload receives that coordinate.
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import habitat_sim
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from habitat_base.time_aware_simulation import TimeAwareSceneSimulator
from tools.awarenav.habitat_adapter import (
    environment_auto_complete,
    make_policy_payload,
    parse_policy_action,
)
from tools.run_time_aware_awarenav import get_budget
from tools.run_time_aware_greedy import load_records, record_to_config


def make_sim(args, record, budget):
    sim_args = SimpleNamespace(
        scene=args.scene,
        scene_dataset=args.scene_dataset,
        success_dis=args.success_dis,
        max_step=budget,
        no_render=False,
    )
    return TimeAwareSceneSimulator(
        sim_args,
        record_to_config(record),
        time_budget=budget,
        target_values=[target.get("value", 1.0) for target in record["targets"]],
    )


def check_payload(record, sim):
    payload = make_policy_payload(record, sim, [], include_time_prompt=True)
    prompt = payload["prompt"]
    forbidden = ("time_budget", "time_used", "time_remaining", "geo dis", "target coord", "oracle")
    if any(token in prompt.lower() for token in forbidden):
        raise AssertionError("Policy prompt includes evaluator-only field name.")
    first_position = str(record["targets"][0]["target_position"])
    if first_position in prompt:
        raise AssertionError("Policy prompt exposes a target coordinate.")
    if "Time condition: time is " not in prompt:
        raise AssertionError("Dynamic fuzzy time prompt is absent.")
    if not payload["rgb_jpeg_base64"] or len(payload["rgb_shape"]) != 3:
        raise AssertionError("Current RGB was not encoded for policy.")
    return {
        "rgb_shape": payload["rgb_shape"],
        "allowed_actions": payload["allowed_actions"],
        "fuzzy_time_present": True,
        "evaluator_fields_absent_from_prompt": True,
    }


def completion_probe(record, args, budget):
    sim = make_sim(args, record, budget)
    try:
        sim.actor("stop")
        target = np.asarray(record["targets"][0]["target_position"], dtype=np.float32)
        state = habitat_sim.AgentState()
        state.position = sim.pathfinder.snap_point(target)
        state.rotation = sim.agent.get_state().rotation
        sim.agent.set_state(state)
        sim.observations = sim.sim.get_sensor_observations()
        completed = environment_auto_complete(sim)
        if 0 not in completed or 0 not in sim.completed_targets:
            raise AssertionError("Environment completion probe did not complete target 0.")
        return {
            "completion_probe_uses_hidden_evaluator_target_coordinate_only": True,
            "auto_completed_targets": completed,
            "time_used_after_auto_completion": sim.time_used,
            "policy_stop_was_not_used": True,
        }
    finally:
        sim.close()


def budget_probe(record, args, budget):
    sim = make_sim(args, record, budget)
    try:
        sim.actor("stop")
        while not sim.episode_over:
            sim.actor("turn_left")
            environment_auto_complete(sim)
        if sim.time_used != budget:
            raise AssertionError(f"Budget did not truncate exactly: {sim.time_used} != {budget}")
        return {"time_budget": budget, "time_used_at_cutoff": sim.time_used, "budget_truncates_exactly": True}
    finally:
        sim.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default="/file_system/nas/algorithm/Intern03/data/time_aware_scene/all_episodes_spatial_start_floor_oracle_time.jsonl")
    parser.add_argument("--budget-ratio", type=float, default=0.5)
    parser.add_argument("--scene", default="data/hm3d/")
    parser.add_argument("--scene-dataset", default="data/hm3d/hm3d_annotated_basis.scene_dataset_config.json")
    parser.add_argument("--success-dis", type=float, default=1.0)
    parser.add_argument("--output", default="output/time_aware_scene/awarenav/smoke_checks.json")
    args = parser.parse_args()
    record = load_records(args.episodes, "all", 1)[0]
    budget = get_budget(record, args.budget_ratio)
    sim = make_sim(args, record, budget)
    try:
        sim.actor("stop")
        payload = check_payload(record, sim)
    finally:
        sim.close()
    action_parse = {
        "json_action": parse_policy_action('{"action": "turn_left"}'),
        "plain_action": parse_policy_action("move_forward"),
        "invalid_action": parse_policy_action('{"action": "stop"}'),
    }
    if action_parse != {"json_action": "turn_left", "plain_action": "move_forward", "invalid_action": None}:
        raise AssertionError(f"Action parser check failed: {action_parse}")
    report = {
        "episode_id": record.get("scene_episode_id") or record.get("episode_id"),
        "payload": payload,
        "action_parse": action_parse,
        "environment_auto_completion": completion_probe(record, args, budget),
        "budget": budget_probe(record, args, budget),
        "model_called": False,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
