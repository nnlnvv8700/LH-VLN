# 用途：通过 RGB+文本动作协议运行 AwareNav 候选视觉策略，并由 Habitat 环境自动判定无序目标完成。
"""Run a policy through the time-aware Habitat evaluator.

This runner is intentionally model-agnostic because the public repository named
AwareNav has no visual-navigation model contract.  A future verified AwareNav
bridge can be supplied with --policy-command: it receives one JSON payload on
stdin (prompt, current RGB JPEG base64 and allowed primitive actions) and must
emit {"action": "move_forward|turn_left|turn_right"}.  No target coordinate,
geodesic distance, global map, step count or numeric budget enters that payload.
"""

import argparse
import contextlib
import csv
import hashlib
import io
import json
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from habitat_base.time_aware_simulation import TimeAwareSceneSimulator
from tools.awarenav.habitat_adapter import (
    POLICY_ACTIONS,
    environment_auto_complete,
    make_policy_payload,
    parse_policy_action,
    save_rgb,
)
from tools.run_time_aware_greedy import (
    load_records,
    parse_budget_ratios,
    record_id,
    record_to_config,
    summarize,
)


def ratio_name(ratio):
    return str(ratio).replace(".", "p")


def get_budget(record, ratio):
    value = record.get("time_budgets", {}).get(str(ratio))
    if value is None:
        raise ValueError(f"Missing time budget {ratio} for {record_id(record)}")
    return int(value)


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


def select_smoke_action(args, episode_id, decision):
    if args.smoke_policy == "cycle":
        return POLICY_ACTIONS[decision % len(POLICY_ACTIONS)]
    digest = hashlib.sha256(f"{args.seed}:{episode_id}:{decision}".encode()).digest()
    return POLICY_ACTIONS[digest[0] % len(POLICY_ACTIONS)]


def call_policy(args, payload, episode_id, decision):
    if args.policy_command:
        try:
            process = subprocess.run(
                args.policy_command,
                input=json.dumps(payload, ensure_ascii=False),
                text=True,
                capture_output=True,
                timeout=args.policy_timeout,
                check=False,
                shell=True,
            )
        except subprocess.TimeoutExpired:
            return None, "policy_timeout"
        raw = process.stdout.strip()
        if process.returncode != 0:
            return None, f"policy_nonzero_exit:{process.returncode}:{process.stderr.strip()}"
        return parse_policy_action(raw), raw
    action = select_smoke_action(args, episode_id, decision)
    raw = json.dumps({"action": action})
    return parse_policy_action(raw), raw


def run_episode(args, record):
    budget = get_budget(record, args.budget_ratio)
    sim = make_sim(args, record, budget)
    trace, history = [], []
    episode_id = record_id(record)
    rgb_dir = Path(args.output_dir) / "rgb" / f"budget_{ratio_name(args.budget_ratio)}" / episode_id.replace("/", "_")
    parse_failures = 0
    try:
        # Initialize the inherited simulator without consuming a control step.
        sim.actor("stop")
        initial_auto = environment_auto_complete(sim)
        decision = 0
        while not sim.episode_over and sim.time_used < budget and decision < args.max_decisions:
            payload = make_policy_payload(record, sim, history, args.with_time_prompt)
            rgb_path = None
            if args.save_rgb and (decision < args.max_saved_rgb):
                rgb_path = save_rgb(sim.observations, rgb_dir / f"observation_{decision:04d}.jpg")
            action, raw = call_policy(args, payload, episode_id, decision)
            event = "policy_action"
            if action not in POLICY_ACTIONS:
                parse_failures += 1
                action = select_smoke_action(args, episode_id, decision)
                event = "invalid_policy_action_fallback"
            sim.actor(action)
            auto_completed = environment_auto_complete(sim)
            history.append(f"- action taken: {action}; environment completed: {len(auto_completed)} target(s)")
            trace.append(
                {
                    "action": action,
                    "event": event,
                    "raw_policy_response": raw,
                    "environment_auto_completed_targets": auto_completed,
                    "time_used_evaluator_only": sim.time_used,
                    "rgb_debug_path": rgb_path,
                }
            )
            decision += 1
        result = sim.return_results()
    finally:
        sim.close()
    result.update(
        {
            "task_id": episode_id,
            "split": record["split"],
            "scene": record["scene"],
            "baseline": args.baseline_name,
            "policy_mode": "external_visual_policy" if args.policy_command else f"smoke_{args.smoke_policy}",
            "policy_role": "low_level_visual_navigation",
            "time_prompt_mode": "dynamic_fuzzy" if args.with_time_prompt else "none",
            "target_completion_mode": "environment_side_distance_auto_completion",
            "initial_environment_auto_completed_targets": initial_auto,
            "policy_parse_failures": parse_failures,
            "awarenav_trace": trace,
        }
    )
    return result


def write_outputs(output_dir, split, ratio, results, args):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(results)
    summary.update(
        {
            "split": split,
            "budget_ratio": ratio,
            "baseline": args.baseline_name,
            "policy_mode": "external_visual_policy" if args.policy_command else f"smoke_{args.smoke_policy}",
            "policy_role": "low_level_visual_navigation",
            "time_prompt_mode": "dynamic_fuzzy" if args.with_time_prompt else "none",
            "target_completion_mode": "environment_side_distance_auto_completion",
            "policy_parse_failures": sum(item["policy_parse_failures"] for item in results),
        }
    )
    stem = f"awarenav_{split}_budget_{ratio_name(ratio)}"
    with (output_dir / f"{stem}.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    with (output_dir / f"{stem}.json").open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "results": results}, handle, ensure_ascii=False, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default="/file_system/nas/algorithm/Intern03/data/time_aware_scene/all_episodes_spatial_start_floor_oracle_time.jsonl")
    parser.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--budget-ratio", type=float, default=None)
    parser.add_argument("--budget-ratios", default="0.5,1.0,1.5")
    parser.add_argument("--policy-command", default=None)
    parser.add_argument("--policy-timeout", type=float, default=120.0)
    parser.add_argument("--smoke-policy", choices=["cycle", "seeded_random"], default="cycle")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--with-time-prompt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-decisions", type=int, default=10000)
    parser.add_argument("--scene", default="data/hm3d/")
    parser.add_argument("--scene-dataset", default="data/hm3d/hm3d_annotated_basis.scene_dataset_config.json")
    parser.add_argument("--success-dis", type=float, default=1.0)
    parser.add_argument("--output-dir", default="output/time_aware_scene/awarenav")
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--save-rgb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-saved-rgb", type=int, default=3)
    parser.add_argument("--baseline-name", default="awarenav_habitat_adapter")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    records = load_records(args.episodes, args.split, args.limit if args.limit > 0 else None)
    ratios = [args.budget_ratio] if args.budget_ratio is not None else parse_budget_ratios(args.budget_ratios)
    summaries = []
    for ratio in ratios:
        args.budget_ratio = ratio
        results = []
        for record in records:
            if args.quiet:
                with contextlib.redirect_stdout(io.StringIO()):
                    results.append(run_episode(args, record))
            else:
                results.append(run_episode(args, record))
        summaries.append(write_outputs(args.output_dir, args.split, ratio, results, args))
    csv_path = Path(args.summary_csv) if args.summary_csv else Path(args.output_dir) / f"awarenav_{args.split}_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
