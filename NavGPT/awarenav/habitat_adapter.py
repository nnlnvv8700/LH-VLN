# 用途：提供不泄露 oracle 信息的 Habitat RGB 动作协议、模糊时间提示和环境侧多目标自动完成逻辑。
"""Shared execution primitives for a future visual-navigation policy.

The policy-facing payload contains RGB bytes, natural-language task state and
action history only.  Target positions, pathfinder/geodesic queries, budgets
and simulator state remain exclusively on the evaluator side.
"""

import base64
import io
import json
import re

import numpy as np
from PIL import Image


POLICY_ACTIONS = ("move_forward", "turn_left", "turn_right")


def fuzzy_time_state(time_used, time_budget, enabled=True):
    """Convert hidden numeric budget state into the only time signal to policy."""
    if not enabled:
        return ""
    fraction = max(0.0, min(1.0, (time_budget - time_used) / max(1, time_budget)))
    if fraction <= 0.15:
        level = "very limited"
    elif fraction <= 0.40:
        level = "tight"
    elif fraction <= 0.70:
        level = "moderate"
    else:
        level = "sufficient"
    return f"Time condition: time is {level}."


def target_text(record, indices):
    lines = []
    for index in sorted(indices):
        target = record["targets"][index]
        name = str(target.get("name") or "unnamed target")
        region = str(target.get("region_name") or "an unspecified room")
        lines.append(f"- {name} in {region}")
    return "\n".join(lines) if lines else "- none"


def rgb_to_jpeg_base64(observations, sensor_key="color_sensor_f", quality=80):
    """Encode current local RGB observation without exposing simulator geometry."""
    rgb = observations.get(sensor_key)
    if rgb is None:
        raise RuntimeError(f"Missing RGB observation '{sensor_key}'.")
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise RuntimeError(f"Unexpected RGB observation shape: {array.shape}.")
    image = Image.fromarray(array[:, :, :3].astype(np.uint8), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("ascii"), tuple(array.shape)


def save_rgb(observations, path, sensor_key="color_sensor_f"):
    """Save an evaluator debug view; this is never used as an oracle feature."""
    rgb = observations.get(sensor_key)
    if rgb is None:
        return None
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array[:, :, :3].astype(np.uint8), mode="RGB").save(path, quality=90)
    return str(path)


def build_policy_prompt(record, completed_targets, history, time_state):
    """Build policy text without coordinates, distances or numeric time state."""
    remaining = set(range(len(record["targets"]))) - set(completed_targets)
    completed_text = target_text(record, completed_targets)
    history_text = "\n".join(history[-12:]) if history else "- no prior actions"
    time_block = f"\n{time_state}\n" if time_state else "\n"
    return f"""You are an embodied visual navigation policy in an unfamiliar indoor scene.
Use the attached current first-person RGB image, the task text and your action history. Choose exactly one primitive action: move_forward, turn_left, or turn_right.
Do not assume a map or a known target location. Target completion is determined externally by the environment.
{time_block}
Long task text:
{record.get('instruction') or record.get('stitched_instruction') or ''}

Remaining targets:
{target_text(record, remaining)}

Completed targets:
{completed_text}

Recent action history:
{history_text}
"""


def make_policy_payload(record, sim, history, include_time_prompt):
    """Create a JSON-serialisable, policy-safe observation payload."""
    image_b64, rgb_shape = rgb_to_jpeg_base64(sim.observations)
    completed = list(sim.completed_targets)
    time_state = fuzzy_time_state(sim.time_used, sim.time_budget, include_time_prompt)
    prompt = build_policy_prompt(record, completed, history, time_state)
    return {
        "prompt": prompt,
        "rgb_jpeg_base64": image_b64,
        "rgb_shape": list(rgb_shape),
        "allowed_actions": list(POLICY_ACTIONS),
    }


def parse_policy_action(raw):
    """Accept a compact JSON response or a single allowed action token."""
    text = (raw or "").strip()
    if not text:
        return None
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            value = value.get("action")
        if isinstance(value, str) and value in POLICY_ACTIONS:
            return value
    except json.JSONDecodeError:
        pass
    tokens = re.findall(r"[a-z_]+", text.lower())
    exact = [token for token in tokens if token in POLICY_ACTIONS]
    return exact[-1] if len(exact) == 1 else None


def environment_auto_complete(sim):
    """Mark all nearby targets after an observation/action, outside policy control.

    The hidden evaluator uses Habitat pathfinder distance and target positions
    only here.  It neither returns these values nor asks the policy for STOP.
    """
    completed_now = []
    for target_index in sorted(tuple(sim.remaining_targets)):
        info = sim.get_target_info(target_index)
        distance = info["geo dis"]
        if np.isfinite(distance) and distance < sim.args.success_dis:
            succeeded, _ = sim._complete_target_if_possible(target_index)
            if succeeded:
                completed_now.append(target_index)
    if not sim.remaining_targets:
        sim.done = True
        sim.episode_over = True
    return completed_now
