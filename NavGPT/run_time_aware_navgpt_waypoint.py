#!/usr/bin/env python3
"""Run a NavGPT-style waypoint agent on scene-level Time-Aware VLN.

This adapter is closer to the original NavGPT setup than the low-level action
runner: the LLM chooses among candidate viewpoints, and Habitat's
GreedyGeodesicFollower executes low-level actions to the selected waypoint.
"""

import argparse
import contextlib
import csv
import io
import json
import math
import random
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import habitat_sim
import numpy as np
import quaternion
from PIL import Image

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
from tools.run_time_aware_navgpt import (
    FUZZY_BY_RATIO,
    SEMANTIC_IGNORE,
    normalize_name,
    semantic_id_to_name,
    visible_semantic_objects,
    visible_target_matches,
)


RELATIVE_ANGLES = [
    ("front", 0.0),
    ("front-left", -45.0),
    ("left", -90.0),
    ("front-right", 45.0),
    ("right", 90.0),
    ("back-left", -135.0),
    ("back-right", 135.0),
    ("back", 180.0),
]


def make_sim(args, record, time_budget):
    sim_args = SimpleNamespace(
        scene=args.scene,
        scene_dataset=args.scene_dataset,
        success_dis=args.success_dis,
        max_step=time_budget,
        no_render=args.no_render,
    )
    sim = TimeAwareSceneSimulator(
        sim_args,
        record_to_config(record),
        time_budget=time_budget,
        target_values=[target.get("value", 1.0) for target in record["targets"]],
    )
    if getattr(args, "disable_action_visualization", False):
        sim.no_render = True
    return sim


def time_condition(args, sim):
    if args.time_prompt_mode == "none":
        return "No time condition is provided."
    overall = FUZZY_BY_RATIO.get(str(args.budget_ratio), "The overall time budget is limited.")
    if args.time_prompt_mode == "ratio_fuzzy":
        return overall
    fraction = sim.time_remaining / sim.time_budget if sim.time_budget else 0.0
    if fraction <= 0.25:
        current = "Current time status: nearly out of time."
    elif fraction <= 0.50:
        current = "Current time status: late in the episode."
    elif fraction <= 0.80:
        current = "Current time status: midway through the episode."
    else:
        current = "Current time status: early in the episode."
    return f"{overall} {current}"


def yaw_from_agent(sim):
    state = sim.agent.get_state()
    rotation = quaternion.as_rotation_matrix(state.rotation)
    forward = rotation @ np.array([0.0, 0.0, -1.0])
    return math.atan2(float(forward[0]), float(-forward[2]))


def yaw_to_direction(yaw, distance):
    return np.array(
        [
            math.sin(yaw) * distance,
            0.0,
            -math.cos(yaw) * distance,
        ],
        dtype=np.float32,
    )


def set_agent_yaw(sim, position, yaw):
    state = habitat_sim.AgentState()
    state.position = np.array(position, dtype=np.float32)
    state.rotation = quaternion.from_rotation_vector([0.0, yaw, 0.0])
    sim.agent.set_state(state)


def geodesic_distance(sim, start, goal):
    path = habitat_sim.nav.ShortestPath()
    path.requested_start = np.array(start, dtype=np.float32)
    path.requested_end = np.array(goal, dtype=np.float32)
    if not sim.pathfinder.find_path(path):
        return math.inf
    return float(path.geodesic_distance)


def save_rgb_observation(args, observations, image_dir, stem):
    if not args.save_candidate_images or image_dir is None:
        return None
    rgb = observations.get("color_sensor_f")
    if rgb is None:
        return None
    array = np.asarray(rgb)
    if array.ndim != 3:
        return None
    if array.shape[-1] == 4:
        image = Image.fromarray(array, mode="RGBA").convert("RGB")
    else:
        image = Image.fromarray(array[:, :, :3])
    image_dir.mkdir(parents=True, exist_ok=True)
    path = image_dir / f"{stem}.jpg"
    image.save(path, quality=90)
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def observation_to_rgb_image(observations):
    rgb = observations.get("color_sensor_f")
    if rgb is None:
        return None
    array = np.asarray(rgb)
    if array.ndim != 3:
        return None
    if array.shape[-1] == 4:
        return Image.fromarray(array, mode="RGBA").convert("RGB")
    return Image.fromarray(array[:, :, :3]).convert("RGB")


def normalize_cache_path(path):
    if not path:
        return None
    text = str(path)
    keys = {text}
    candidate = Path(text)
    if not candidate.is_absolute():
        keys.add(str((REPO_ROOT / candidate).resolve()))
    else:
        try:
            keys.add(str(candidate.resolve().relative_to(REPO_ROOT)))
        except ValueError:
            pass
    return keys


def load_vision_text_cache(path):
    if not path:
        return {}
    cache_path = Path(path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Vision text cache not found: {cache_path}")
    cache = {}
    if cache_path.suffix.lower() == ".jsonl":
        rows = []
        with cache_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        rows = data.values() if isinstance(data, dict) else data
    for row in rows:
        image = row.get("image") or row.get("image_path") or row.get("path")
        for key in normalize_cache_path(image) or []:
            cache[key] = row
    return cache


def format_vision_text(row):
    if not row:
        return None
    parts = []
    caption = row.get("caption") or row.get("summary")
    if caption:
        parts.append(f"caption: {caption}")
    tags = row.get("tags") or row.get("objects")
    if isinstance(tags, str):
        tags = [item.strip() for item in re.split(r"[,|;]", tags) if item.strip()]
    if tags:
        parts.append("tags: " + ", ".join(str(item) for item in tags[:12]))
    return "; ".join(parts) if parts else None


def lookup_vision_text(args, image_path):
    if not image_path or not getattr(args, "vision_text_cache", None):
        return None
    for key in normalize_cache_path(image_path) or []:
        row = args.vision_text_cache.get(key)
        if row:
            return format_vision_text(row)
    return None


def load_online_vision_tagger(args):
    if args.vision_text_provider == "none":
        return None
    if args.vision_text_provider != "ram":
        raise ValueError(f"Unsupported vision text provider: {args.vision_text_provider}")

    import torch

    from nav_gen.recognize_anything.ram import get_transform
    from nav_gen.recognize_anything.ram import inference_ram as ram_inference
    from nav_gen.recognize_anything.ram.models import ram

    checkpoint = Path(args.ram_checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"RAM checkpoint not found: {checkpoint}")
    device = torch.device(args.vision_device or ("cuda" if torch.cuda.is_available() else "cpu"))
    transform = get_transform(image_size=args.ram_image_size)
    model = ram(pretrained=str(checkpoint), image_size=args.ram_image_size, vit=args.ram_vit)
    model.eval()
    model = model.to(device)
    return {
        "provider": "ram",
        "device": device,
        "transform": transform,
        "model": model,
        "inference": ram_inference,
    }


def tags_from_ram(image, tagger):
    import torch

    tensor = tagger["transform"](image).unsqueeze(0).to(tagger["device"])
    with torch.no_grad():
        result = tagger["inference"](tensor, tagger["model"])
    english = result[0]
    if isinstance(english, str):
        tags = [item.strip() for item in english.replace("|", ",").split(",") if item.strip()]
    else:
        tags = [str(item).strip() for item in english if str(item).strip()]
    return tags


def online_vision_text(args, observations):
    tagger = getattr(args, "online_vision_tagger", None)
    if not tagger:
        return None
    image = observation_to_rgb_image(observations)
    if image is None:
        return None
    if tagger["provider"] == "ram":
        tags = tags_from_ram(image, tagger)
        if tags:
            return "ram_tags: " + ", ".join(tags[: args.vision_top_k])
    return None


def render_candidate_observation(args, sim, position, yaw, image_dir=None, image_stem=None):
    if args.no_render:
        return [], None, None
    original_state = sim.agent.get_state()
    original_observations = sim.observations
    try:
        set_agent_yaw(sim, position, yaw)
        sim.observations = sim.sim.get_sensor_observations()
        visible = visible_semantic_objects(sim, args.semantic_top_k)
        rgb_image = save_rgb_observation(args, sim.observations, image_dir, image_stem)
        vision_text = lookup_vision_text(args, rgb_image)
        if not vision_text:
            vision_text = online_vision_text(args, sim.observations)
    finally:
        sim.agent.set_state(original_state)
        sim.observations = original_observations
    return visible, rgb_image, vision_text


def candidate_target_matches(record, sim, visible):
    matches = visible_target_matches(record, sim, visible)
    return matches[:4]


def candidate_target_cues(record, sim, visible):
    matches = candidate_target_matches(record, sim, visible)
    if not matches:
        return "no visible remaining target category"
    return "; ".join(
        f"target {item['target_index']} ({item['target_name']}) may match visible {item['visible_name']}"
        for item in matches
    )


def prominence_label(pixel_fraction):
    if pixel_fraction >= 0.12:
        return "dominant"
    if pixel_fraction >= 0.035:
        return "clear"
    return "small"


def object_summary(objects):
    if not objects:
        return "no recognizable objects"
    return ", ".join(
        f"{item['name']} ({prominence_label(item.get('pixel_fraction', 0.0))})"
        for item in objects[:5]
    )


def candidate_novelty(candidate_position, visited_positions):
    stats = candidate_visit_stats(candidate_position, visited_positions)
    nearest = stats["recent_path_distance"]
    if nearest < 0.8:
        return "recently visited"
    if nearest < 1.8:
        return "near previous path"
    return "new area"


def candidate_visit_stats(candidate_position, visited_positions):
    if not visited_positions:
        return {"visit_count": 0, "recently_visited": False, "recent_path_distance": math.inf}
    current = np.array(candidate_position, dtype=np.float32)
    distances = [
        float(np.linalg.norm(current[[0, 2]] - np.array(position, dtype=np.float32)[[0, 2]]))
        for position in visited_positions
    ]
    nearest = min(distances) if distances else math.inf
    visit_count = sum(1 for distance in distances if distance < 0.8)
    return {
        "visit_count": visit_count,
        "recently_visited": nearest < 0.8,
        "recent_path_distance": nearest,
    }


def should_show_exploration(args):
    return not getattr(args, "hide_exploration_label", False) and getattr(args, "memory_prompt_mode", "full") != "target_only"


def candidate_line(args, candidate):
    object_text = object_summary(candidate["visible"])
    if candidate.get("vision_text"):
        object_text = f"{object_text}; visual_text={candidate['vision_text']}"
    matches = candidate.get("target_matches") or []
    if matches:
        match_text = ", ".join(
            f"{item['target_index']}:{item['target_name']}" for item in matches
        )
    else:
        match_text = "none"
    exploration_text = f"exploration={candidate['novelty']}, " if should_show_exploration(args) else ""
    repeat_text = ""
    if getattr(args, "show_repeat_info", False):
        recent_distance = candidate.get("recent_path_distance")
        if recent_distance is None or math.isinf(float(recent_distance)):
            recent_distance_text = "unvisited"
        else:
            recent_distance_text = f"{float(recent_distance):.1f}m"
        repeat_text = (
            f"recently_visited={'yes' if candidate.get('recently_visited') else 'no'}, "
            f"visit_count={candidate.get('visit_count', 0)}, "
            f"distance_from_recent_path={recent_distance_text}, "
        )
    match_label = "possible_category_overlap" if getattr(args, "weak_target_cues", False) else "target_matches"
    cue_text = candidate['target_cue']
    if getattr(args, "weak_target_cues", False) and matches:
        cue_text = f"weak category cue only; {cue_text}"
    return (
        f"- index={candidate['index']}, direction={candidate['direction']}, "
        f"path={candidate['distance_label']}, {exploration_text}{repeat_text}"
        f"visible={object_text}, {match_label}={match_text}, cue={cue_text}, "
        f"rgb_image={candidate.get('rgb_image') or 'not saved'}"
    )


def parse_visual_text_tags(vision_text):
    if not vision_text:
        return []
    text = str(vision_text)
    if ":" in text:
        text = text.split(":", 1)[1]
    tags = [item.strip() for item in re.split(r"[,|;]", text) if item.strip()]
    return tags


def candidate_memory_keywords(candidate, limit=10):
    names = [item["name"] for item in candidate.get("visible", []) if item.get("name")]
    names.extend(parse_visual_text_tags(candidate.get("vision_text")))
    keywords = []
    seen = set()
    for name in names:
        normalized = normalize_name(name)
        if not normalized or normalized in SEMANTIC_IGNORE or normalized in seen:
            continue
        seen.add(normalized)
        keywords.append(name)
        if len(keywords) >= limit:
            break
    return keywords


def position_distance_2d(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    return float(np.linalg.norm(a[[0, 2]] - b[[0, 2]]))


def find_memory_node(memory_nodes, position, merge_radius):
    if not memory_nodes:
        return None
    best = min(memory_nodes, key=lambda node: position_distance_2d(node["position"], position))
    if position_distance_2d(best["position"], position) <= merge_radius:
        return best
    return None


def update_topological_memory(args, memory_nodes, position, candidate, decision_index, record, auto_completed):
    if not args.use_global_memory or candidate is None:
        return
    node = find_memory_node(memory_nodes, position, args.memory_merge_radius)
    if node is None:
        node = {
            "node_id": len(memory_nodes),
            "position": np.array(position, dtype=np.float32).tolist(),
            "first_seen_decision": decision_index,
            "last_seen_decision": decision_index,
            "visit_count": 0,
            "directions": [],
            "objects": [],
            "target_matches": [],
            "completed_targets": [],
            "rgb_images": [],
        }
        memory_nodes.append(node)
    node["visit_count"] += 1
    node["last_seen_decision"] = decision_index
    if candidate.get("direction") and candidate["direction"] not in node["directions"]:
        node["directions"].append(candidate["direction"])
    for keyword in candidate_memory_keywords(candidate):
        if keyword not in node["objects"]:
            node["objects"].append(keyword)
    for match in candidate.get("target_matches") or []:
        item = {
            "target_index": match.get("target_index"),
            "target_name": match.get("target_name"),
            "visible_name": match.get("visible_name"),
        }
        if item not in node["target_matches"]:
            node["target_matches"].append(item)
    for target_index in auto_completed or []:
        target = record["targets"][target_index] if target_index < len(record["targets"]) else {}
        item = {
            "target_index": target_index,
            "target_name": target.get("name", str(target_index)),
        }
        if item not in node["completed_targets"]:
            node["completed_targets"].append(item)
    if candidate.get("rgb_image") and candidate["rgb_image"] not in node["rgb_images"]:
        node["rgb_images"].append(candidate["rgb_image"])


def memory_node_line(node):
    objects = ", ".join(node.get("objects", [])[:8]) or "unknown objects"
    directions = ", ".join(node.get("directions", [])[:4]) or "unknown direction"
    matches = node.get("target_matches", [])
    if matches:
        match_text = ", ".join(
            f"{item['target_index']}:{item['target_name']}" for item in matches[:5]
        )
    else:
        match_text = "none"
    completed = node.get("completed_targets", [])
    if completed:
        completed_text = ", ".join(
            f"{item['target_index']}:{item['target_name']}" for item in completed
        )
    else:
        completed_text = "none"
    return (
        f"- node {node['node_id']}: visits={node['visit_count']}, "
        f"last_seen=decision {node['last_seen_decision']}, from={directions}, "
        f"objects={objects}, target_matches={match_text}, completed_here={completed_text}"
    )


def build_global_memory_text(args, memory_nodes, candidates):
    if not memory_nodes:
        memory_text = "No explored waypoint has been recorded yet."
    else:
        recent = sorted(memory_nodes, key=lambda node: node["last_seen_decision"], reverse=True)
        recent = recent[: args.memory_window]
        memory_text = "\n".join(memory_node_line(node) for node in recent)

    target_related = []
    for node in memory_nodes:
        for match in node.get("target_matches", []):
            target_related.append(
                f"node {node['node_id']} saw possible target {match['target_index']}:{match['target_name']} "
                f"as {match['visible_name']}"
            )
    if target_related:
        target_text = "\n".join(f"- {item}" for item in target_related[-args.memory_window :])
    else:
        target_text = "- No previous target-related visual cue has been recorded."

    recent_areas = []
    for node in sorted(memory_nodes, key=lambda item: item["last_seen_decision"], reverse=True)[: args.memory_window]:
        objects = ", ".join(node.get("objects", [])[:5]) or "unknown objects"
        recent_areas.append(
            f"- node {node['node_id']}: visits={node['visit_count']}, "
            f"last_seen=decision {node['last_seen_decision']}, objects={objects}"
        )
    recent_area_text = "\n".join(recent_areas) if recent_areas else "- No visited area has been recorded yet."

    if args.memory_prompt_mode == "target_only":
        return f"""Target-related memory:
{target_text}

Recently visited areas:
{recent_area_text}

Use this memory only to avoid repeated unhelpful areas and to revisit useful target cues.
Prioritize current visual evidence, remaining targets, and candidate metadata."""

    frontier_candidates = [
        candidate for candidate in candidates
        if candidate.get("target_matches") or candidate.get("novelty") == "new area"
    ][: args.frontier_top_k]
    if frontier_candidates:
        frontier_text = "\n".join(
            f"- candidate {candidate['index']}: direction={candidate['direction']}, "
            f"exploration={candidate['novelty']}, visible={object_summary(candidate['visible'])}, "
            f"visual_text={candidate.get('vision_text') or 'none'}"
            for candidate in frontier_candidates
        )
    else:
        frontier_text = "- No candidate is clearly new or target-related."

    return f"""Explored topological nodes:
{memory_text}

Target-related memory:
{target_text}

Current frontier summary:
{frontier_text}"""


def distance_label(distance):
    if distance < 0.9:
        return "very short"
    if distance < 1.6:
        return "short"
    if distance < 2.6:
        return "medium"
    return "long"


def generate_candidates(args, record, sim, image_dir=None, decision_index=None, visited_positions=None):
    position, _ = sim.return_state()
    base_yaw = yaw_from_agent(sim)
    candidates = []
    seen = set()
    min_distance = max(
        float(args.min_candidate_distance),
        float(args.waypoint_radius) + 0.1,
        float(args.success_dis) + 0.2,
    )
    for direction, angle_degrees in RELATIVE_ANGLES:
        for radius in args.candidate_radii:
            yaw = base_yaw + math.radians(angle_degrees)
            raw_goal = np.array(position, dtype=np.float32) + yaw_to_direction(yaw, radius)
            snapped = np.array(sim.pathfinder.snap_point(raw_goal), dtype=np.float32)
            key = tuple(np.round(snapped[[0, 2]], 2))
            if key in seen:
                continue
            distance = geodesic_distance(sim, position, snapped)
            if math.isinf(distance):
                continue
            if distance < min_distance or distance > args.max_candidate_geodesic:
                continue
            seen.add(key)
            image_stem = None
            if decision_index is not None:
                image_stem = f"decision_{decision_index:03d}_candidate_{len(candidates):02d}_{direction}"
            visible, rgb_image, vision_text = render_candidate_observation(
                args, sim, position, yaw, image_dir, image_stem
            )
            target_matches = candidate_target_matches(record, sim, visible)
            visit_stats = candidate_visit_stats(snapped, visited_positions or [])
            candidates.append(
                {
                    "index": len(candidates),
                    "direction": direction,
                    "relative_angle": angle_degrees,
                    "radius": radius,
                    "position": snapped.tolist(),
                    "geodesic_distance": distance,
                    "distance_label": distance_label(distance),
                    "visible": visible,
                    "target_matches": target_matches,
                    "target_cue": candidate_target_cues(record, sim, visible),
                    "novelty": candidate_novelty(snapped, visited_positions or []),
                    "visit_count": visit_stats["visit_count"],
                    "recently_visited": visit_stats["recently_visited"],
                    "recent_path_distance": visit_stats["recent_path_distance"],
                    "rgb_image": rgb_image,
                    "vision_text": vision_text,
                }
            )
            if len(candidates) >= args.max_candidates:
                return candidates
    return candidates


def nearby_target_text(args, record, sim):
    nearby = []
    for index in sorted(sim.remaining_targets):
        info = sim.get_target_info(index)
        distance = info["geo dis"]
        if not math.isinf(distance) and distance <= args.stop_hint_distance:
            target = record["targets"][index]
            nearby.append(f"target {index} ({target.get('name', index)})")
    if nearby:
        return "Nearby target hint: " + "; ".join(nearby) + ". STOP may complete a target."
    return "Nearby target hint: no remaining target is believed to be immediately nearby."


def nearby_target_indices(args, sim, distance_threshold=None):
    threshold = args.stop_hint_distance if distance_threshold is None else distance_threshold
    nearby = []
    for index in sorted(sim.remaining_targets):
        info = sim.get_target_info(index)
        distance = info["geo dis"]
        if not math.isinf(distance) and distance < threshold:
            nearby.append(index)
    return nearby


def auto_stop_if_near_target(args, sim):
    completed = []
    while not sim.episode_over:
        nearby = nearby_target_indices(args, sim, distance_threshold=args.success_dis)
        if not nearby:
            break
        before = set(sim.completed_targets)
        sim.actor("stop")
        after = set(sim.completed_targets)
        new_completed = sorted(after - before)
        if not new_completed:
            break
        completed.extend(new_completed)
    return completed


def choose_guard_fallback_candidate(candidates):
    if not candidates:
        return None

    def score(candidate):
        match_bonus = 10 if candidate.get("target_matches") else 0
        novelty_bonus = {
            "new area": 2,
            "near previous path": 1,
            "recently visited": 0,
        }.get(candidate.get("novelty"), 0)
        distance_penalty = float(candidate.get("geodesic_distance", 0.0))
        return (match_bonus + novelty_bonus, -distance_penalty)

    return max(candidates, key=score)


def build_prompt(args, record, sim, history, candidates, memory_nodes=None):
    completed = [
        record["targets"][index].get("name", str(index))
        for index in sim.completed_targets
    ]
    remaining = [
        f"- target {index}: {record['targets'][index].get('name', index)} in "
        f"{record['targets'][index].get('region_name') or record['targets'][index].get('region') or 'unknown'}"
        for index in sorted(sim.remaining_targets)
    ]
    history_text = "\n".join(history[-args.history_window:]) if history else "No previous waypoint choices."
    candidate_text = "\n".join(candidate_line(args, candidate) for candidate in candidates) or "No navigable candidates are available."
    memory_section = ""
    if args.use_global_memory:
        memory_title = "Episodic memory" if args.memory_prompt_mode == "target_only" else "Global memory / topological map"
        memory_section = f"\n{memory_title}:\n{build_global_memory_text(args, memory_nodes or [], candidates)}\n"
    if args.auto_stop:
        stop_instruction = (
            "The runner will automatically complete a target when the agent reaches it. "
            "You must not output STOP; choose the next candidate viewpoint only."
        )
        output_instruction = "Output exactly one candidate index."
    else:
        stop_instruction = (
            "Choose STOP only when a nearby target hint says a target may be immediately nearby, "
            "or when the current view clearly indicates you are adjacent to a target."
        )
        output_instruction = "Output exactly one candidate index, or STOP."
    if args.weak_target_cues:
        selection_rules = """Choose the candidate viewpoint that is most likely to help complete remaining targets.
Use visual evidence, task instruction, progress, candidate metadata, and memory.
Treat possible_category_overlap as a weak cue only; it is not a confirmed target match.
Do not choose a candidate only because a common category such as table, book, chair, or desk overlaps with a target name.
Prefer candidates that combine useful visual evidence with progress to less visited or more promising areas.
Avoid repeatedly visiting areas that did not provide useful target cues."""
    else:
        selection_rules = """Choose the candidate viewpoint that is most likely to help complete remaining targets.
Use visual evidence, task instruction, progress, candidate metadata, and memory.
Prefer candidates with target_matches when the evidence is reliable.
If no candidate clearly matches a target, choose the candidate that seems most useful for finding or approaching remaining targets.
Avoid repeatedly visiting areas that did not provide useful target cues."""
    return f"""You are NavGPT, an embodied navigation agent in a Habitat indoor scene.
You navigate by choosing one candidate viewpoint at a time. A low-level Habitat follower will move you to the chosen candidate.

Time condition: {time_condition(args, sim)}

Goal:
Complete as many unordered targets as possible before time runs out.
{stop_instruction}
{selection_rules}

Instruction:
{record_instruction(record)}

Progress:
Completed {len(sim.completed_targets)}/{len(record['targets'])}: {', '.join(completed) if completed else 'none'}

Remaining targets:
{chr(10).join(remaining)}

History:
{history_text}
{memory_section}

{nearby_target_text(args, record, sim)}

Candidate viewpoints:
{candidate_text}

{output_instruction}
"""


def parse_choice(raw_response, candidates):
    text = raw_response.strip()
    if re.search(r"\bSTOP\b", text, flags=re.IGNORECASE):
        return "stop"
    valid = {candidate["index"] for candidate in candidates}
    for match in re.finditer(r"-?\d+", text):
        value = int(match.group(0))
        if value in valid:
            return value
    return None


def choose_waypoint(args, prompt, candidates):
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
        return None, f"NavGPT waypoint command timed out after {args.llm_timeout}s", True
    raw = completed.stdout.strip()
    if completed.stderr.strip():
        raw = f"{raw}\n[stderr]\n{completed.stderr.strip()}".strip()
    if completed.returncode != 0:
        raw = f"{raw}\n[returncode]\n{completed.returncode}".strip()
        return None, raw, True
    return parse_choice(completed.stdout, candidates), raw, False


def execute_to_waypoint(args, sim, goal_position):
    executed = 0
    goal = np.array(goal_position, dtype=np.float32)
    while not sim.episode_over and executed < args.max_waypoint_steps:
        distance = geodesic_distance(sim, sim.return_state()[0], goal)
        if math.isinf(distance) or distance <= args.waypoint_radius:
            break
        try:
            action = sim.get_next_action(goal)
        except habitat_sim.errors.GreedyFollowerError:
            break
        if action is None or action == "stop":
            break
        sim.actor(action)
        executed += 1
    return executed


def episode_image_dir(args, record, ratio_key):
    if not args.save_candidate_images:
        return None
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", record_id(record))
    return Path(args.output_dir) / "candidate_images" / f"budget_{ratio_name(ratio_key)}" / safe_id


def run_episode(args, record):
    ratio_key = str(args.budget_ratio)
    time_budget = record["time_budgets"].get(ratio_key)
    if time_budget is None:
        oracle_time = record.get("oracle_optimal_time")
        if oracle_time is None:
            raise ValueError(f"Missing budget ratio {ratio_key} for {record_id(record)}")
        time_budget = max(1, int(round(float(args.budget_ratio) * float(oracle_time))))
    sim = make_sim(args, record, time_budget)
    trace = []
    history = []
    memory_nodes = []
    start_position, _ = sim.return_state()
    visited_positions = [np.array(start_position, dtype=np.float32).tolist()]
    image_dir = episode_image_dir(args, record, args.budget_ratio)
    try:
        sim.actor("stop")
        while not sim.episode_over and sim.remaining_targets and len(trace) < args.max_decisions:
            decision_index = len(trace) + 1
            candidates = generate_candidates(
                args,
                record,
                sim,
                image_dir=image_dir,
                decision_index=decision_index,
                visited_positions=visited_positions,
            )
            if args.shuffle_candidates and candidates:
                rng = random.Random(f"{args.candidate_shuffle_seed}:{record_id(record)}:{args.budget_ratio}:{decision_index}")
                for candidate in candidates:
                    candidate.setdefault("original_index", candidate["index"])
                rng.shuffle(candidates)
                for display_index, candidate in enumerate(candidates):
                    candidate["index"] = display_index
            prompt = build_prompt(args, record, sim, history, candidates, memory_nodes=memory_nodes)
            choice, raw_response, command_failed = choose_waypoint(args, prompt, candidates)
            event = "selected_candidate"
            executed = 0
            selected = None
            if command_failed:
                event = "command_failed"
                if args.invalid_choice_policy == "end_episode":
                    trace.append(
                        {
                            "time_used": sim.time_used,
                            "time_remaining": sim.time_remaining,
                            "event": event,
                            "raw_response": raw_response,
                            "prompt": prompt if args.save_prompts else None,
                        }
                    )
                    break
            if choice == "stop":
                nearby_targets = nearby_target_indices(args, sim)
                if args.stop_guard and not nearby_targets:
                    selected = choose_guard_fallback_candidate(candidates)
                    if selected is not None:
                        executed = execute_to_waypoint(args, sim, selected["position"])
                        current_position, _ = sim.return_state()
                        visited_positions.append(np.array(current_position, dtype=np.float32).tolist())
                        event = "stop_guard_blocked_fallback_candidate"
                    else:
                        event = "stop_guard_blocked_no_candidate"
                        break
                else:
                    completed_before = len(sim.completed_targets)
                    sim.actor("stop")
                    executed = 1
                    event = "stop_succeeded" if len(sim.completed_targets) > completed_before else "stop_failed"
            elif isinstance(choice, int) and 0 <= choice < len(candidates):
                selected = candidates[choice]
                executed = execute_to_waypoint(args, sim, selected["position"])
                current_position, _ = sim.return_state()
                visited_positions.append(np.array(current_position, dtype=np.float32).tolist())
            else:
                event = "invalid_choice"
                if candidates and args.invalid_choice_policy == "first":
                    selected = candidates[0]
                    executed = execute_to_waypoint(args, sim, selected["position"])
                    event = "fallback_first_candidate"
                elif args.invalid_choice_policy == "stop":
                    sim.actor("stop")
                    executed = 1
                    event = "fallback_stop"
                else:
                    break

            auto_completed = []
            if args.auto_stop and selected is not None and not sim.episode_over:
                before_time = sim.time_used
                auto_completed = auto_stop_if_near_target(args, sim)
                executed += sim.time_used - before_time
                if auto_completed:
                    event = f"{event}_auto_stop_completed"

            if selected is not None:
                current_position, _ = sim.return_state()
                update_topological_memory(
                    args,
                    memory_nodes,
                    current_position,
                    selected,
                    len(trace) + 1,
                    record,
                    auto_completed,
                )

            if selected is not None:
                if should_show_exploration(args):
                    selected_text = (
                        f"direction={selected['direction']}, path={selected['distance_label']}, "
                        f"exploration={selected['novelty']}, visible={object_summary(selected['visible'])}"
                    )
                else:
                    selected_text = (
                        f"direction={selected['direction']}, path={selected['distance_label']}, "
                        f"visible={object_summary(selected['visible'])}"
                    )
            else:
                selected_text = "no waypoint selected"
            history.append(
                f"Decision {len(history)+1}: choice={choice}, event={event}, {selected_text}, "
                f"executed_steps={executed}, completed={len(sim.completed_targets)}/{len(record['targets'])}"
            )
            trace.append(
                {
                    "time_used": sim.time_used,
                    "time_remaining": sim.time_remaining,
                    "choice": choice,
                    "event": event,
                    "executed_steps": executed,
                    "auto_completed_targets": auto_completed,
                    "selected_candidate": selected,
                    "candidates": candidates,
                    "raw_response": raw_response,
                    "prompt": prompt if args.save_prompts else None,
                    "memory_size": len(memory_nodes),
                }
            )
        result = sim.return_results()
    finally:
        sim.close()
    result["task_id"] = record_id(record)
    result["split"] = record["split"]
    result["scene"] = record["scene"]
    result["robot"] = record["robot"]
    result["baseline"] = args.baseline_name
    if args.vision_text_provider != "none":
        result["observation_mode"] = f"waypoint_semantic_text_with_{args.vision_text_provider}_vision_text"
    elif args.vision_text_cache_path:
        result["observation_mode"] = "waypoint_semantic_text_with_cached_vision_text"
    else:
        result["observation_mode"] = "waypoint_semantic_text_rgb_saved" if args.save_candidate_images else "waypoint_semantic_text"
    if args.use_global_memory:
        result["observation_mode"] += "_global_memory"
    result["stop_guard"] = bool(args.stop_guard)
    result["auto_stop"] = bool(args.auto_stop)
    result["global_memory"] = bool(args.use_global_memory)
    result["topological_memory_nodes"] = memory_nodes if args.use_global_memory else []
    result["navgpt_waypoint_trace"] = trace
    return result


def ratio_name(ratio):
    return str(ratio).replace(".", "p")


def write_summary_csv_with_name(path, rows):
    for row in rows:
        row.setdefault("baseline", "navgpt_waypoint_adapter")
    write_summary_csv(path, rows)


def read_checkpoint(path):
    results = []
    if not path.exists():
        return results
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))
    return results


def append_checkpoint(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False) + "\n")


def parse_radii(value):
    return [float(item) for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes",
        default="/file_system/nas/algorithm/Intern03/data/time_aware_scene/all_episodes_spatial_start_floor_oracle_time.jsonl",
    )
    parser.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--limit", type=int, default=2, help="Use 0 or a negative value to run all matching records.")
    parser.add_argument("--budget-ratio", type=float, default=None)
    parser.add_argument("--budget-ratios", default="0.5,1.0,1.5")
    parser.add_argument("--navgpt-command", required=True)
    parser.add_argument("--baseline-name", default="navgpt_waypoint_adapter")
    parser.add_argument("--llm-timeout", type=float, default=60.0)
    parser.add_argument("--history-window", type=int, default=10)
    parser.add_argument("--use-global-memory", action="store_true")
    parser.add_argument("--memory-prompt-mode", choices=["full", "target_only"], default="full")
    parser.add_argument("--hide-exploration-label", action="store_true")
    parser.add_argument("--memory-window", type=int, default=8)
    parser.add_argument("--memory-merge-radius", type=float, default=1.0)
    parser.add_argument("--frontier-top-k", type=int, default=5)
    parser.add_argument("--max-decisions", type=int, default=80)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--candidate-radii", default="1.8,2.6")
    parser.add_argument("--min-candidate-distance", type=float, default=0.35)
    parser.add_argument("--max-candidate-geodesic", type=float, default=3.0)
    parser.add_argument("--waypoint-radius", type=float, default=0.75)
    parser.add_argument("--max-waypoint-steps", type=int, default=12)
    parser.add_argument("--semantic-top-k", type=int, default=8)
    parser.add_argument("--stop-hint-distance", type=float, default=1.0)
    parser.add_argument("--shuffle-candidates", action="store_true", help="Shuffle displayed candidate order each decision to reduce index-order bias.")
    parser.add_argument("--candidate-shuffle-seed", type=int, default=0)
    parser.add_argument("--weak-target-cues", action="store_true", help="Present category overlaps as weak cues instead of target_matches.")
    parser.add_argument("--show-repeat-info", action="store_true", help="Show recently_visited, visit_count, and distance_from_recent_path in candidate text.")
    parser.add_argument("--time-prompt-mode", choices=["dynamic_fuzzy", "ratio_fuzzy", "none"], default="dynamic_fuzzy")
    parser.add_argument("--invalid-choice-policy", choices=["end_episode", "first", "stop"], default="first")
    parser.add_argument("--disable-stop-guard", dest="stop_guard", action="store_false")
    parser.add_argument("--auto-stop", action="store_true")
    parser.set_defaults(stop_guard=True)
    parser.add_argument("--scene", default="data/hm3d/")
    parser.add_argument(
        "--scene-dataset",
        default="data/hm3d/hm3d_annotated_basis.scene_dataset_config.json",
    )
    parser.add_argument("--success-dis", type=float, default=1.0)
    parser.add_argument("--no-render", action="store_true", default=True)
    parser.add_argument("--render", dest="no_render", action="store_false")
    parser.add_argument("--output-dir", default="output/time_aware_scene/navgpt_waypoint")
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--save-prompts", action="store_true")
    parser.add_argument("--save-candidate-images", action="store_true")
    parser.add_argument(
        "--disable-action-visualization",
        action="store_true",
        help="Keep render sensors for candidate observations but skip per-action visualization image generation.",
    )
    parser.add_argument(
        "--vision-text-cache",
        dest="vision_text_cache_path",
        default=None,
        help="Optional JSON/JSONL mapping candidate RGB image paths to caption/tags text.",
    )
    parser.add_argument(
        "--vision-text-provider",
        choices=["none", "ram"],
        default="none",
        help="Optionally generate visual text online from candidate RGB images.",
    )
    parser.add_argument(
        "--ram-checkpoint",
        default="/file_system/nas/algorithm/Intern03/models/recognize_anything/ram_swin_large_14m.pth",
    )
    parser.add_argument("--ram-vit", default="swin_l")
    parser.add_argument("--ram-image-size", type=int, default=384)
    parser.add_argument("--vision-device", default=None)
    parser.add_argument("--vision-top-k", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    args.candidate_radii = parse_radii(args.candidate_radii)
    args.vision_text_cache = load_vision_text_cache(args.vision_text_cache_path)
    args.online_vision_tagger = load_online_vision_tagger(args)
    if args.no_render:
        raise ValueError("Waypoint semantic adapter requires --render.")

    load_limit = None if args.limit <= 0 else args.limit
    records = load_records(args.episodes, args.split, load_limit)
    budget_ratios = [args.budget_ratio] if args.budget_ratio is not None else parse_budget_ratios(args.budget_ratios)
    summary_rows = []
    for ratio in budget_ratios:
        args.budget_ratio = ratio
        checkpoint = Path(args.output_dir) / f"navgpt_waypoint_{args.split}_budget_{ratio_name(ratio)}.jsonl"
        results = read_checkpoint(checkpoint) if args.resume else []
        completed_ids = {item.get("task_id") for item in results}
        print(f"\n===== NavGPT-Waypoint budget_ratio={ratio} split={args.split} episodes={len(records)} =====")
        for index, record in enumerate(records, start=1):
            rid = record_id(record)
            if rid in completed_ids:
                print(f"===== [{index}/{len(records)}] {rid} (checkpoint skip) =====")
                continue
            print(f"===== [{index}/{len(records)}] {record_id(record)} =====")
            if args.quiet:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = run_episode(args, record)
            else:
                result = run_episode(args, record)
            results.append(result)
            append_checkpoint(checkpoint, result)
            completed_ids.add(result.get("task_id"))
        summary = summarize(results)
        summary["split"] = args.split
        summary["budget_ratio"] = ratio
        summary["limit"] = args.limit
        if args.vision_text_provider != "none":
            summary["observation_mode"] = f"waypoint_semantic_text_with_{args.vision_text_provider}_vision_text"
        elif args.vision_text_cache_path:
            summary["observation_mode"] = "waypoint_semantic_text_with_cached_vision_text"
        else:
            summary["observation_mode"] = "waypoint_semantic_text_rgb_saved" if args.save_candidate_images else "waypoint_semantic_text"
        if args.use_global_memory:
            summary["observation_mode"] += "_global_memory"
        summary["time_prompt_mode"] = args.time_prompt_mode
        summary["baseline"] = args.baseline_name
        summary["stop_guard"] = bool(args.stop_guard)
        summary["auto_stop"] = bool(args.auto_stop)
        summary["global_memory"] = bool(args.use_global_memory)
        print("\nsummary:")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        output = Path(args.output_dir) / f"navgpt_waypoint_{args.split}_budget_{ratio_name(ratio)}.json"
        write_json(output, summary, results)
        summary_rows.append(summary)

    summary_csv = Path(args.summary_csv) if args.summary_csv else Path(args.output_dir) / f"navgpt_waypoint_{args.split}_summary.csv"
    write_summary_csv_with_name(summary_csv, summary_rows)


if __name__ == "__main__":
    main()
