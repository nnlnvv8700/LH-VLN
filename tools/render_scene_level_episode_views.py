# 用途：在 Habitat/HM3D 中渲染指定 scene-level episode 的起点、目标附近及候选视角 RGB 和鸟瞰图。
#!/usr/bin/env python3
"""Render real RGB views for a scene-level time-aware episode.

This is a prototype visualizer for checking whether HM3D scenes can be shown as
real RGB images rather than only Habitat navmesh masks.
"""

import argparse
import json
import math
import random
import sys
import textwrap
from pathlib import Path

import habitat_sim
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import quaternion

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from habitat_base.config import make_cfg, make_setting


def load_episodes(path):
    episodes = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                episodes.append(json.loads(line))
    return episodes


def safe_name(name):
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in name)


def select_episode(episodes, episode_id=None, seed=0):
    if episode_id:
        for episode in episodes:
            if episode["scene_episode_id"] == episode_id:
                return episode
        raise ValueError(f"Episode id not found: {episode_id}")
    return random.Random(seed).choice(episodes)


def make_sim(episode, scene_root, scene_dataset, width, height, sensor_kind, ortho_scale=None):
    class Args:
        pass

    args = Args()
    args.scene = scene_root
    args.scene_dataset = scene_dataset
    settings = make_setting(args, episode["scene"], episode.get("robot") or "stretch")
    settings["width"] = width
    settings["height"] = height
    for sensor_key in (
        "color_sensor_f",
        "color_sensor_l",
        "color_sensor_r",
        "color_sensor_3rd",
        "depth_sensor_f",
        "depth_sensor_l",
        "depth_sensor_r",
        "semantic_sensor",
    ):
        settings[sensor_key] = False
    settings["color_sensor_f"] = True
    cfg = make_cfg(settings)
    sensor_spec = cfg.agents[0].sensor_specifications[0]
    if sensor_kind == "topdown":
        sensor_spec.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
        sensor_spec.orientation = [-math.pi / 2.0, 0.0, 0.0]
        if ortho_scale is not None:
            sensor_spec.ortho_scale = float(ortho_scale)
    else:
        sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_spec.hfov = 90.0
        sensor_spec.orientation = [0.0, 0.0, 0.0]
    return habitat_sim.Simulator(cfg)


def get_scene_bounds(episode, scene_root, scene_dataset):
    sim = make_sim(episode, scene_root, scene_dataset, 64, 64, "first_person")
    try:
        return sim.pathfinder.get_bounds()
    finally:
        sim.close()


def set_agent_state(sim, position, yaw=0.0):
    state = habitat_sim.AgentState()
    state.position = np.asarray(position, dtype=np.float32)
    state.rotation = quaternion.from_rotation_vector([0.0, float(yaw), 0.0])
    sim.initialize_agent(0, state)


def render_color(sim):
    observations = sim.get_sensor_observations()
    image = observations["color_sensor_f"]
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image


def yaw_towards(src, dst):
    dx = float(dst[0]) - float(src[0])
    dz = float(dst[2]) - float(src[2])
    return math.atan2(-dx, -dz)


def topdown_pixel(pathfinder, point, meters_per_pixel):
    bounds = pathfinder.get_bounds()
    px = (float(point[0]) - bounds[0][0]) / meters_per_pixel
    py = (float(point[2]) - bounds[0][2]) / meters_per_pixel
    return px, py


def save_navmesh_reference(episode, output_path, scene_root, scene_dataset, meters_per_pixel):
    sim = make_sim(episode, scene_root, scene_dataset, 64, 64, "first_person")
    try:
        pathfinder = sim.pathfinder
        start = episode.get("start_position")
        height = float(start[1]) if start is not None else float(pathfinder.get_bounds()[0][1])
        mask = pathfinder.get_topdown_view(meters_per_pixel, height)
        image = np.zeros((*mask.shape, 3), dtype=np.uint8)
        image[mask > 0] = [242, 242, 242]
        image[mask == 0] = [36, 36, 36]

        fig, ax = plt.subplots(figsize=(9, 9), dpi=160)
        ax.imshow(image)
        if start is not None:
            sx, sy = topdown_pixel(pathfinder, start, meters_per_pixel)
            ax.scatter([sx], [sy], marker="*", s=280, c="#d62728", edgecolors="black")
            ax.annotate("START", (sx, sy), xytext=(6, 6), textcoords="offset points", weight="bold")
        colors = plt.cm.tab10.colors
        for target in episode.get("targets", []):
            if target.get("target_position") is None:
                continue
            tx, ty = topdown_pixel(pathfinder, target["target_position"], meters_per_pixel)
            index = int(target.get("global_index", 0))
            ax.scatter([tx], [ty], s=130, c=[colors[index % len(colors)]], edgecolors="black")
            ax.annotate(str(index), (tx, ty), xytext=(5, 5), textcoords="offset points", weight="bold")
        ax.set_axis_off()
        ax.set_title(f"Navmesh reference: {episode['scene_episode_id']}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
    finally:
        sim.close()


def save_topdown_rgb(episode, output_path, scene_root, scene_dataset, width, height, bounds):
    lower, upper = bounds
    center = [
        (float(lower[0]) + float(upper[0])) / 2.0,
        float(upper[1]) + 1.5,
        (float(lower[2]) + float(upper[2])) / 2.0,
    ]
    x_extent = float(upper[0]) - float(lower[0])
    z_extent = float(upper[2]) - float(lower[2])
    ortho_scale = max(x_extent, z_extent) * 1.08

    sim = make_sim(episode, scene_root, scene_dataset, width, height, "topdown", ortho_scale=ortho_scale)
    try:
        set_agent_state(sim, center, yaw=0.0)
        image = render_color(sim)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(output_path, image)
    finally:
        sim.close()


def save_local_views(episode, output_dir, scene_root, scene_dataset, width, height, max_targets):
    points = []
    start = episode.get("start_position")
    if start is not None:
        points.append(("start", "START", start))
    for target in episode.get("targets", [])[:max_targets]:
        pos = target.get("target_position")
        if pos is None:
            continue
        label = f"{target.get('global_index')}_{target.get('name', 'target')}"
        points.append((f"target_{safe_name(label)}", label, pos))

    sim = make_sim(episode, scene_root, scene_dataset, width, height, "first_person")
    try:
        bounds = sim.pathfinder.get_bounds()
        scene_center = [
            (float(bounds[0][0]) + float(bounds[1][0])) / 2.0,
            0.0,
            (float(bounds[0][2]) + float(bounds[1][2])) / 2.0,
        ]
        output_dir.mkdir(parents=True, exist_ok=True)
        for stem, label, point in points:
            snapped = sim.pathfinder.snap_point(np.asarray(point, dtype=np.float32))
            camera_pos = [float(snapped[0]), float(snapped[1]) + 1.0, float(snapped[2])]
            yaw = yaw_towards(camera_pos, scene_center)
            set_agent_state(sim, camera_pos, yaw=yaw)
            image = render_color(sim)
            imageio.imwrite(output_dir / f"{stem}.png", image)
    finally:
        sim.close()


def save_index(episode, output_path):
    lines = [
        f"# {episode['scene_episode_id']}",
        "",
        f"- scene: `{episode['scene']}`",
        f"- robot: `{episode.get('robot')}`",
        f"- targets: `{len(episode.get('targets', []))}`",
        "",
        "## Targets",
    ]
    for target in episode.get("targets", []):
        lines.append(
            f"- {target.get('global_index')}: {target.get('name')} "
            f"({target.get('region_name')}) from {target.get('source_task_id')}"
        )
    lines.append("")
    lines.append("## Instructions")
    for index, instruction in enumerate(episode.get("instructions", []), start=1):
        wrapped = "\n  ".join(textwrap.wrap(instruction or "", width=96))
        lines.append(f"{index}. {wrapped}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default="data/time_aware_scene/test_episodes.jsonl")
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="output/time_aware_scene/real_views")
    parser.add_argument("--scene", default="data/hm3d/")
    parser.add_argument(
        "--scene-dataset",
        default="data/hm3d/hm3d_annotated_basis.scene_dataset_config.json",
    )
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--meters-per-pixel", type=float, default=0.05)
    parser.add_argument("--max-target-views", type=int, default=8)
    args = parser.parse_args()

    episode = select_episode(load_episodes(args.episodes), args.episode_id, args.seed)
    out_dir = Path(args.output_dir) / safe_name(episode["scene_episode_id"])
    out_dir.mkdir(parents=True, exist_ok=True)

    bounds = get_scene_bounds(episode, args.scene, args.scene_dataset)
    save_navmesh_reference(
        episode,
        out_dir / "navmesh_reference.png",
        args.scene,
        args.scene_dataset,
        args.meters_per_pixel,
    )
    save_topdown_rgb(
        episode,
        out_dir / "topdown_rgb_attempt.png",
        args.scene,
        args.scene_dataset,
        args.width,
        args.height,
        bounds,
    )
    save_local_views(
        episode,
        out_dir / "local_views",
        args.scene,
        args.scene_dataset,
        args.width,
        args.height,
        args.max_target_views,
    )
    save_index(episode, out_dir / "README.md")

    print(f"episode: {episode['scene_episode_id']}")
    print(f"wrote: {out_dir / 'navmesh_reference.png'}")
    print(f"wrote: {out_dir / 'topdown_rgb_attempt.png'}")
    print(f"wrote: {out_dir / 'local_views'}")
    print(f"wrote: {out_dir / 'README.md'}")


if __name__ == "__main__":
    main()
