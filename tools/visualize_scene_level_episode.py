#!/usr/bin/env python3
"""Visualize scene-level time-aware episodes.

This lightweight visualizer plots the episode start and target points on an
x-z top-down scatter plot. It does not require Habitat rendering, so it is meant
for fast dataset inspection and annotation sanity checks.
"""

import argparse
import json
import random
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt


def load_episodes(path):
    episodes = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                episodes.append(json.loads(line))
    return episodes


def safe_name(name):
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in name)


def xz(position):
    return float(position[0]), float(position[2])


def target_label(target):
    name = target.get("name", "unknown")
    region = target.get("region_name") or "unknown"
    return f"{target.get('global_index')}: {name} ({region})"


def draw_episode(episode, output_path):
    targets_with_pos = [
        target for target in episode.get("targets", [])
        if target.get("target_position") is not None
    ]
    targets_missing = [
        target for target in episode.get("targets", [])
        if target.get("target_position") is None
    ]

    fig = plt.figure(figsize=(15, 8.5), dpi=150)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0])
    ax = fig.add_subplot(grid[0, 0])
    text_ax = fig.add_subplot(grid[0, 1])
    text_ax.axis("off")

    start = episode.get("start_position")
    xs, zs = [], []
    if start is not None:
        sx, sz = xz(start)
        ax.scatter([sx], [sz], marker="*", s=260, c="#d62728", edgecolors="black", linewidths=0.8, label="start")
        ax.annotate("START", (sx, sz), xytext=(6, 6), textcoords="offset points", fontsize=9, weight="bold")
        xs.append(sx)
        zs.append(sz)

    colors = plt.cm.tab10.colors
    for target in targets_with_pos:
        tx, tz = xz(target["target_position"])
        index = int(target.get("global_index", 0))
        color = colors[index % len(colors)]
        ax.scatter([tx], [tz], s=120, c=[color], edgecolors="black", linewidths=0.7)
        ax.annotate(str(index), (tx, tz), xytext=(5, 5), textcoords="offset points", fontsize=10, weight="bold")
        xs.append(tx)
        zs.append(tz)

    if xs and zs:
        pad = max(1.0, 0.12 * max(max(xs) - min(xs), max(zs) - min(zs), 1.0))
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(zs) - pad, max(zs) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    ax.set_title(f"{episode['scene_episode_id']} | scene={episode['scene']}")

    budgets = ", ".join(f"{key}: {value}" for key, value in episode.get("time_budgets", {}).items())
    lines = [
        f"Scene episode: {episode['scene_episode_id']}",
        f"Scene: {episode['scene']}",
        f"Robot: {episode.get('robot')}",
        f"Source tasks: {episode.get('num_source_tasks')} | Targets: {len(episode.get('targets', []))}",
        f"Oracle time proxy: {episode.get('oracle_time_proxy_ordered_sum')}",
        f"Budgets: {budgets}",
        "",
        "Targets:",
    ]
    for target in episode.get("targets", []):
        pos_flag = "pos" if target.get("target_position") is not None else "missing-pos"
        lines.append(f"- {target_label(target)} [{pos_flag}] from {target.get('source_task_id')}")
    if targets_missing:
        lines.append("")
        lines.append(f"Missing target positions: {len(targets_missing)}")
    lines.append("")
    lines.append("Instructions:")
    for index, instruction in enumerate(episode.get("instructions", []), start=1):
        wrapped = textwrap.wrap(instruction or "", width=64)
        lines.append(f"{index}. {wrapped[0] if wrapped else ''}")
        for extra in wrapped[1:]:
            lines.append(f"   {extra}")

    text_ax.text(
        0.0,
        1.0,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=8,
        family="monospace",
        linespacing=1.25,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default="data/time_aware_scene/test_episodes.jsonl")
    parser.add_argument("--output-dir", default="output/time_aware_scene/visualizations")
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--random", type=int, default=0, help="Randomly visualize N episodes.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    episodes = load_episodes(args.episodes)
    if args.episode_id:
        selected = [episode for episode in episodes if episode["scene_episode_id"] == args.episode_id]
        if not selected:
            raise ValueError(f"Episode id not found: {args.episode_id}")
    elif args.random:
        rng = random.Random(args.seed)
        selected = rng.sample(episodes, min(args.random, len(episodes)))
    else:
        selected = episodes[:1]

    output_dir = Path(args.output_dir)
    for episode in selected:
        filename = safe_name(episode["scene_episode_id"]) + ".png"
        output = draw_episode(episode, output_dir / filename)
        print(f"wrote: {output}")


if __name__ == "__main__":
    main()
