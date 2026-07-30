#!/usr/bin/env python3
"""Random selector for NavGPT-style waypoint candidates.

The runner sends a text prompt listing candidate waypoint indices. This script
parses the valid indices and prints one random index. It is intended as a
control baseline for "could the agent complete targets by randomly wandering?".
"""

import argparse
import random
import re
import sys


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prefer-unvisited",
        action="store_true",
        help="Optional ablation: sample recently visited candidates with lower probability.",
    )
    return parser


def parse_candidates(prompt):
    candidates = []
    for line in prompt.splitlines():
        match = re.match(r"\s*-\s*index=(\d+),", line)
        if not match:
            continue
        index = int(match.group(1))
        recently_visited = "exploration=recently visited" in line
        candidates.append((index, recently_visited))
    return candidates


def main():
    args = build_parser().parse_args()
    prompt = sys.stdin.read()
    candidates = parse_candidates(prompt)
    if not candidates:
        print("0")
        return 0

    rng = random.Random(args.seed + abs(hash(prompt)) % 1_000_000_007)
    if args.prefer_unvisited:
        weights = [0.25 if recently_visited else 1.0 for _, recently_visited in candidates]
        print(rng.choices([index for index, _ in candidates], weights=weights, k=1)[0])
    else:
        print(rng.choice([index for index, _ in candidates]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
