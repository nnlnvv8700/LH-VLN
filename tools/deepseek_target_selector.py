#!/usr/bin/env python3
"""Select the next target index with DeepSeek Chat Completions.

The script reads a planner prompt from stdin and prints only the model response
to stdout. It is designed to be called by tools/run_time_aware_llm_planner.py
through --llm-command.
"""

import argparse
import json
import os
import sys

import requests


DEFAULT_SYSTEM_PROMPT = """You are a deterministic target selector for a time-aware unordered multi-target VLN benchmark.
The user prompt lists remaining navigation targets with integer indices and estimated distances.
The targets are unordered. Do not blindly follow the instruction order.
Choose the target that maximizes useful progress under the time budget, usually a reachable nearby target when time is tight.
Return exactly one remaining target index and nothing else.
Do not explain your choice."""


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api-key-env",
        default="DEEPSEEK_API_KEY",
        help="Environment variable that stores the DeepSeek API key.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        help="DeepSeek model name. Can also be set with DEEPSEEK_MODEL.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print the raw DeepSeek response to stderr for debugging.",
    )
    parser.add_argument(
        "--mock-first-index",
        action="store_true",
        help="For local plumbing tests: output the first index in the prompt without calling the API.",
    )
    return parser


def first_index_from_prompt(prompt):
    for line in prompt.splitlines():
        line = line.strip()
        if line.startswith("- index="):
            return line.split("=", 1)[1].split(",", 1)[0].strip()
    return "0"


def chat_completion(args, prompt):
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key. Please export {args.api_key_env}.")

    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    response = requests.post(
        endpoint,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps(payload),
        timeout=args.timeout,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"DeepSeek API error {response.status_code}: {response.text}")

    data = response.json()
    if args.debug:
        print(json.dumps(data, ensure_ascii=False), file=sys.stderr)

    message = data["choices"][0].get("message", {})
    content = message.get("content")
    if content is None:
        content = message.get("reasoning_content", "")
    content = str(content).strip()
    if not content:
        raise RuntimeError(f"DeepSeek returned an empty message: {json.dumps(data, ensure_ascii=False)}")
    return content


def main():
    args = build_parser().parse_args()
    prompt = sys.stdin.read()
    try:
        if args.mock_first_index:
            print(first_index_from_prompt(prompt))
        else:
            print(chat_completion(args, prompt))
    except Exception as exc:
        print(f"deepseek_target_selector_error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
