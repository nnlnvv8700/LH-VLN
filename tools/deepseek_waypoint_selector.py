# 用途：将局部候选视点的视觉文本、任务和历史发送给 DeepSeek，解析其选择的候选点索引。
#!/usr/bin/env python3
"""DeepSeek selector for NavGPT-style Habitat waypoint candidates.

The script reads a prompt from stdin and prints the raw model response. The
runner accepts either an integer candidate index or STOP.
"""

import argparse
import json
import os
import sys

import requests


DEFAULT_SYSTEM_PROMPT = """You are NavGPT, an embodied navigation agent.
You choose among candidate viewpoints in a Habitat indoor scene.
The prompt lists candidate waypoint indices and a STOP option.

Return exactly one candidate index, or STOP if the agent should verify a nearby target.
Do not return low-level actions such as move_forward or turn_left.
Do not explain your choice."""


NO_STOP_SYSTEM_PROMPT = """You are NavGPT, an embodied navigation agent.
You choose among candidate viewpoints in a Habitat indoor scene.
The prompt lists candidate waypoint indices.

Return exactly one candidate index.
Do not return STOP.
Do not return low-level actions such as move_forward or turn_left.
Do not explain your choice."""


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--no-stop", action="store_true", help="Use a system prompt that forbids STOP.")
    parser.add_argument(
        "--mock-choice",
        default=None,
        help="Local plumbing test: print a fixed candidate index or STOP without calling the API.",
    )
    return parser


def chat_completion(args, prompt):
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing API key. Please export {args.api_key_env}.")
    try:
        api_key.encode("ascii")
    except UnicodeEncodeError as exc:
        raise RuntimeError(
            f"{args.api_key_env} contains non-ASCII characters. "
            "Please export the real key, e.g. export DEEPSEEK_API_KEY=\"sk-...\"."
        ) from exc

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
    content = data["choices"][0].get("message", {}).get("content", "")
    return str(content).strip()


def main():
    args = build_parser().parse_args()
    if args.no_stop and args.system_prompt == DEFAULT_SYSTEM_PROMPT:
        args.system_prompt = NO_STOP_SYSTEM_PROMPT
    prompt = sys.stdin.read()
    try:
        if args.mock_choice is not None:
            print(args.mock_choice)
        else:
            print(chat_completion(args, prompt))
    except Exception as exc:
        print(f"deepseek_waypoint_selector_error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
