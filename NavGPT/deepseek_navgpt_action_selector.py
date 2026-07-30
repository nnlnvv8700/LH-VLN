#!/usr/bin/env python3
"""DeepSeek action selector for the NavGPT-Habitat adapter.

The script reads a NavGPT-style prompt from stdin and prints the raw model
response. The runner parses either:
  Action Input: "move_forward x 3"
or:
  Final Answer: Finished!
"""

import argparse
import json
import os
import sys

import requests


DEFAULT_SYSTEM_PROMPT = """You are NavGPT, an embodied navigation agent.
You must choose exactly one valid action from the prompt.
Use the requested format:
Thought: ...
Action: action_maker
Action Input: "move_forward x 3"

If you believe the current location satisfies a target, output:
Thought: ...
Final Answer: Finished!

Valid Action Input values are: move_forward, turn_left, turn_right, stop.
You may repeat move_forward/turn_left/turn_right with a small integer, e.g. "move_forward x 3".
Do not invent other actions."""


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument(
        "--mock-action",
        choices=["move_forward", "turn_left", "turn_right", "stop"],
        default=None,
        help="Local plumbing test: print a fixed NavGPT-style action without calling the API.",
    )
    parser.add_argument("--mock-repeat", type=int, default=1)
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
    prompt = sys.stdin.read()
    try:
        if args.mock_action:
            if args.mock_action == "stop":
                print('Thought: I will check whether a target is completed.\nFinal Answer: Finished!')
            else:
                repeat = max(1, int(args.mock_repeat))
                print(f'Thought: I will continue navigation.\nAction: action_maker\nAction Input: "{args.mock_action} x {repeat}"')
        else:
            print(chat_completion(args, prompt))
    except Exception as exc:
        print(f"deepseek_navgpt_action_selector_error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
