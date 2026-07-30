# 用途：提供通用 VLM 局部候选视点选择接口，负责构造多视图提示、调用外部模型并解析候选索引。
#!/usr/bin/env python3
"""OpenAI-compatible multimodal selector for waypoint candidates.

The script reads the NavGPT-style prompt from stdin, extracts candidate RGB
image paths from `rgb_image=...`, sends the prompt plus images to a vision
language model, and prints the raw model response. The runner then parses a
candidate index from stdout.

This keeps the original text-only DeepSeek selector untouched. Use this script
only when the target API/model supports OpenAI-style multimodal chat messages.
"""

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
from pathlib import Path

import requests


REPO_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_SYSTEM_PROMPT = """You are an embodied navigation agent.
You choose among candidate viewpoint images in an indoor scene.

The user message contains:
- task instruction and progress
- fuzzy time condition
- candidate metadata
- candidate images, where image N corresponds to candidate index N

Return exactly one candidate index.
Do not return STOP.
Do not return low-level actions.
Do not explain your choice."""


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key-env", default="VLM_API_KEY")
    parser.add_argument("--base-url", default=os.environ.get("VLM_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--model", default=os.environ.get("VLM_MODEL", "gpt-4o-mini"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--image-detail", choices=["auto", "low", "high"], default="low")
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument(
        "--mock-choice",
        default=None,
        help="Local plumbing test: print a fixed candidate index without calling the API.",
    )
    return parser


def resolve_image_path(path_text):
    if not path_text or path_text == "not saved":
        return None
    path = Path(path_text.strip())
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path if path.exists() else None


def extract_candidate_images(prompt):
    images = []
    for line in prompt.splitlines():
        index_match = re.match(r"\s*-\s*index=(\d+),", line)
        image_match = re.search(r"rgb_image=([^,\n]+(?:\.jpg|\.jpeg|\.png)?)", line)
        if not index_match or not image_match:
            continue
        path = resolve_image_path(image_match.group(1).strip())
        if path is not None:
            images.append((int(index_match.group(1)), path))
    return images


def image_to_data_url(path):
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def build_user_content(prompt, images, detail, max_images):
    content = [{"type": "text", "text": prompt}]
    if images:
        legend = "\n".join(f"Image {index} corresponds to candidate index {index}." for index, _ in images[:max_images])
        content.append({"type": "text", "text": "Candidate image mapping:\n" + legend})
    for index, path in images[:max_images]:
        content.append({"type": "text", "text": f"Image {index}: candidate index {index}"})
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_to_data_url(path),
                    "detail": detail,
                },
            }
        )
    return content


def chat_completion(args, prompt):
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing API key. Please export {args.api_key_env}.")
    try:
        api_key.encode("ascii")
    except UnicodeEncodeError as exc:
        raise RuntimeError(f"{args.api_key_env} contains non-ASCII characters.") from exc

    images = extract_candidate_images(prompt)
    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": build_user_content(prompt, images, args.image_detail, args.max_images)},
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
        raise RuntimeError(f"VLM API error {response.status_code}: {response.text}")
    data = response.json()
    return str(data["choices"][0].get("message", {}).get("content", "")).strip()


def main():
    args = build_parser().parse_args()
    prompt = sys.stdin.read()
    try:
        if args.mock_choice is not None:
            print(args.mock_choice)
        else:
            print(chat_completion(args, prompt))
    except Exception as exc:
        print(f"vlm_waypoint_selector_error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
