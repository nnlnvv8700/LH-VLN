# 用途：为已保存的候选视点 RGB 图像调用 RAM 或 mock 标注器，生成视觉文本描述缓存。
#!/usr/bin/env python3
"""Build cached visual text for NavGPT-style waypoint candidate images.

The main waypoint runner can save candidate RGB images. This script converts
those images into a small JSONL cache that maps image paths to visual tags. The
runner can then load the cache with --vision-text-cache and include the tags in
the NavGPT prompt.
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(image_root):
    root = Path(image_root)
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def display_path(path):
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_ram_model(args):
    import torch

    from nav_gen.recognize_anything.ram import get_transform
    from nav_gen.recognize_anything.ram import inference_ram as ram_inference
    from nav_gen.recognize_anything.ram.models import ram

    checkpoint = Path(args.ram_checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"RAM checkpoint not found: {checkpoint}")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    transform = get_transform(image_size=args.image_size)
    model = ram(pretrained=str(checkpoint), image_size=args.image_size, vit=args.ram_vit)
    model.eval()
    model = model.to(device)
    return device, transform, model, ram_inference


def ram_tags(path, runtime):
    device, transform, model, ram_inference = runtime
    image = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    result = ram_inference(image, model)
    english = result[0]
    if isinstance(english, str):
        tags = [item.strip() for item in english.replace("|", ",").split(",") if item.strip()]
    else:
        tags = [str(item).strip() for item in english if str(item).strip()]
    return tags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", required=True, help="Directory containing saved candidate RGB images.")
    parser.add_argument("--output", required=True, help="Output JSONL cache path.")
    parser.add_argument("--provider", choices=["ram", "mock"], default="ram")
    parser.add_argument(
        "--ram-checkpoint",
        default="/file_system/nas/algorithm/Intern03/models/recognize_anything/ram_swin_large_14m.pth",
    )
    parser.add_argument("--ram-vit", default="swin_l")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--device", default=None)
    parser.add_argument("--mock-tags", default="indoor scene")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    image_paths = list(iter_images(args.image_root))
    if args.limit:
        image_paths = image_paths[: args.limit]
    runtime = load_ram_model(args) if args.provider == "ram" else None

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index, path in enumerate(image_paths, start=1):
            if args.provider == "ram":
                tags = ram_tags(path, runtime)
            else:
                tags = [item.strip() for item in args.mock_tags.split(",") if item.strip()]
            row = {
                "image": display_path(path),
                "provider": args.provider,
                "tags": tags,
                "caption": "",
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{index}/{len(image_paths)}] {row['image']} -> {', '.join(tags[:8])}")


if __name__ == "__main__":
    main()
