#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def frame_number(path: Path) -> int:
    match = re.search(r"frame_(\d+)_boxes\.jpg$", path.name)
    return int(match.group(1)) if match else -1


def load_font(size: int) -> ImageFont.ImageFont:
    if FONT_PATH.exists():
        return ImageFont.truetype(str(FONT_PATH), size=size)
    return ImageFont.load_default()


def make_sheet_from_paths(paths: list[Path], output_path: Path, columns: int, thumb_width: int) -> bool:
    images = sorted(paths, key=lambda path: (frame_number(path), str(path)))
    if not images:
        print(f"WARNING: no frame_*_boxes.jpg files found for {output_path}; skipped contact sheet")
        return False

    valid_paths = []
    opened = []
    for path in images:
        try:
            opened.append(Image.open(path).convert("RGB"))
            valid_paths.append(path)
        except Exception as exc:
            print(f"WARNING: skipped unreadable image {path}: {exc}")
    if not opened:
        print(f"WARNING: no readable frame_*_boxes.jpg files found for {output_path}; skipped contact sheet")
        return False
    images = valid_paths
    thumb_sizes = []
    for image in opened:
        ratio = thumb_width / image.width
        thumb_sizes.append((thumb_width, max(1, int(round(image.height * ratio)))))

    label_h = 26
    gap = 8
    rows = math.ceil(len(opened) / columns)
    cell_w = thumb_width
    cell_h = max(height for _, height in thumb_sizes) + label_h
    sheet_w = columns * cell_w + (columns - 1) * gap
    sheet_h = rows * cell_h + (rows - 1) * gap
    sheet = Image.new("RGB", (sheet_w, sheet_h), (245, 247, 250))
    draw = ImageDraw.Draw(sheet)
    font = load_font(18)

    for index, (path, image, size) in enumerate(zip(images, opened, thumb_sizes)):
        row = index // columns
        col = index % columns
        x = col * (cell_w + gap)
        y = row * (cell_h + gap)
        thumb = image.resize(size, Image.Resampling.LANCZOS)
        sheet.paste(thumb, (x, y + label_h))
        label = f"{index + 1:02d}  frame {frame_number(path)}"
        draw.rectangle((x, y, x + cell_w, y + label_h), fill=(31, 41, 55))
        draw.text((x + 8, y + 4), label, fill=(255, 255, 255), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    print(f"Saved contact sheet: {output_path}")
    return True


def make_sheet(input_dir: Path, output_path: Path, columns: int, thumb_width: int, *, recursive: bool = False) -> bool:
    pattern = "frame_*_boxes.jpg"
    images = list(input_dir.rglob(pattern) if recursive else input_dir.glob(pattern))
    return make_sheet_from_paths(images, output_path, columns, thumb_width)


def branch_output_path(output_path: Path, branch: str) -> Path:
    if output_path.stem.endswith("_contact_sheet"):
        stem = f"{output_path.stem.removesuffix('_contact_sheet')}_{branch}_contact_sheet"
    else:
        stem = f"{output_path.stem}_{branch}"
    return output_path.with_name(f"{stem}{output_path.suffix}")


def has_frame_box_images(input_dir: Path) -> bool:
    return any(input_dir.glob("frame_*_boxes.jpg"))


def child_frame_box_dirs(input_dir: Path, *, object_branch: bool) -> list[Path]:
    dirs = []
    for path in sorted(input_dir.iterdir()):
        if not path.is_dir():
            continue
        is_object = path.name.endswith("_object")
        if is_object != object_branch:
            continue
        if has_frame_box_images(path):
            dirs.append(path)
    return dirs


def make_branch_sheets(input_dir: Path, output_path: Path, columns: int, thumb_width: int) -> None:
    wrote_any = False
    human_images = list(input_dir.glob("frame_*_boxes.jpg"))
    for human_dir in child_frame_box_dirs(input_dir, object_branch=False):
        human_images.extend(human_dir.glob("frame_*_boxes.jpg"))
    if human_images:
        wrote_any = make_sheet_from_paths(
            human_images,
            branch_output_path(output_path, "human"),
            columns,
            thumb_width,
        ) or wrote_any

    object_dirs = child_frame_box_dirs(input_dir, object_branch=True)
    for index, object_dir in enumerate(object_dirs, start=1):
        branch = "object" if len(object_dirs) == 1 else f"object_{index}"
        object_output = branch_output_path(output_path, branch)
        wrote_any = make_sheet(object_dir, object_output, columns, thumb_width) or wrote_any

    if not wrote_any:
        print(f"WARNING: no branch frame box images found under {input_dir}; skipped contact sheets")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a contact sheet from annotated frame box images.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--thumb-width", type=int, default=320)
    parser.add_argument("--recursive", action="store_true", help="Include frame box images under nested subdirectories.")
    parser.add_argument("--split-branches", action="store_true", help="Write separate human and object contact sheets.")
    args = parser.parse_args()
    if args.split_branches:
        make_branch_sheets(args.input_dir, args.output, max(1, args.columns), max(80, args.thumb_width))
    else:
        make_sheet(args.input_dir, args.output, max(1, args.columns), max(80, args.thumb_width), recursive=args.recursive)


if __name__ == "__main__":
    main()
