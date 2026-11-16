"""Overlay YOLO-seg labels on images for visual QA of the converter."""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from cardd import CLASS_NAMES

# src/cardd/viz/overlay.py -> parents[3] is the repository root.
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OVERLAY_DIR = REPO_ROOT / "docs" / "figures" / "overlays"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
ALPHA = 0.4
PALETTE_BGR: tuple[tuple[int, int, int], ...] = (
    (178, 114, 0),
    (0, 159, 230),
    (115, 158, 0),
    (167, 121, 204),
    (66, 51, 213),
    (66, 228, 240),
)


def _read_image(path: Path) -> np.ndarray:
    """Read an image as BGR; ``np.fromfile`` keeps non-ASCII Windows paths working."""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img


def _parse_label_line(line: str) -> tuple[int, np.ndarray] | None:
    parts = line.split()
    if len(parts) < 7 or (len(parts) - 1) % 2:
        return None
    try:
        cls = int(parts[0])
        coords = np.array(parts[1:], dtype=np.float64).reshape(-1, 2)
    except ValueError:
        return None
    return cls, coords


def draw_yolo_labels(image_path: Path, label_path: Path, class_names: Sequence[str]) -> np.ndarray:
    """Draw filled semi-transparent instance masks and class names; returns a BGR image."""
    img = _read_image(Path(image_path))
    h, w = img.shape[:2]
    label_path = Path(label_path)
    lines = label_path.read_text(encoding="utf-8").splitlines() if label_path.exists() else []

    fill = img.copy()
    outlines: list[tuple[np.ndarray, tuple[int, int, int], str]] = []
    for line in lines:
        parsed = _parse_label_line(line)
        if parsed is None:
            continue
        cls, coords = parsed
        pts = np.round(coords * (w, h)).astype(np.int32)
        color = PALETTE_BGR[cls % len(PALETTE_BGR)]
        name = class_names[cls] if 0 <= cls < len(class_names) else str(cls)
        cv2.fillPoly(fill, [pts], color)
        outlines.append((pts, color, name))

    out = cv2.addWeighted(fill, ALPHA, img, 1 - ALPHA, 0)
    scale = max(0.5, min(h, w) / 1000)
    thickness = max(1, round(scale * 2))
    font = cv2.FONT_HERSHEY_SIMPLEX
    for pts, color, name in outlines:
        cv2.polylines(out, [pts], True, color, thickness, cv2.LINE_AA)
        x, y = int(pts[:, 0].min()), int(pts[:, 1].min())
        org = (x, max(y - 4, int(14 * scale)))
        cv2.putText(out, name, org, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(out, name, org, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def save_random_overlays(
    images_dir: Path,
    labels_dir: Path,
    n: int = 30,
    seed: int = 42,
    out_dir: Path = DEFAULT_OVERLAY_DIR,
) -> None:
    """Save overlays for ``n`` randomly chosen images (reproducible via ``seed``)."""
    images = sorted(p for p in Path(images_dir).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    chosen = random.Random(seed).sample(images, min(n, len(images)))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in chosen:
        img = draw_yolo_labels(path, Path(labels_dir) / f"{path.stem}.txt", CLASS_NAMES)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise ValueError(f"Cannot encode overlay for {path}")
        buf.tofile(str(out_dir / f"{path.stem}.jpg"))
