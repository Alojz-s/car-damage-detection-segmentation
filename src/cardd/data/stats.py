"""Dataset statistics (class distribution, mask areas) for EDA.

Loaders read the raw CarDD COCO annotations (not the converted YOLO labels).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from cardd import CLASS_NAMES

TIRE_FLAT = "tire_flat"
SPLIT_TO_COCO: dict[str, str] = {"train": "train2017", "val": "val2017", "test": "test2017"}

INSTANCE_COLUMNS = [
    "split",
    "image_id",
    "file_name",
    "class_name",
    "area_px",
    "area_frac",
    "bbox_w",
    "bbox_h",
    "n_polygons",
]
IMAGE_COLUMNS = ["split", "image_id", "file_name", "width", "height", "n_instances"]


@lru_cache(maxsize=8)
def _load_coco(path: str, mtime_ns: int) -> dict[str, Any]:
    """Parse a COCO JSON once per (path, mtime); callers must not mutate the result."""
    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return data


def _read_split(
    coco_dir: Path, split: str, drop_tire_flat: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (image rows, instance rows) for one split."""
    if split not in SPLIT_TO_COCO:
        raise ValueError(f"Unknown split {split!r}; expected one of {tuple(SPLIT_TO_COCO)}")
    path = Path(coco_dir) / "annotations" / f"instances_{SPLIT_TO_COCO[split]}.json"
    coco = _load_coco(str(path.resolve()), path.stat().st_mtime_ns)

    cat_names = {c["id"]: str(c["name"]).replace(" ", "_") for c in coco["categories"]}
    images = {img["id"]: img for img in coco["images"]}
    counts: dict[int, int] = dict.fromkeys(images, 0)
    instances: list[dict[str, Any]] = []

    for ann in coco.get("annotations", []):
        name = cat_names[ann["category_id"]]
        if drop_tire_flat and name == TIRE_FLAT:
            continue
        img = images[ann["image_id"]]
        seg = ann.get("segmentation")
        area_px = float(ann["area"])
        _, _, w, h = ann["bbox"]
        counts[img["id"]] += 1
        instances.append(
            {
                "split": split,
                "image_id": img["id"],
                "file_name": img["file_name"],
                "class_name": name,
                "area_px": area_px,
                "area_frac": area_px / (img["width"] * img["height"]),
                "bbox_w": float(w),
                "bbox_h": float(h),
                "n_polygons": len(seg) if isinstance(seg, list) else 0,
            }
        )

    image_rows = [
        {
            "split": split,
            "image_id": img["id"],
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"],
            "n_instances": counts[img["id"]],
        }
        for img in coco["images"]
    ]
    return image_rows, instances


def load_instances(
    coco_dir: Path,
    splits: tuple[str, ...] = ("train", "val", "test"),
    drop_tire_flat: bool = True,
) -> pd.DataFrame:
    """Load one row per COCO instance; class names use underscores (``glass_shatter``)."""
    rows: list[dict[str, Any]] = []
    for split in splits:
        rows.extend(_read_split(coco_dir, split, drop_tire_flat)[1])
    return pd.DataFrame(rows, columns=INSTANCE_COLUMNS)


def load_images(
    coco_dir: Path,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> pd.DataFrame:
    """Load one row per image; ``n_instances`` excludes ``tire flat`` like ``load_instances``."""
    rows: list[dict[str, Any]] = []
    for split in splits:
        rows.extend(_read_split(coco_dir, split, drop_tire_flat=True)[0])
    return pd.DataFrame(rows, columns=IMAGE_COLUMNS)


def class_order(instances: pd.DataFrame) -> list[str]:
    """Project classes first, then any extra classes (e.g. ``tire_flat``) as they appear."""
    extra = [c for c in instances["class_name"].unique() if c not in CLASS_NAMES]
    return [*CLASS_NAMES, *extra]


def class_distribution(instances: pd.DataFrame) -> pd.DataFrame:
    """Instance and image counts for every split x class (zeros included).

    Columns: ``split``, ``class_name``, ``n_instances``, ``n_images``.
    """
    classes = class_order(instances)
    splits = list(dict.fromkeys(instances["split"]))
    grouped = instances.groupby(["split", "class_name"])
    n_inst = grouped.size()
    n_img = grouped["image_id"].nunique()
    rows = [
        {
            "split": s,
            "class_name": c,
            "n_instances": int(n_inst.get((s, c), 0)),
            "n_images": int(n_img.get((s, c), 0)),
        }
        for s in splits
        for c in classes
    ]
    return pd.DataFrame(rows, columns=["split", "class_name", "n_instances", "n_images"])


def imbalance_ratio(instances: pd.DataFrame) -> float:
    """Max / min instance count over project classes on the train split.

    Returns ``inf`` when a class has no train instances and ``nan`` when train is empty.
    """
    train = instances[instances["split"] == "train"]
    counts = train["class_name"].value_counts().reindex(list(CLASS_NAMES), fill_value=0)
    if counts.max() == 0:
        return float("nan")
    if counts.min() == 0:
        return float("inf")
    return float(counts.max() / counts.min())
