"""Tests for EDA loaders, plots and overlays (synthetic data only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pytest

from cardd import CLASS_NAMES
from cardd.data.stats import (
    class_distribution,
    imbalance_ratio,
    load_images,
    load_instances,
)
from cardd.viz.eda import plot_area_distribution, plot_class_distribution
from cardd.viz.labels import LABELS
from cardd.viz.overlay import draw_yolo_labels, save_random_overlays

CATS = [
    {"id": 1, "name": "dent"},
    {"id": 2, "name": "scratch"},
    {"id": 3, "name": "crack"},
    {"id": 4, "name": "glass shatter"},
    {"id": 5, "name": "lamp broken"},
    {"id": 6, "name": "tire flat"},
]
SQUARE: list[float] = [10, 10, 50, 10, 50, 50, 10, 50]  # 40 x 40 px


def _ann(i: int, image_id: int, cat: int, area: float = 1600.0) -> dict[str, Any]:
    return {
        "id": i,
        "image_id": image_id,
        "category_id": cat,
        "segmentation": [SQUARE],
        "area": area,
        "bbox": [10, 10, 40, 40],
        "iscrowd": 0,
    }


def _write_split(root: Path, split: str, images: list[int], anns: list[dict[str, Any]]) -> None:
    ann_dir = root / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    coco = {
        "images": [
            {"id": i, "file_name": f"{split}_{i}.jpg", "width": 100, "height": 100} for i in images
        ],
        "annotations": anns,
        "categories": CATS,
    }
    (ann_dir / f"instances_{split}2017.json").write_text(json.dumps(coco), encoding="utf-8")


@pytest.fixture
def coco_dir(tmp_path: Path) -> Path:
    root = tmp_path / "coco"
    _write_split(
        root,
        "train",
        [1, 2],
        [
            _ann(1, 1, 1),
            _ann(2, 1, 2),
            _ann(3, 1, 6),  # tire flat
            _ann(4, 2, 1),
            _ann(5, 2, 1),
            _ann(6, 2, 3),
            _ann(7, 2, 4),
            _ann(8, 2, 5),
        ],
    )
    _write_split(root, "val", [3], [_ann(9, 3, 2), _ann(10, 3, 6)])
    _write_split(root, "test", [4, 5], [_ann(11, 4, 5)])
    return root


@pytest.fixture
def instances(coco_dir: Path) -> pd.DataFrame:
    return load_instances(coco_dir)


def test_load_instances_counts(instances: pd.DataFrame) -> None:
    assert len(instances) == 7 + 1 + 1  # train + val + test, tire flat excluded
    train = instances[instances["split"] == "train"]["class_name"].value_counts()
    assert train["dent"] == 3
    assert train["scratch"] == 1
    assert instances[instances["split"] == "val"]["class_name"].tolist() == ["scratch"]
    assert instances[instances["split"] == "test"]["class_name"].tolist() == ["lamp_broken"]


def test_area_frac_and_polygons(instances: pd.DataFrame) -> None:
    assert np.allclose(instances["area_frac"], 1600 / (100 * 100))
    assert (instances["n_polygons"] == 1).all()
    assert (instances["bbox_w"] == 40).all()


def test_drop_tire_flat(coco_dir: Path) -> None:
    assert "tire_flat" not in set(load_instances(coco_dir)["class_name"])
    raw = load_instances(coco_dir, drop_tire_flat=False)
    assert (raw["class_name"] == "tire_flat").sum() == 2
    assert set(class_distribution(raw)["class_name"]) == {*CLASS_NAMES, "tire_flat"}


def test_split_subset_and_unknown(coco_dir: Path) -> None:
    assert set(load_instances(coco_dir, splits=("val",))["split"]) == {"val"}
    with pytest.raises(ValueError):
        load_instances(coco_dir, splits=("holdout",))


def test_load_images(coco_dir: Path) -> None:
    images = load_images(coco_dir)
    assert list(images.columns) == [
        "split",
        "image_id",
        "file_name",
        "width",
        "height",
        "n_instances",
    ]
    assert len(images) == 5
    by_id = images.set_index(["split", "image_id"])["n_instances"]
    assert by_id[("train", 1)] == 2  # tire flat not counted
    assert by_id[("val", 3)] == 1
    assert by_id[("test", 5)] == 0


def test_class_distribution(instances: pd.DataFrame) -> None:
    dist = class_distribution(instances)
    assert list(dist.columns) == ["split", "class_name", "n_instances", "n_images"]
    assert len(dist) == 3 * len(CLASS_NAMES)
    row = dist[(dist["split"] == "train") & (dist["class_name"] == "dent")].iloc[0]
    assert (row["n_instances"], row["n_images"]) == (3, 2)
    zero = dist[(dist["split"] == "val") & (dist["class_name"] == "dent")].iloc[0]
    assert zero["n_instances"] == 0


def test_imbalance_ratio(instances: pd.DataFrame) -> None:
    assert imbalance_ratio(instances) == 3.0
    assert imbalance_ratio(instances[instances["class_name"] != "crack"]) == float("inf")
    assert np.isnan(imbalance_ratio(instances[instances["split"] != "train"]))


def test_plots_save_files(instances: pd.DataFrame, tmp_path: Path) -> None:
    for lang in ("pl", "en"):
        out = tmp_path / f"dist_{lang}.png"
        fig = plot_class_distribution(class_distribution(instances), lang=lang, out=out)
        assert out.stat().st_size > 0
        assert fig.axes[0].get_title() == LABELS[lang]["class_dist_title"]
        area = plot_area_distribution(instances, lang=lang, out=tmp_path / f"area_{lang}.png")
        assert area.axes[0].get_yscale() == "log"
    assert plot_class_distribution(instances, lang="en") is not None


def _make_image(path: Path, size: tuple[int, int] = (80, 60)) -> None:
    cv2.imwrite(str(path), np.full((size[1], size[0], 3), 120, dtype=np.uint8))


def test_draw_yolo_labels(tmp_path: Path) -> None:
    img_path, lbl_path = tmp_path / "a.jpg", tmp_path / "a.txt"
    _make_image(img_path)
    lbl_path.write_text("1 0.1 0.1 0.6 0.1 0.6 0.6 0.1 0.6\nbad line\n", encoding="utf-8")
    out = draw_yolo_labels(img_path, lbl_path, CLASS_NAMES)
    assert out.shape == (60, 80, 3)
    assert out.dtype == np.uint8
    plain = draw_yolo_labels(img_path, tmp_path / "missing.txt", CLASS_NAMES)
    assert not np.array_equal(out, plain)


def test_save_random_overlays_is_reproducible(tmp_path: Path) -> None:
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    for i in range(6):
        _make_image(images / f"img{i}.jpg")
        (labels / f"img{i}.txt").write_text("0 0.1 0.1 0.5 0.1 0.5 0.5\n", encoding="utf-8")
    save_random_overlays(images, labels, n=3, seed=7, out_dir=tmp_path / "a")
    save_random_overlays(images, labels, n=3, seed=7, out_dir=tmp_path / "b")
    names_a = sorted(p.name for p in (tmp_path / "a").iterdir())
    assert len(names_a) == 3
    assert names_a == sorted(p.name for p in (tmp_path / "b").iterdir())


def test_area_plot_class_set_matches_tire_flat_policy(coco_dir: Path) -> None:
    default = plot_area_distribution(load_instances(coco_dir))
    raw = plot_area_distribution(load_instances(coco_dir, drop_tire_flat=False))
    assert len(default.axes[0].get_xticklabels()) == 5
    assert len(raw.axes[0].get_xticklabels()) == 6


def test_coco_cache_sees_rewritten_file(coco_dir: Path) -> None:
    assert len(load_instances(coco_dir, splits=("val",))) == 1
    _write_split(coco_dir, "val", [3], [_ann(9, 3, 2), _ann(10, 3, 1)])
    assert len(load_instances(coco_dir, splits=("val",))) == 2
