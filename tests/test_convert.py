"""Tests for the COCO -> YOLO-seg converter (synthetic data only)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from PIL import Image

from cardd import CLASS_NAMES
from cardd.data import convert as conv
from cardd.data.convert import (
    COCO_NAME_TO_YOLO_ID,
    coco_polygons_to_yolo,
    convert,
    convert_split,
    main,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CATS = [
    {"id": 1, "name": "dent"},
    {"id": 2, "name": "scratch"},
    {"id": 3, "name": "crack"},
    {"id": 4, "name": "glass shatter"},
    {"id": 5, "name": "lamp broken"},
    {"id": 6, "name": "tire flat"},
]
SQUARE: list[float] = [10, 10, 50, 10, 50, 50, 10, 50]
TRIANGLE: list[float] = [60, 10, 90, 10, 90, 40]


def _ann(i: int, image_id: int, cat: int, seg: Any, **extra: Any) -> dict[str, Any]:
    return {
        "id": i,
        "image_id": image_id,
        "category_id": cat,
        "segmentation": seg,
        "iscrowd": 0,
        **extra,
    }


def _default_anns() -> list[dict[str, Any]]:
    return [
        _ann(1, 1, 4, [SQUARE, TRIANGLE]),  # glass shatter, two parts
        _ann(2, 1, 1, [[-10, -10, 200, -10, 200, 100]]),  # dent (COCO id 1), clipped
        _ann(3, 2, 6, [SQUARE]),  # tire-only image
        _ann(4, 3, 5, [[1, 1, 2, 2]]),  # degenerate polygon only
    ]


def _make_raw(root: Path, anns: list[dict[str, Any]] | None = None) -> Path:
    raw = root / "raw"
    (raw / "annotations").mkdir(parents=True)
    for split in ("train2017", "val2017", "test2017"):
        (raw / split).mkdir()
        images = []
        for i in (1, 2, 3):
            Image.new("RGB", (100, 50), (i, i, i)).save(raw / split / f"img{i}.jpg")
            images.append({"id": i, "file_name": f"img{i}.jpg", "width": 100, "height": 50})
        data = {
            "images": images,
            "annotations": _default_anns() if anns is None else anns,
            "categories": CATS,
        }
        (raw / "annotations" / f"instances_{split}.json").write_text(json.dumps(data))
    return raw


def _split(raw: Path, out: Path, **kw: Any) -> conv.SplitStats:
    return convert_split(
        raw / "annotations" / "instances_train2017.json",
        raw / "train2017",
        out / "images",
        out / "labels",
        **kw,
    )


def _lines(path: Path) -> list[str]:
    return path.read_text().splitlines()


# --- class mapping -------------------------------------------------------------------------


def test_class_mapping_derived_from_class_names() -> None:
    assert COCO_NAME_TO_YOLO_ID == {
        "dent": 0,
        "scratch": 1,
        "crack": 2,
        "glass shatter": 3,
        "lamp broken": 4,
    }


def test_data_yaml_names_match_class_names() -> None:
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "data.yaml").read_text())
    assert tuple(cfg["names"][i] for i in range(len(cfg["names"]))) == CLASS_NAMES


def test_class_reindex(tmp_path: Path) -> None:
    raw = _make_raw(tmp_path)
    _split(raw, tmp_path / "o")
    ids = sorted(int(line.split()[0]) for line in _lines(tmp_path / "o/labels/img1.txt"))
    assert ids == [COCO_NAME_TO_YOLO_ID["dent"], 3]


# --- polygons ------------------------------------------------------------------------------


def test_normalization_and_clipping() -> None:
    out = coco_polygons_to_yolo([SQUARE, [-10, -10, 200, -10, 200, 100]], 100, 50)
    assert out[0] == pytest.approx([0.1, 0.2, 0.5, 0.2, 0.5, 1.0, 0.1, 1.0])
    assert out[1] == pytest.approx([0, 0, 1, 0, 1, 1])


def test_degenerate_polygons_dropped() -> None:
    assert coco_polygons_to_yolo([[1, 1, 2, 2], [0, 0, 5, 5, 10, 10]], 100, 100) == []


def test_polygon_outside_image_dropped() -> None:
    assert coco_polygons_to_yolo([[150, 10, 200, 10, 200, 40]], 100, 50) == []


def test_polygon_collapsing_on_edge_after_clipping_dropped() -> None:
    # Becomes a line along x == 1 once clipped.
    assert coco_polygons_to_yolo([[100, 0, 150, 20, 120, 50]], 100, 50) == []


@pytest.mark.parametrize(("w", "h"), [(0, 10), (10, 0), (-1, 5)])
def test_invalid_size_raises(w: int, h: int) -> None:
    with pytest.raises(ValueError):
        coco_polygons_to_yolo([SQUARE], w, h)


def test_invalid_image_size_in_split_names_image(tmp_path: Path) -> None:
    raw = _make_raw(tmp_path)
    path = raw / "annotations" / "instances_train2017.json"
    data = json.loads(path.read_text())
    data["images"][0]["width"] = 0
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="img1.jpg"):
        _split(raw, tmp_path / "o")


# --- multipart -----------------------------------------------------------------------------


def test_multipart_merge(tmp_path: Path) -> None:
    raw = _make_raw(tmp_path)
    stats = _split(raw, tmp_path / "o")  # merge is the default
    lines = _lines(tmp_path / "o/labels/img1.txt")
    glass = [line for line in lines if line.startswith("3 ")]
    assert len(glass) == 1
    coords = [float(v) for v in glass[0].split()[1:]]
    assert len(coords) % 2 == 0 and len(coords) // 2 >= 3
    assert all(0.0 <= v <= 1.0 for v in coords)
    assert stats.multipart_instances == 1
    assert sum(stats.instances_per_class.values()) == stats.label_lines == 2


def test_multipart_split(tmp_path: Path) -> None:
    raw = _make_raw(tmp_path)
    stats = _split(raw, tmp_path / "o", multipart="split")
    lines = _lines(tmp_path / "o/labels/img1.txt")
    assert len([line for line in lines if line.startswith("3 ")]) == 2
    assert stats.multipart_instances == 1
    assert stats.instances_per_class["glass_shatter"] == 1
    assert stats.label_lines == 3
    assert sum(stats.instances_per_class.values()) == 2


# --- tire-only / empty images --------------------------------------------------------------


def test_tire_only_and_degenerate_kept(tmp_path: Path) -> None:
    raw = _make_raw(tmp_path)
    stats = _split(raw, tmp_path / "o")
    assert stats.images == 3
    assert stats.images_emptied == 1  # tire-only only
    assert stats.images_without_labels == 2  # tire-only + degenerate-only
    assert stats.skipped_instances == 1
    assert (tmp_path / "o/labels/img2.txt").read_text() == ""
    assert (tmp_path / "o/images/img2.jpg").exists()


def test_drop_empty_removes_all_images_without_labels(tmp_path: Path) -> None:
    raw = _make_raw(tmp_path)
    stats = _split(raw, tmp_path / "o", keep_empty=False)
    assert stats.images == 1
    assert stats.images_emptied == 1
    assert stats.images_without_labels == 2
    assert stats.skipped_instances == 1  # still counted for dropped images
    for name in ("img2", "img3"):
        assert not (tmp_path / f"o/images/{name}.jpg").exists()
        assert not (tmp_path / f"o/labels/{name}.txt").exists()
    assert (tmp_path / "o/images/img1.jpg").exists()


# --- skipped annotations -------------------------------------------------------------------


def test_rle_crowd_and_unknown_category_skipped(tmp_path: Path) -> None:
    anns = [
        _ann(1, 1, 1, {"counts": [0, 5], "size": [50, 100]}),  # RLE dict
        _ann(2, 1, 2, [SQUARE], iscrowd=1),
        _ann(3, 1, 99, [SQUARE]),  # unknown category
        _ann(4, 1, 3, [SQUARE]),  # valid
    ]
    raw = _make_raw(tmp_path, anns)
    stats = _split(raw, tmp_path / "o")
    assert stats.skipped_instances == 3
    assert stats.instances_per_class["crack"] == 1
    assert len(_lines(tmp_path / "o/labels/img1.txt")) == 1


# --- file names ----------------------------------------------------------------------------


def test_original_names_kept(tmp_path: Path) -> None:
    raw = _make_raw(tmp_path)
    _split(raw, tmp_path / "keep")
    assert sorted(p.name for p in (tmp_path / "keep/images").iterdir()) == [
        "img1.jpg",
        "img2.jpg",
        "img3.jpg",
    ]
    assert sorted(p.name for p in (tmp_path / "keep/labels").iterdir()) == [
        "img1.txt",
        "img2.txt",
        "img3.txt",
    ]


def test_duplicate_file_names_rejected(tmp_path: Path) -> None:
    raw = _make_raw(tmp_path)
    path = raw / "annotations" / "instances_train2017.json"
    data = json.loads(path.read_text())
    data["images"][1]["file_name"] = "sub/img1.jpg"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Duplicate"):
        _split(raw, tmp_path / "o")


# --- convert(): layout, stats, mapping -----------------------------------------------------


def test_convert_layout_stats_and_mapping(tmp_path: Path) -> None:
    raw = _make_raw(tmp_path)
    out = tmp_path / "out"
    result = convert(raw, out)
    assert set(result) == {"train", "val", "test"}
    for split in ("train", "val", "test"):
        assert (out / "images" / split / "img1.jpg").exists()
        assert (out / "labels" / split / "img1.txt").exists()
        with (out / f"{split}_images.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        assert [r["file_name"] for r in rows] == ["img1.jpg", "img2.jpg", "img3.jpg"]
        assert rows[0]["n_label_lines"] == "2"
        assert rows[0]["width"] == "100" and rows[0]["image_id"] == "1"
    stats = json.loads((out / "conversion_stats.json").read_text())
    assert set(stats) == {"train", "val", "test"}
    assert stats["val"]["label_lines"] == 2
    assert stats["val"]["images_emptied"] == 1
    assert stats["val"]["images_without_labels"] == 2
    assert stats["val"]["instances_per_class"] == {
        "dent": 1,
        "scratch": 0,
        "crack": 0,
        "glass_shatter": 1,
        "lamp_broken": 0,
    }
    assert not (tmp_path / "out.tmp").exists()


def test_convert_drop_empty_mapping_has_row_per_written_image(tmp_path: Path) -> None:
    raw = _make_raw(tmp_path)
    out = tmp_path / "out"
    convert(raw, out, keep_empty=False)
    with (out / "train_images.csv").open(newline="") as f:
        assert len(list(csv.DictReader(f))) == 1


def test_convert_refuses_overwrite(tmp_path: Path) -> None:
    raw = _make_raw(tmp_path)
    out = tmp_path / "out"
    convert(raw, out)
    with pytest.raises(FileExistsError):
        convert(raw, out)


def test_missing_raw_dir_message(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    with pytest.raises(FileNotFoundError, match="nope"):
        convert(missing, tmp_path / "out")
    assert not (tmp_path / "out").exists()


# --- safety --------------------------------------------------------------------------------


def test_force_replaces_previous_output(tmp_path: Path) -> None:
    raw = _make_raw(tmp_path)
    out = tmp_path / "out"
    convert(raw, out)
    (out / "stale.txt").write_text("old")
    convert(raw, out, force=True)
    assert not (out / "stale.txt").exists()
    assert (out / "conversion_stats.json").exists()
    assert not (tmp_path / "out.tmp").exists()


def test_failed_conversion_keeps_existing_output(tmp_path: Path) -> None:
    raw = _make_raw(tmp_path)
    out = tmp_path / "out"
    convert(raw, out)
    (out / "marker.txt").write_text("keep me")
    (raw / "val2017" / "img2.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="img2.jpg"):
        convert(raw, out, force=True)
    assert (out / "marker.txt").read_text() == "keep me"
    assert (out / "conversion_stats.json").exists()
    assert not (tmp_path / "out.tmp").exists()


def test_error_during_conversion_removes_tmp(tmp_path: Path) -> None:
    raw = _make_raw(tmp_path)
    path = raw / "annotations" / "instances_test2017.json"
    data = json.loads(path.read_text())
    data["images"][1]["file_name"] = "img1.jpg"  # duplicate, found after train/val were written
    path.write_text(json.dumps(data))
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="Duplicate"):
        convert(raw, out)
    assert not out.exists()
    assert not (tmp_path / "out.tmp").exists()


def test_missing_images_reported_with_count(tmp_path: Path) -> None:
    raw = _make_raw(tmp_path)
    for split in ("train2017", "val2017", "test2017"):
        (raw / split / "img1.jpg").unlink()
        (raw / split / "img2.jpg").unlink()
    with pytest.raises(FileNotFoundError, match=r"6 required input"):
        convert(raw, tmp_path / "out")


def test_force_refuses_foreign_directory(tmp_path: Path) -> None:
    raw = _make_raw(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "precious.txt").write_text("data")
    with pytest.raises(ValueError, match="Refusing"):
        convert(raw, out, force=True)
    assert (out / "precious.txt").read_text() == "data"


def test_force_accepts_empty_directory(tmp_path: Path) -> None:
    raw = _make_raw(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    convert(raw, out, force=True)
    assert (out / "conversion_stats.json").exists()


# --- hardlink ------------------------------------------------------------------------------


def test_hardlink_same_content(tmp_path: Path) -> None:
    raw = _make_raw(tmp_path)
    out = tmp_path / "out"
    convert(raw, out, link="hardlink")
    assert (out / "images/train/img1.jpg").read_bytes() == (raw / "train2017/img1.jpg").read_bytes()


def test_hardlink_falls_back_to_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("cross-device")

    monkeypatch.setattr(conv.os, "link", fail)
    raw = _make_raw(tmp_path)
    out = tmp_path / "out"
    convert(raw, out, link="hardlink")
    assert (out / "images/val/img3.jpg").read_bytes() == (raw / "val2017/img3.jpg").read_bytes()


# --- CLI -----------------------------------------------------------------------------------


def test_main_defaults(tmp_path: Path) -> None:
    raw = _make_raw(tmp_path)
    out = tmp_path / "out"
    main(["--raw", str(raw), "--out", str(out)])
    assert len(_lines(out / "labels/train/img1.txt")) == 2
    assert (out / "images/train/img2.jpg").exists()


def test_main_drop_empty_force_split_hardlink(tmp_path: Path) -> None:
    raw = _make_raw(tmp_path)
    out = tmp_path / "out"
    main(["--raw", str(raw), "--out", str(out)])
    main(
        [
            "--raw",
            str(raw),
            "--out",
            str(out),
            "--force",
            "--drop-empty",
            "--multipart",
            "split",
            "--link",
            "hardlink",
        ]
    )
    assert len(_lines(out / "labels/train/img1.txt")) == 3
    assert not (out / "images/train/img2.jpg").exists()


def test_main_without_force_fails_on_existing_output(tmp_path: Path) -> None:
    raw = _make_raw(tmp_path)
    out = tmp_path / "out"
    main(["--raw", str(raw), "--out", str(out)])
    with pytest.raises(FileExistsError):
        main(["--raw", str(raw), "--out", str(out)])


def test_default_paths_are_repo_relative() -> None:
    assert conv.DEFAULT_RAW_DIR == REPO_ROOT / "data" / "raw" / "CarDD"
    assert conv.DEFAULT_OUT_DIR == REPO_ROOT / "data" / "processed" / "cardd_yolo"


# --- final swap ----------------------------------------------------------------------------


def _first_output(tmp_path: Path) -> tuple[Path, Path]:
    raw = _make_raw(tmp_path)
    out = tmp_path / "out"
    convert(raw, out)
    (out / "marker.txt").write_text("keep me")
    return raw, out


def _assert_old_dataset_intact(out: Path) -> None:
    assert (out / "marker.txt").read_text() == "keep me"
    assert (out / "conversion_stats.json").exists()
    assert not out.with_name("out.old").exists()
    assert not out.with_name("out.tmp").exists()


def _patch_rename(monkeypatch: pytest.MonkeyPatch, failing_name: str) -> None:
    original = Path.rename

    def fake(self: Path, target: Any) -> Path:
        if self.name == failing_name:
            raise PermissionError("locked")
        return original(self, target)

    monkeypatch.setattr(Path, "rename", fake)


def test_swap_failure_restores_previous_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw, out = _first_output(tmp_path)
    _patch_rename(monkeypatch, "out.tmp")
    with pytest.raises(PermissionError):
        convert(raw, out, force=True)
    _assert_old_dataset_intact(out)


def test_first_rename_failure_leaves_output_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw, out = _first_output(tmp_path)
    _patch_rename(monkeypatch, "out")
    with pytest.raises(PermissionError):
        convert(raw, out, force=True)
    _assert_old_dataset_intact(out)


def test_old_copy_delete_failure_only_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    raw, out = _first_output(tmp_path)
    original = conv.shutil.rmtree

    def fake(path: Any, *args: Any, **kwargs: Any) -> None:
        if Path(path).name == "out.old":
            raise PermissionError("locked")
        original(path, *args, **kwargs)

    monkeypatch.setattr(conv.shutil, "rmtree", fake)
    with caplog.at_level("WARNING", logger="cardd.data.convert"):
        convert(raw, out, force=True)
    assert not (out / "marker.txt").exists()  # new output is in place
    assert (out / "conversion_stats.json").exists()
    assert (tmp_path / "out.old" / "marker.txt").exists()
    assert not (tmp_path / "out.tmp").exists()
    assert "out.old" in caplog.text


def test_force_swap_leaves_no_leftovers(tmp_path: Path) -> None:
    raw, out = _first_output(tmp_path)
    convert(raw, out, force=True)
    assert not (out / "marker.txt").exists()
    assert not (tmp_path / "out.old").exists()
    assert not (tmp_path / "out.tmp").exists()


def test_fresh_output_creates_no_old_dir(tmp_path: Path) -> None:
    raw = _make_raw(tmp_path)
    convert(raw, tmp_path / "out")
    assert not (tmp_path / "out.old").exists()


def test_leftover_old_without_output_raises(tmp_path: Path) -> None:
    raw = _make_raw(tmp_path)
    old = tmp_path / "out.old"
    old.mkdir()
    (old / "precious.txt").write_text("data")
    with pytest.raises(FileExistsError, match="out.old"):
        convert(raw, tmp_path / "out")
    assert (old / "precious.txt").read_text() == "data"
    assert not (tmp_path / "out").exists()


def test_leftover_old_with_output_is_removed(tmp_path: Path) -> None:
    raw, out = _first_output(tmp_path)
    (tmp_path / "out.old").mkdir()
    (tmp_path / "out.old" / "x.txt").write_text("x")
    convert(raw, out, force=True)
    assert not (tmp_path / "out.old").exists()


def test_foreign_tmp_dir_not_deleted(tmp_path: Path) -> None:
    raw = _make_raw(tmp_path)
    tmp = tmp_path / "out.tmp"
    tmp.mkdir()
    (tmp / "precious.txt").write_text("data")
    with pytest.raises(ValueError, match="Refusing"):
        convert(raw, tmp_path / "out")
    assert (tmp / "precious.txt").read_text() == "data"


def test_label_files_use_unix_newlines(tmp_path: Path) -> None:
    raw = _make_raw(tmp_path)
    out = tmp_path / "out"
    convert(raw, out)
    assert b"\r" not in (out / "labels/train/img1.txt").read_bytes()
    assert b"\r" not in (out / "train_images.csv").read_bytes()
    assert b"\r" not in (out / "conversion_stats.json").read_bytes()
