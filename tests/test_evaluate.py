"""Tests for the evaluation entry point (fake model, no GPU, dataset or weights needed)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

import cardd.training.evaluate as eval_mod
import cardd.training.train as train_mod
from cardd.training.evaluate import count_false_positives, evaluate, list_images


def _metric(base: float) -> SimpleNamespace:
    return SimpleNamespace(mp=base, mr=base + 0.1, map50=base + 0.2, map=base + 0.3)


def _fake_metrics(save_dir: Path) -> SimpleNamespace:
    rows = [
        {
            "Class": "dent",
            "Images": np.int64(10),
            "Instances": np.int64(12),
            "mAP50": 0.5,
            "mAP50-95": 0.3,
            "Mask-mAP50": 0.4,
        },
        {
            "Class": "scratch",
            "Images": np.int64(8),
            "Instances": np.int64(9),
            "mAP50": 0.6,
            "mAP50-95": 0.35,
            "Mask-mAP50": 0.45,
        },
    ]
    return SimpleNamespace(
        box=_metric(0.1),
        seg=_metric(0.05),
        speed={"preprocess": 0.3, "inference": 4.0, "loss": 0.0, "postprocess": 5.0},
        save_dir=save_dir,
        summary=lambda decimals=5: rows,
    )


class FakeModel:
    """Records val kwargs and fakes predictions with fixed confidences per image."""

    instances: list[FakeModel] = []
    names = {0: "dent", 1: "scratch"}
    save_dir: Path

    def __init__(self, weights: str) -> None:
        self.weights = weights
        self.val_kwargs: dict[str, Any] = {}
        FakeModel.instances.append(self)

    def val(self, **kwargs: Any) -> SimpleNamespace:
        self.val_kwargs = kwargs
        FakeModel.save_dir.mkdir(parents=True, exist_ok=True)
        return _fake_metrics(FakeModel.save_dir)

    def predict(self, sources: list[str], conf: float, **_: Any) -> Any:
        per_image = [
            ([0, 1], [0.9, 0.3]),  # image 1: dent 0.9, scratch 0.3
            ([1], [0.4]),  # image 2: scratch 0.4
            ([], []),  # image 3: nothing
        ]
        for (cls, scores), _src in zip(per_image, sources, strict=False):
            keep = [(c, s) for c, s in zip(cls, scores, strict=True) if s >= conf]
            yield SimpleNamespace(
                boxes=SimpleNamespace(cls=[c for c, _ in keep], conf=[s for _, s in keep])
            )


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "repo"
    for split in ("train", "val", "test"):
        folder = root / "data" / "processed" / "ds" / "images" / split
        folder.mkdir(parents=True)
        (folder / "a.jpg").write_bytes(b"x")
    (root / "configs").mkdir()
    (root / "configs" / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": "data/processed/ds",
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {0: "dent", 1: "scratch"},
            }
        ),
        encoding="utf-8",
    )
    weights = root / "runs" / "exp" / "weights"
    weights.mkdir(parents=True)
    (weights / "best.pt").write_bytes(b"x")
    heldout = root / "heldout"
    heldout.mkdir()
    for i in range(3):
        (heldout / f"i{i}.jpg").write_bytes(b"x")
    (heldout / "notes.txt").write_text("not an image")

    monkeypatch.setattr(train_mod, "REPO_ROOT", root)
    monkeypatch.setattr(eval_mod, "REPO_ROOT", root)

    import ultralytics

    FakeModel.instances = []
    FakeModel.save_dir = root / "runs" / "eval" / "exp_test"
    monkeypatch.setattr(ultralytics, "YOLO", FakeModel)
    return root


def test_list_images_ignores_other_files(repo: Path) -> None:
    assert [p.name for p in list_images(repo / "heldout")] == ["i0.jpg", "i1.jpg", "i2.jpg"]


def test_count_false_positives_applies_each_threshold() -> None:
    model = FakeModel("w")
    out = count_false_positives(model, [Path("a"), Path("b"), Path("c")], [0.25, 0.5])
    assert out["0.25"] == {
        "images": 3,
        "images_with_detections": 2,
        "detections": 3,
        "by_class": {"dent": 1, "scratch": 2},
    }
    assert out["0.5"]["images_with_detections"] == 1
    assert out["0.5"]["by_class"] == {"dent": 1}


def test_evaluate_writes_report_and_uses_absolute_data_path(repo: Path) -> None:
    out = evaluate(Path("runs/exp/weights/best.pt"), split="test", heldout=Path("heldout"))

    assert out == FakeModel.save_dir
    kwargs = FakeModel.instances[0].val_kwargs
    assert kwargs["split"] == "test" and kwargs["conf"] == 0.001 and kwargs["plots"] is True
    resolved = yaml.safe_load(Path(kwargs["data"]).read_text(encoding="utf-8"))
    assert Path(resolved["path"]).is_absolute()
    assert Path(kwargs["project"]) == repo / "runs" / "eval"

    report = json.loads((out / "evaluation.json").read_text(encoding="utf-8"))
    assert report["weights"] == "runs/exp/weights/best.pt"
    assert report["split"] == "test"
    assert report["overall"]["box"]["mAP50-95"] == pytest.approx(0.4)
    assert report["overall"]["mask"]["mAP50"] == pytest.approx(0.25)
    assert [r["Class"] for r in report["per_class"]] == ["dent", "scratch"]
    assert report["per_class"][0]["Instances"] == 12
    assert report["speed_ms_per_image"]["inference"] == 4.0
    assert report["false_positives_on_intact_cars"]["0.5"]["images_with_detections"] == 1

    with (out / "evaluation_per_class.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["Class"] for r in rows] == ["dent", "scratch"]


def test_evaluate_without_heldout_has_no_false_positive_section(repo: Path) -> None:
    out = evaluate(Path("runs/exp/weights/best.pt"), split="val")
    report = json.loads((out / "evaluation.json").read_text(encoding="utf-8"))
    assert "false_positives_on_intact_cars" not in report


def test_evaluate_rejects_bad_input(repo: Path) -> None:
    with pytest.raises(ValueError, match="split"):
        evaluate(Path("runs/exp/weights/best.pt"), split="train")
    with pytest.raises(FileNotFoundError, match="Weights"):
        evaluate(Path("runs/missing/weights/best.pt"))
    (repo / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="No images"):
        evaluate(Path("runs/exp/weights/best.pt"), heldout=Path("empty"))
    assert FakeModel.instances == []
