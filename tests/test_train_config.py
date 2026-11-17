"""Tests for the training entry point (no GPU, dataset, weights or real training needed)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

import cardd.training.train as train_mod
from cardd.training.train import load_config, summarize_results, train

RESULTS_HEADER = (
    "                  epoch,                   time,      metrics/mAP50(B),"
    "   metrics/mAP50-95(B),      metrics/mAP50(M),   metrics/mAP50-95(M)\n"
)


def write_yaml(path: Path, data: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake repository root with a dataset, configs and a data YAML with a relative path."""
    root = tmp_path / "repo"
    for split in ("train", "val"):
        img_dir = root / "data" / "processed" / "ds" / "images" / split
        img_dir.mkdir(parents=True)
        (img_dir / "a.jpg").write_bytes(b"x")
    (root / "configs").mkdir()
    write_yaml(
        root / "configs" / "data.yaml",
        {
            "path": "data/processed/ds",
            "train": "images/train",
            "val": "images/val",
            "names": {0: "dent"},
        },
    )
    write_yaml(
        root / "configs" / "train.yaml",
        {
            "model": "m.pt",
            "data": "configs/data.yaml",
            "epochs": 2,
            "seed": 42,
            "deterministic": True,
            "project": "runs",
            "name": "exp",
        },
    )
    monkeypatch.setattr(train_mod, "REPO_ROOT", root)
    return root


class FakeYOLO:
    """Records the train kwargs and fakes a trainer with a run directory and results.csv."""

    instances: list[FakeYOLO] = []
    save_dir: Path

    def __init__(self, weights: str) -> None:
        self.weights = weights
        self.kwargs: dict[str, Any] = {}
        self.trainer = type("Trainer", (), {})()
        FakeYOLO.instances.append(self)

    def train(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.trainer.save_dir = FakeYOLO.save_dir
        FakeYOLO.save_dir.mkdir(parents=True, exist_ok=True)
        (FakeYOLO.save_dir / "results.csv").write_text(
            RESULTS_HEADER
            + "1,60.0,0.10,0.05,0.09,0.04\n"
            + "2,120.0,0.50,0.30,0.45,0.25\n"
            + "3,180.0,0.40,0.20,0.35,0.15\n",
            encoding="utf-8",
        )


@pytest.fixture
def fake_yolo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> type[FakeYOLO]:
    import ultralytics

    FakeYOLO.instances = []
    FakeYOLO.save_dir = tmp_path / "fake_run_dir" / "exp7"
    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)
    return FakeYOLO


def test_load_config_parses_yaml_and_typed_overrides(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "c.yaml", {"model": "m.pt", "data": "d.yaml", "epochs": 50})
    cfg = load_config(
        path, ["epochs=5", "amp=false", "name=foo", "lr0=0.01", "custom_unknown=3", "x=a=b"]
    )
    assert cfg["epochs"] == 5 and isinstance(cfg["epochs"], int)
    assert cfg["amp"] is False
    assert cfg["name"] == "foo"
    assert cfg["lr0"] == 0.01 and isinstance(cfg["lr0"], float)
    assert cfg["custom_unknown"] == 3
    assert cfg["x"] == "a=b"
    assert cfg["model"] == "m.pt"


@pytest.mark.parametrize("missing", ["model", "data"])
def test_load_config_missing_required_key(tmp_path: Path, missing: str) -> None:
    content = {"model": "m.pt", "data": "d.yaml"}
    del content[missing]
    with pytest.raises(ValueError, match=missing):
        load_config(write_yaml(tmp_path / "c.yaml", content))


def test_load_config_override_without_equals(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "c.yaml", {"model": "m.pt", "data": "d.yaml"})
    with pytest.raises(ValueError, match="key=value"):
        load_config(path, ["epochs"])


def test_missing_dataset_mentions_converter(repo: Path, fake_yolo: type[FakeYOLO]) -> None:
    shutil.rmtree(repo / "data" / "processed" / "ds")
    with pytest.raises(FileNotFoundError, match=r"cardd\.data\.convert"):
        train(Path("configs/train.yaml"))
    assert fake_yolo.instances == []


def test_empty_image_folder_raises(repo: Path, fake_yolo: type[FakeYOLO]) -> None:
    (repo / "data" / "processed" / "ds" / "images" / "val" / "a.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="empty"):
        train(Path("configs/train.yaml"))
    assert fake_yolo.instances == []


def test_train_resolves_paths_and_writes_outputs(repo: Path, fake_yolo: type[FakeYOLO]) -> None:
    original_data = (repo / "configs" / "data.yaml").read_text(encoding="utf-8")

    run_dir = train(Path("configs/train.yaml"), ["epochs=3"])

    assert run_dir == fake_yolo.save_dir
    (fake,) = fake_yolo.instances
    assert fake.weights == "m.pt"
    kwargs = fake.kwargs
    assert "model" not in kwargs
    assert kwargs["epochs"] == 3 and kwargs["seed"] == 42 and kwargs["deterministic"] is True
    assert Path(kwargs["project"]).is_absolute()
    assert Path(kwargs["project"]) == (repo / "runs").resolve()
    assert kwargs["name"] == "exp"

    data_yaml = Path(kwargs["data"])
    assert data_yaml.is_absolute()
    resolved = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    assert Path(resolved["path"]).is_absolute()
    assert Path(resolved["path"]) == (repo / "data" / "processed" / "ds").resolve()
    assert resolved["train"] == "images/train"
    # The original data YAML must stay untouched.
    assert (repo / "configs" / "data.yaml").read_text(encoding="utf-8") == original_data

    resolved_cfg = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    assert resolved_cfg["epochs"] == 3 and resolved_cfg["model"] == "m.pt"
    assert Path(resolved_cfg["project"]).is_absolute() and Path(resolved_cfg["data"]).is_absolute()
    assert (run_dir / "data_resolved.yaml").is_file()

    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    for key in ("ultralytics", "torch", "cuda_version", "gpu_name", "peak_vram_mb"):
        assert key in summary
    assert summary["epochs_run"] == 3
    assert summary["mean_seconds_per_epoch"] == 60.0
    assert summary["total_training_minutes"] == 3.0
    assert summary["best_epoch"] == 2
    assert summary["best_metrics"] == {
        "box_mAP50": 0.5,
        "box_mAP50-95": 0.3,
        "mask_mAP50": 0.45,
        "mask_mAP50-95": 0.25,
    }


def test_summarize_results_handles_missing_file(tmp_path: Path) -> None:
    assert summarize_results(tmp_path / "results.csv") == {}


def test_train_without_results_csv_still_writes_summary(
    repo: Path, fake_yolo: type[FakeYOLO], monkeypatch: pytest.MonkeyPatch
) -> None:
    def train_no_csv(self: FakeYOLO, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.trainer.save_dir = FakeYOLO.save_dir
        FakeYOLO.save_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(FakeYOLO, "train", train_no_csv)
    run_dir = train(Path("configs/train.yaml"))
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert "epochs_run" not in summary and "torch" in summary


def test_resume_uses_last_checkpoint(repo: Path, fake_yolo: type[FakeYOLO]) -> None:
    weights = repo / "runs" / "exp" / "weights"
    weights.mkdir(parents=True)
    (weights / "last.pt").write_bytes(b"x")

    train(Path("configs/train.yaml"), resume=True)

    (fake,) = fake_yolo.instances
    assert Path(fake.weights) == weights / "last.pt"
    assert fake.kwargs == {"resume": True}


def test_resume_without_checkpoint_raises(repo: Path, fake_yolo: type[FakeYOLO]) -> None:
    with pytest.raises(FileNotFoundError, match="last.pt"):
        train(Path("configs/train.yaml"), resume=True)
    assert fake_yolo.instances == []
