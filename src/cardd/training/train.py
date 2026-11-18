"""Config-driven training entry point for YOLO-seg (Ultralytics).

Usage::

    python -m cardd.training.train --config configs/train_yolo11n_seg.yaml [--set key=value ...]
    python -m cardd.training.train --config configs/train_yolo11n_seg.yaml --resume

All relative paths (data YAML, ``project``) are resolved against the repository root, so the
command works from any directory. Nothing heavy happens at import time (``ultralytics`` and
``torch`` are imported lazily), which keeps dataloader worker processes on Windows cheap.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# src/cardd/training/train.py -> parents[3] is the repository root.
REPO_ROOT = Path(__file__).resolve().parents[3]

REQUIRED_KEYS = ("model", "data")
DEFAULT_PROJECT = "runs"
RESOLVED_DATA_DIR = "_resolved_data"  # under <project>, stable so that --resume still finds it
RESOLVED_CONFIG_FILE = "resolved_config.yaml"
RESOLVED_DATA_FILE = "data_resolved.yaml"
SUMMARY_FILE = "run_summary.json"

_BOX_MAP50 = "metrics/mAP50(B)"
_BOX_MAP5095 = "metrics/mAP50-95(B)"
_MASK_MAP50 = "metrics/mAP50(M)"
_MASK_MAP5095 = "metrics/mAP50-95(M)"


def resolve_path(path: str | Path) -> Path:
    """Resolve ``path`` against the repository root unless it is already absolute."""
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def load_config(path: Path, overrides: Sequence[str] = ()) -> dict[str, Any]:
    """Load a training YAML and apply ``key=value`` overrides.

    Override values are parsed with ``yaml.safe_load`` (``epochs=5`` -> ``int``,
    ``amp=false`` -> ``bool``, ``name=foo`` -> ``str``). Unknown keys are allowed because they
    are passed on to Ultralytics as training arguments.

    Raises:
        ValueError: if an override has no ``=`` or a required key (``model``, ``data``) is missing.
    """
    with Path(path).open(encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config {path} must be a YAML mapping, got {type(loaded).__name__}.")
    cfg: dict[str, Any] = dict(loaded)

    for item in overrides:
        key, sep, raw = item.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ValueError(f"Invalid override {item!r}, expected key=value.")
        cfg[key] = yaml.safe_load(raw)

    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"Config is missing required key(s): {', '.join(missing)}.")
    return cfg


def _split_dirs(dataset_dir: Path, value: Any) -> list[Path]:
    """Return the image directories referenced by a ``train``/``val`` entry of a data YAML."""
    entries = value if isinstance(value, list) else [value]
    return [dataset_dir / str(e) for e in entries if e]


def prepare_data_yaml(
    data_path: Path, project: Path, name: str, splits: Sequence[str] = ("train", "val")
) -> tuple[dict[str, Any], Path]:
    """Validate the dataset and write a copy of the data YAML with an absolute ``path``.

    Ultralytics resolves a relative ``path:`` against its own ``datasets_dir`` setting instead
    of the current directory (``check_det_dataset``), so the relative path in
    ``configs/data.yaml`` is replaced by the absolute dataset directory here. The original file
    is left untouched. Returns the resolved data dict and the path of the written YAML.
    """
    with data_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict) or "path" not in data:
        raise ValueError(f"Data YAML {data_path.name} must be a mapping with a 'path' key.")

    dataset_dir = resolve_path(str(data["path"]))
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory {dataset_dir} does not exist. "
            "Create it first with `python -m cardd.data.convert`."
        )
    for split in splits:
        if split not in data:
            raise ValueError(f"Data YAML {data_path.name} has no '{split}' entry.")
        for img_dir in _split_dirs(dataset_dir, data[split]):
            if not img_dir.is_dir() or not any(p.is_file() for p in img_dir.iterdir()):
                raise FileNotFoundError(
                    f"Image folder {img_dir} is missing or empty. "
                    "Re-create the dataset with `python -m cardd.data.convert`."
                )

    data["path"] = str(dataset_dir)
    out_dir = project / RESOLVED_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return data, out_path


def _to_float(value: str | None) -> float | None:
    try:
        number = float(value) if value is not None else math.nan
    except ValueError:
        return None
    return None if math.isnan(number) else number


def summarize_results(results_csv: Path) -> dict[str, Any]:
    """Derive timing and best-epoch metrics from an Ultralytics ``results.csv``.

    The ``time`` column is cumulative (seconds), so mean seconds per epoch is its last value
    divided by the number of epochs. Column names are padded in the file and get stripped.
    Returns an empty dict when the file is missing or has no rows.
    """
    if not results_csv.is_file():
        return {}
    with results_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [{(k or "").strip(): v for k, v in row.items()} for row in reader]
    if not rows:
        return {}

    def metric(row: dict[str, str], key: str) -> float | None:
        return _to_float(row.get(key))

    def fitness(row: dict[str, str]) -> float:
        # Ultralytics seg fitness: 0.1 * mAP50 + 0.9 * mAP50-95, summed over box and mask.
        total = 0.0
        for k50, k5095 in ((_BOX_MAP50, _BOX_MAP5095), (_MASK_MAP50, _MASK_MAP5095)):
            total += 0.1 * (metric(row, k50) or 0.0) + 0.9 * (metric(row, k5095) or 0.0)
        return total

    summary: dict[str, Any] = {"epochs_run": len(rows)}
    total_seconds = _to_float(rows[-1].get("time"))
    if total_seconds is not None:
        summary["total_training_minutes"] = round(total_seconds / 60, 2)
        summary["mean_seconds_per_epoch"] = round(total_seconds / len(rows), 1)

    best = max(rows, key=fitness)
    best_epoch = _to_float(best.get("epoch"))
    summary["best_epoch"] = int(best_epoch) if best_epoch is not None else None
    summary["best_metrics"] = {
        "box_mAP50": metric(best, _BOX_MAP50),
        "box_mAP50-95": metric(best, _BOX_MAP5095),
        "mask_mAP50": metric(best, _MASK_MAP50),
        "mask_mAP50-95": metric(best, _MASK_MAP5095),
    }
    return summary


def _environment_info() -> dict[str, Any]:
    """Collect library versions and GPU info (imports torch and ultralytics lazily)."""
    import torch
    import ultralytics

    info: dict[str, Any] = {
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": None,
        "peak_vram_mb": None,
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    return info


def _write_outputs(run_dir: Path, cfg: Mapping[str, Any], resolved_data: Path) -> None:
    """Write ``resolved_config.yaml`` and ``run_summary.json`` into ``run_dir``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    data_copy = run_dir / RESOLVED_DATA_FILE
    shutil.copyfile(resolved_data, data_copy)

    final_cfg = dict(cfg)
    final_cfg["data"] = str(data_copy)
    with (run_dir / RESOLVED_CONFIG_FILE).open("w", encoding="utf-8") as f:
        yaml.safe_dump(final_cfg, f, sort_keys=False, allow_unicode=True)

    summary = {"run_name": run_dir.name, **_environment_info()}
    summary.update(summarize_results(run_dir / "results.csv"))
    with (run_dir / SUMMARY_FILE).open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote %s and %s to %s", RESOLVED_CONFIG_FILE, SUMMARY_FILE, run_dir)
    logger.info("Run summary: %s", json.dumps(summary))


def train(config_path: Path, overrides: Sequence[str] = (), *, resume: bool = False) -> Path:
    """Train a YOLO-seg model as described by ``config_path`` and return the real run directory.

    Args:
        config_path: training YAML (relative paths are resolved against the repository root).
        overrides: ``key=value`` strings overriding config entries.
        resume: continue ``<project>/<name>/weights/last.pt`` of the run named in the config.

    Raises:
        FileNotFoundError: dataset missing or empty, or no ``last.pt`` to resume from.
        ValueError: invalid config or overrides.
    """
    cfg = load_config(resolve_path(config_path), overrides)
    project = resolve_path(str(cfg.get("project") or DEFAULT_PROJECT))
    cfg["project"] = str(project)
    name = str(cfg.get("name") or "train")
    cfg["name"] = name

    # Validate everything that can fail cheaply before importing ultralytics or loading weights.
    _, data_yaml = prepare_data_yaml(resolve_path(str(cfg["data"])), project, name)
    cfg["data"] = str(data_yaml)

    last_pt = project / name / "weights" / "last.pt"
    if resume and not last_pt.is_file():
        raise FileNotFoundError(
            f"Cannot resume: {last_pt} does not exist. "
            "Check the run name (--set name=...) and that the run has saved a checkpoint."
        )
    if not resume and not cfg.get("exist_ok") and (project / name).exists():
        logger.warning(
            "Run directory %s already exists, Ultralytics will use an incremented name "
            "(for example %s2). Set exist_ok=true to overwrite or pick another name.",
            project / name,
            name,
        )

    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        logger.warning("CUDA is not available, training will run on the CPU and be very slow.")
    else:
        torch.cuda.reset_peak_memory_stats()

    model_spec = cfg["model"]
    train_kwargs = {k: v for k, v in cfg.items() if k != "model"}
    if resume:
        model = YOLO(str(last_pt))
        model.train(resume=True)
    else:
        model = YOLO(str(model_spec))
        model.train(**train_kwargs)

    run_dir = Path(model.trainer.save_dir)
    _write_outputs(run_dir, cfg, data_yaml)
    return run_dir


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a YOLO-seg model from a YAML config.")
    parser.add_argument("--config", type=Path, required=True, help="training YAML config")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config entry (repeatable), e.g. --set epochs=5",
    )
    parser.add_argument("--resume", action="store_true", help="resume from the run's last.pt")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_dir = train(args.config, args.overrides, resume=args.resume)
    logger.info("Done. Run directory: %s", run_dir)


if __name__ == "__main__":
    main()
