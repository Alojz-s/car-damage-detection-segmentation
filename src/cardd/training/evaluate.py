"""Evaluate a trained YOLO-seg model on a dataset split (per-class box and mask metrics).

Usage::

    python -m cardd.training.evaluate --weights runs/yolo11n_seg_baseline/weights/best.pt
    python -m cardd.training.evaluate --weights <best.pt> --split test --heldout <images folder>

Writes to ``runs/eval/<name>/``: ``evaluation.json`` (overall and per-class metrics, speed,
optional false positive counts), ``evaluation_per_class.csv`` and the Ultralytics plots
(confusion matrix, precision-recall curves). Use the ``test`` split once per final model; choose
models and thresholds on ``val``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from cardd.training.train import REPO_ROOT, prepare_data_yaml, resolve_path

logger = logging.getLogger(__name__)

SPLITS = ("val", "test")
DEFAULT_DATA = "configs/data.yaml"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _plain(value: Any) -> Any:
    """Convert numpy scalars to plain Python numbers so the result is JSON serialisable."""
    return value.item() if hasattr(value, "item") else value


def overall_metrics(metrics: Any) -> dict[str, dict[str, float]]:
    """Overall precision, recall, mAP50 and mAP50-95 for boxes and masks."""
    result: dict[str, dict[str, float]] = {}
    for key, part in (("box", metrics.box), ("mask", metrics.seg)):
        result[key] = {
            "precision": round(float(part.mp), 5),
            "recall": round(float(part.mr), 5),
            "mAP50": round(float(part.map50), 5),
            "mAP50-95": round(float(part.map), 5),
        }
    return result


def per_class_rows(metrics: Any) -> list[dict[str, Any]]:
    """Per-class rows (counts, box and mask P/R/F1/mAP) from the Ultralytics summary."""
    return [{k: _plain(v) for k, v in row.items()} for row in metrics.summary(decimals=5)]


def count_false_positives(
    model: Any, images: Sequence[Path], confs: Sequence[float], imgsz: int = 640
) -> dict[str, dict[str, Any]]:
    """Count detections on images that contain no damage (every detection is a false positive).

    The model runs once at the lowest threshold in ``confs``; higher thresholds are applied to
    the saved scores. Returns, per threshold, the number of images with at least one detection,
    the total number of detections and the detections per class.
    """
    low = min(confs)
    found: list[list[tuple[str, float]]] = []
    for result in model.predict(
        [str(p) for p in images], conf=low, imgsz=imgsz, stream=True, verbose=False
    ):
        classes = [model.names[int(c)] for c in result.boxes.cls]
        scores = [float(s) for s in result.boxes.conf]
        found.append(list(zip(classes, scores, strict=True)))

    summary: dict[str, dict[str, Any]] = {}
    for conf in confs:
        kept = [[(c, s) for c, s in image if s >= conf] for image in found]
        by_class = Counter(c for image in kept for c, _ in image)
        summary[f"{conf:g}"] = {
            "images": len(found),
            "images_with_detections": sum(1 for image in kept if image),
            "detections": sum(by_class.values()),
            "by_class": dict(sorted(by_class.items())),
        }
    return summary


def list_images(folder: Path) -> list[Path]:
    """Sorted image files directly inside ``folder``."""
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def _write_per_class_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(
    weights: Path,
    *,
    split: str = "test",
    data: str | Path = DEFAULT_DATA,
    imgsz: int = 640,
    batch: int = 8,
    conf: float = 0.001,
    iou: float = 0.7,
    workers: int = 4,
    name: str | None = None,
    heldout: Path | None = None,
    fp_conf: Sequence[float] = (0.25, 0.5),
) -> Path:
    """Evaluate ``weights`` on ``split`` and return the output directory.

    ``conf=0.001`` and ``iou=0.7`` are the Ultralytics validation defaults used for mAP.
    With ``heldout`` (a folder of undamaged-car images) the false positive counts at the
    thresholds ``fp_conf`` are added.

    Raises:
        ValueError: unknown split.
        FileNotFoundError: missing weights, dataset or split folder.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}.")
    weights_path = resolve_path(weights)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights {weights} not found.")

    run_name = name or f"{weights_path.parents[1].name}_{split}"
    project = REPO_ROOT / "runs" / "eval"
    _, data_yaml = prepare_data_yaml(resolve_path(data), project, run_name, ("train", split))
    heldout_images = list_images(resolve_path(heldout)) if heldout else []
    if heldout and not heldout_images:
        raise FileNotFoundError(f"No images found in {heldout}.")

    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        batch=batch,
        conf=conf,
        iou=iou,
        workers=workers,
        project=str(project),
        name=run_name,
        exist_ok=True,
        plots=True,
    )
    out_dir = Path(metrics.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = per_class_rows(metrics)
    report: dict[str, Any] = {
        "weights": weights_path.relative_to(REPO_ROOT).as_posix()
        if weights_path.is_relative_to(REPO_ROOT)
        else weights_path.name,
        "split": split,
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "overall": overall_metrics(metrics),
        "per_class": rows,
        "speed_ms_per_image": {k: round(float(v), 2) for k, v in metrics.speed.items()},
    }
    if heldout_images:
        report["false_positives_on_intact_cars"] = count_false_positives(
            model, heldout_images, fp_conf, imgsz
        )

    (out_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_per_class_csv(out_dir / "evaluation_per_class.csv", rows)
    logger.info("Overall: %s", json.dumps(report["overall"]))
    if "false_positives_on_intact_cars" in report:
        logger.info("False positives: %s", json.dumps(report["false_positives_on_intact_cars"]))
    logger.info("Wrote evaluation.json and evaluation_per_class.csv to %s", out_dir)
    return out_dir


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO-seg model.")
    parser.add_argument("--weights", type=Path, required=True, help="path to best.pt")
    parser.add_argument("--split", choices=SPLITS, default="test", help="split to evaluate")
    parser.add_argument(
        "--data", default=DEFAULT_DATA, help="data YAML (default: configs/data.yaml)"
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default=None, help="output folder name under runs/eval")
    parser.add_argument(
        "--heldout",
        type=Path,
        default=None,
        help="folder of undamaged-car images (false positives)",
    )
    parser.add_argument(
        "--fp-conf",
        type=float,
        nargs="+",
        default=[0.25, 0.5],
        help="confidence thresholds for the false positive counts",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    evaluate(
        args.weights,
        split=args.split,
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        name=args.name,
        heldout=args.heldout,
        fp_conf=args.fp_conf,
    )


if __name__ == "__main__":
    main()
