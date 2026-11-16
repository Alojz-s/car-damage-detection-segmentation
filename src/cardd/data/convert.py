"""Convert CarDD (COCO format) to YOLO-seg format.

Drops the ``tire flat`` category and re-indexes the remaining classes to 0..4
(see ``cardd.CLASS_NAMES``). Uses the official CarDD train/val/test split.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from cardd import CLASS_NAMES

logger = logging.getLogger(__name__)

Multipart = Literal["merge", "split"]
LinkMode = Literal["copy", "hardlink"]

COCO_NAME_TO_YOLO_ID: dict[str, int] = {
    name.replace("_", " "): i for i, name in enumerate(CLASS_NAMES)
}
TIRE_FLAT = "tire flat"
SPLITS: dict[str, str] = {"train2017": "train", "val2017": "val", "test2017": "test"}

# src/cardd/data/convert.py -> parents[3] is the repository root.
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw" / "CarDD"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "processed" / "cardd_yolo"

STATS_FILE = "conversion_stats.json"
MIN_AREA = 1e-9  # in normalized units
PROGRESS_EVERY = 500


@dataclass
class SplitStats:
    """Conversion statistics for one split.

    ``images`` counts images written to the output. ``images_emptied`` counts images that
    only had ``tire flat`` annotations. ``images_without_labels`` counts all images that end
    up with zero label lines (tire-only ones included), whether kept or dropped.
    """

    images: int = 0
    images_emptied: int = 0
    instances_per_class: dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(CLASS_NAMES, 0)
    )
    multipart_instances: int = 0
    skipped_instances: int = 0
    label_lines: int = 0
    images_without_labels: int = 0


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", " ")


def _polygon_area(flat: list[float]) -> float:
    """Absolute polygon area (shoelace formula) of a flat ``[x1, y1, ...]`` list."""
    n = len(flat) // 2
    area2 = 0.0
    for i in range(n):
        j = (i + 1) % n
        area2 += flat[2 * i] * flat[2 * j + 1] - flat[2 * j] * flat[2 * i + 1]
    return abs(area2) / 2


def coco_polygons_to_yolo(
    segmentation: list[list[float]], width: int, height: int
) -> list[list[float]]:
    """Convert pixel polygons to flat lists normalized to 0..1 (clipped).

    Coordinates are clipped to the image first; polygons with fewer than 3 points or
    (after clipping) a near-zero area are dropped.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size {width}x{height}")
    result: list[list[float]] = []
    for poly in segmentation:
        n = len(poly) // 2
        if n < 3:
            continue
        flat: list[float] = []
        for i in range(n):
            flat.append(min(max(poly[2 * i] / width, 0.0), 1.0))
            flat.append(min(max(poly[2 * i + 1] / height, 0.0), 1.0))
        if _polygon_area(flat) < MIN_AREA:
            continue
        result.append(flat)
    return result


def merge_polygons(polygons: list[list[float]]) -> list[float]:
    """Merge several flat polygons into one by connecting them at their closest points.

    Uses ``ultralytics.data.converter.merge_multi_segment`` (the same routine Ultralytics
    applies to COCO multi-part instances) and clips the result to 0..1.
    """
    if len(polygons) == 1:
        return polygons[0]
    import numpy as np
    from ultralytics.data.converter import merge_multi_segment

    merged = np.concatenate(merge_multi_segment(polygons), axis=0)
    return [float(v) for v in np.clip(merged, 0.0, 1.0).reshape(-1)]


def _format_line(class_id: int, flat: list[float]) -> str:
    return f"{class_id} " + " ".join(f"{v:.6f}" for v in flat)


def _place_image(src: Path, dst: Path, link: LinkMode) -> None:
    """Copy or hardlink ``src`` to ``dst`` (hardlink falls back to copy on OSError)."""
    if link == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            logger.debug("Hardlink failed for %s, copying instead", src)
    shutil.copy2(src, dst)


def convert_split(
    coco_json: Path,
    images_dir: Path,
    out_images: Path,
    out_labels: Path,
    *,
    keep_empty: bool = True,
    multipart: Multipart = "merge",
    link: LinkMode = "copy",
    mapping_csv: Path | None = None,
) -> SplitStats:
    """Convert one COCO split to YOLO layout under ``out_images`` and ``out_labels``.

    With ``keep_empty=False`` every image that ends up without label lines is dropped.
    ``mapping_csv`` optionally receives one row per written image.
    """
    if multipart not in ("merge", "split"):
        raise ValueError(f"Unknown multipart mode: {multipart!r}")
    with coco_json.open(encoding="utf-8") as f:
        coco: dict[str, Any] = json.load(f)

    names = [Path(img["file_name"]).name for img in coco["images"]]
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise ValueError(f"Duplicate file name {name!r} in {coco_json}")
        seen.add(name)

    cat_names = {c["id"]: _normalize_name(c["name"]) for c in coco["categories"]}
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        anns_by_image[ann["image_id"]].append(ann)

    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    stats = SplitStats()
    rows: list[tuple[str, int, int, int, int]] = []
    images = sorted(coco["images"], key=lambda i: i["id"])
    for n, img in enumerate(images, start=1):
        if n % PROGRESS_EVERY == 0:
            logger.info("%s: %d/%d images", coco_json.name, n, len(images))
        file_name = Path(img["file_name"]).name
        width, height = int(img["width"]), int(img["height"])
        if width <= 0 or height <= 0:
            raise ValueError(
                f"Image {file_name!r} (id {img['id']}) has invalid size {width}x{height}"
            )

        anns = anns_by_image.get(img["id"], [])
        if anns and all(cat_names.get(a["category_id"]) == TIRE_FLAT for a in anns):
            stats.images_emptied += 1

        lines: list[str] = []
        per_class = dict.fromkeys(CLASS_NAMES, 0)
        multipart_count = 0
        skipped = 0
        for ann in anns:
            name = cat_names.get(ann["category_id"])
            if name == TIRE_FLAT:
                continue
            class_id = COCO_NAME_TO_YOLO_ID.get(name) if name else None
            seg = ann.get("segmentation")
            if (
                class_id is None
                or ann.get("iscrowd", 0)
                or not isinstance(seg, list)
                or not all(isinstance(p, list) for p in seg)
            ):
                skipped += 1
                continue
            polys = coco_polygons_to_yolo(seg, width, height)
            if not polys:
                skipped += 1
                continue
            if len(polys) > 1:
                multipart_count += 1
                if multipart == "merge":
                    polys = [merge_polygons(polys)]
            lines.extend(_format_line(class_id, p) for p in polys)
            per_class[CLASS_NAMES[class_id]] += 1

        stats.skipped_instances += skipped
        if not lines:
            stats.images_without_labels += 1
            if not keep_empty:
                continue

        _place_image(images_dir / img["file_name"], out_images / file_name, link)
        (out_labels / f"{Path(file_name).stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n"
        )
        stats.images += 1
        stats.label_lines += len(lines)
        stats.multipart_instances += multipart_count
        for cls, count in per_class.items():
            stats.instances_per_class[cls] += count
        rows.append((file_name, int(img["id"]), width, height, len(lines)))

    if mapping_csv is not None:
        with mapping_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["file_name", "image_id", "width", "height", "n_label_lines"])
            writer.writerows(rows)
    return stats


def _log_stats(all_stats: dict[str, SplitStats]) -> None:
    header = f"{'split':<6}{'images':>8}{'tire':>6}{'nolabel':>9}{'lines':>8}" + "".join(
        f"{n:>15}" for n in CLASS_NAMES
    )
    logger.info(header)
    for split, s in all_stats.items():
        logger.info(
            f"{split:<6}{s.images:>8}{s.images_emptied:>6}{s.images_without_labels:>9}"
            f"{s.label_lines:>8}" + "".join(f"{s.instances_per_class[n]:>15}" for n in CLASS_NAMES)
        )
        logger.info(
            "%s: multipart=%d skipped=%d", split, s.multipart_instances, s.skipped_instances
        )


def _validate_inputs(raw_dir: Path) -> None:
    """Check that all annotation files, image folders and listed images exist."""
    hint = (
        "Unpack the CarDD COCO data into data/raw/CarDD "
        "(train2017/, val2017/, test2017/, annotations/)."
    )
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_dir}. {hint}")
    missing: list[Path] = []
    for coco_split in SPLITS:
        ann = raw_dir / "annotations" / f"instances_{coco_split}.json"
        img_dir = raw_dir / coco_split
        if not ann.is_file():
            missing.append(ann)
        if not img_dir.is_dir():
            missing.append(img_dir)
        if not ann.is_file() or not img_dir.is_dir():
            continue
        with ann.open(encoding="utf-8") as f:
            images = json.load(f)["images"]
        missing.extend(p for p in (img_dir / i["file_name"] for i in images) if not p.is_file())
    if missing:
        shown = "\n".join(f"  {p}" for p in missing[:10])
        raise FileNotFoundError(
            f"{len(missing)} required input path(s) missing (showing up to 10):\n{shown}\n{hint}"
        )


def _check_replaceable(out_dir: Path) -> None:
    """Refuse to delete a directory that is not a previous converter output."""
    if not out_dir.is_dir():
        raise ValueError(f"Output path {out_dir} exists and is not a directory")
    if not (out_dir / STATS_FILE).is_file() and any(out_dir.iterdir()):
        raise ValueError(
            f"Refusing to replace {out_dir}: it is not empty and has no {STATS_FILE}, "
            "so it does not look like a previous conversion output"
        )


def _old_dir(out_dir: Path) -> Path:
    return out_dir.with_name(out_dir.name + ".old")


def _swap_in(tmp_dir: Path, out_dir: Path) -> None:
    """Move ``tmp_dir`` to ``out_dir`` without ever losing a previous ``out_dir``.

    Sequence: ``out_dir`` -> ``<out_dir>.old``, ``tmp_dir`` -> ``out_dir``, then delete the
    ``.old`` copy. If the second rename fails the old output is restored and the error is
    re-raised. A failure to delete the ``.old`` copy only logs a warning, because the new
    output is already in place.
    """
    if not out_dir.exists():
        tmp_dir.rename(out_dir)
        return
    old_dir = _old_dir(out_dir)
    out_dir.rename(old_dir)
    try:
        tmp_dir.rename(out_dir)
    except BaseException:
        try:
            old_dir.rename(out_dir)
        except OSError as restore_exc:
            raise RuntimeError(
                f"Could not restore the previous output: it is still available as {old_dir}; "
                f"rename it back to {out_dir}"
            ) from restore_exc
        raise
    try:
        shutil.rmtree(old_dir)
    except OSError as exc:
        logger.warning(
            "New output is in place, but could not delete %s (%s); remove it manually",
            old_dir,
            exc,
        )


def convert(
    raw_dir: Path = DEFAULT_RAW_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    keep_empty: bool = True,
    force: bool = False,
    multipart: Multipart = "merge",
    link: LinkMode = "copy",
) -> dict[str, SplitStats]:
    """Convert all splits in ``raw_dir`` into YOLO-seg layout in ``out_dir``.

    Inputs are validated first and the output is built in ``<out_dir>.tmp`` and renamed
    into place only on success, so a failure never damages an existing ``out_dir``.
    """
    _validate_inputs(raw_dir)
    if out_dir.exists():
        if not force:
            raise FileExistsError(f"Output directory {out_dir} already exists; use force=True")
        _check_replaceable(out_dir)

    old_dir = _old_dir(out_dir)
    if old_dir.exists():
        if not out_dir.exists():
            raise FileExistsError(
                f"{old_dir} exists but {out_dir} does not: a previous run may have been "
                f"interrupted. Inspect {old_dir} and rename it back to {out_dir} if it is good."
            )
        shutil.rmtree(old_dir)  # older backup of a run that completed

    tmp_dir = out_dir.with_name(out_dir.name + ".tmp")
    if tmp_dir.exists():
        _check_replaceable(tmp_dir)
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    try:
        all_stats: dict[str, SplitStats] = {}
        for coco_split, split in SPLITS.items():
            all_stats[split] = convert_split(
                raw_dir / "annotations" / f"instances_{coco_split}.json",
                raw_dir / coco_split,
                tmp_dir / "images" / split,
                tmp_dir / "labels" / split,
                keep_empty=keep_empty,
                multipart=multipart,
                link=link,
                mapping_csv=tmp_dir / f"{split}_images.csv",
            )
        (tmp_dir / STATS_FILE).write_text(
            json.dumps({k: asdict(v) for k, v in all_stats.items()}, indent=2),
            encoding="utf-8",
            newline="\n",
        )
        _swap_in(tmp_dir, out_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    _log_stats(all_stats)
    return all_stats


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Convert CarDD COCO to YOLO-seg.")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_DIR, help="CarDD COCO directory")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="output directory")
    parser.add_argument(
        "--drop-empty",
        action="store_true",
        help="drop all images without label lines (tire-only included) instead of keeping them",
    )
    parser.add_argument("--force", action="store_true", help="replace an existing output")
    parser.add_argument(
        "--multipart",
        choices=["merge", "split"],
        default="merge",
        help="multi-polygon instances: merge into one label line (default) or one line per part",
    )
    parser.add_argument(
        "--link",
        choices=["copy", "hardlink"],
        default="copy",
        help="copy images (default) or hardlink them (do not modify the linked files)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    convert(
        args.raw,
        args.out,
        keep_empty=not args.drop_empty,
        force=args.force,
        multipart=args.multipart,
        link=args.link,
    )


if __name__ == "__main__":
    main()
