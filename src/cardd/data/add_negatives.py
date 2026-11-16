"""Add undamaged-car images ("negatives") to the converted YOLO-seg dataset.

CarDD contains almost only damaged cars, so a model trained on it alone flags reflections and
intact lamps as damage. This script samples undamaged-car photos from a folder (by default the
``F_Normal``/``R_Normal`` classes of the Hugging Face dataset ``DrBimmer/comprehensive-car-damage``,
see ``docs/decisions.md``) and

* copies ``--n-train`` of them into ``<dataset>/images/train`` with an EMPTY label file (YOLO
  treats such images as background), and
* writes ``--n-heldout`` more, disjoint from the first, to ``<heldout>/images`` so the false
  positive rate can be measured on cars the model never saw.

Images are downscaled (long side ``--max-side``) and saved as JPEG. Training copies get the file
name prefix ``neg_`` so they are easy to recognise and are replaced when the script is re-run.
Re-run it after every ``python -m cardd.data.convert --force``, which removes them.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# src/cardd/data/add_negatives.py -> parents[3] is the repository root.
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = REPO_ROOT / "data" / "raw" / "comprehensive_car_damage_normal"
DEFAULT_DATASET = REPO_ROOT / "data" / "processed" / "cardd_yolo"
DEFAULT_HELDOUT = REPO_ROOT / "data" / "processed" / "intact_heldout"
DEFAULT_EXCLUDE = REPO_ROOT / "configs" / "negatives_exclude.txt"

SOURCE_CLASSES = ("F_Normal", "R_Normal")  # front and rear views of undamaged cars
NEG_PREFIX = "neg_"
MANIFEST_FILE = "negatives_manifest.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class NegativesResult:
    """Counts of what was written."""

    train: int
    heldout: int
    available: int


def read_exclude_list(path: Path | None) -> set[str]:
    """Read relative paths (``F_Normal/FN_13.jpg``), ignoring blank lines and ``#`` comments."""
    if path is None or not path.is_file():
        return set()
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return {line.replace("\\", "/") for line in lines if line and not line.startswith("#")}


def _natural_key(path: Path) -> tuple[str, int, str]:
    match = re.search(r"(\d+)", path.stem)
    return (path.parent.name, int(match.group(1)) if match else -1, path.name)


def list_candidates(source: Path, exclude: set[str]) -> list[Path]:
    """Sorted usable source images (only ``SOURCE_CLASSES`` folders, minus the exclusions)."""
    found: list[Path] = []
    for cls in SOURCE_CLASSES:
        folder = source / cls
        if not folder.is_dir():
            continue
        for path in folder.iterdir():
            rel = f"{cls}/{path.name}"
            if path.suffix.lower() in IMAGE_SUFFIXES and rel not in exclude:
                found.append(path)
    return sorted(found, key=_natural_key)


def _save_downscaled(src: Path, dst: Path, max_side: int) -> None:
    with Image.open(src) as im:
        img = ImageOps.exif_transpose(im).convert("RGB")
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, format="JPEG", quality=95)


def _remove_previous(dataset: Path, heldout: Path) -> None:
    """Delete negatives of an earlier run (training files with the prefix, held-out files)."""
    for sub in ("images/train", "labels/train"):
        for path in (dataset / sub).glob(f"{NEG_PREFIX}*"):
            path.unlink()
    for sub in ("images", "labels"):
        for path in (heldout / sub).glob(f"{NEG_PREFIX}*"):
            path.unlink()
    for cache in (dataset / "labels").glob("*.cache"):
        cache.unlink()  # Ultralytics label cache, rebuilt on the next run


def add_negatives(
    source: Path = DEFAULT_SOURCE,
    dataset: Path = DEFAULT_DATASET,
    heldout: Path = DEFAULT_HELDOUT,
    *,
    exclude_file: Path | None = DEFAULT_EXCLUDE,
    n_train: int = 200,
    n_heldout: int = 100,
    seed: int = 42,
    max_side: int = 1024,
) -> NegativesResult:
    """Sample negatives deterministically and write them (see the module docstring).

    Raises:
        FileNotFoundError: the converted dataset or the source folder is missing.
        ValueError: negative counts, or fewer usable images than ``n_train + n_heldout``.
    """
    if n_train < 0 or n_heldout < 0:
        raise ValueError("n_train and n_heldout must be >= 0.")
    if not (dataset / "images" / "train").is_dir():
        raise FileNotFoundError(
            f"Dataset {dataset} not found. Create it first with `python -m cardd.data.convert`."
        )
    if not source.is_dir():
        raise FileNotFoundError(f"Negatives source folder {source} not found.")

    candidates = list_candidates(source, read_exclude_list(exclude_file))
    if len(candidates) < n_train + n_heldout:
        raise ValueError(
            f"Only {len(candidates)} usable images, but {n_train} + {n_heldout} were requested."
        )

    rng = random.Random(seed)
    shuffled = candidates.copy()
    rng.shuffle(shuffled)
    train_files = shuffled[:n_train]
    heldout_files = shuffled[n_train : n_train + n_heldout]

    _remove_previous(dataset, heldout)
    manifest: dict[str, Any] = {
        "seed": seed,
        "max_side": max_side,
        "available": len(candidates),
        "train": [],
        "heldout": [],
    }

    for src in train_files:
        stem = f"{NEG_PREFIX}{src.parent.name}_{src.stem}"
        _save_downscaled(src, dataset / "images" / "train" / f"{stem}.jpg", max_side)
        label = dataset / "labels" / "train" / f"{stem}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("", encoding="utf-8")
        manifest["train"].append(f"{src.parent.name}/{src.name}")

    for src in heldout_files:
        stem = f"{NEG_PREFIX}{src.parent.name}_{src.stem}"
        _save_downscaled(src, heldout / "images" / f"{stem}.jpg", max_side)
        label = heldout / "labels" / f"{stem}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("", encoding="utf-8")
        manifest["heldout"].append(f"{src.parent.name}/{src.name}")

    for folder in (dataset, heldout):
        if folder.is_dir():
            (folder / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info(
        "Added %d negatives to training and %d to the held-out set (%d usable images).",
        len(train_files),
        len(heldout_files),
        len(candidates),
    )
    return NegativesResult(
        train=len(train_files), heldout=len(heldout_files), available=len(candidates)
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Add undamaged-car negatives to the dataset.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="negatives folder")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="YOLO-seg dataset")
    parser.add_argument("--heldout", type=Path, default=DEFAULT_HELDOUT, help="held-out folder")
    parser.add_argument("--exclude", type=Path, default=DEFAULT_EXCLUDE, help="exclusion list")
    parser.add_argument("--n-train", type=int, default=200, help="negatives added to training")
    parser.add_argument("--n-heldout", type=int, default=100, help="held-out intact images")
    parser.add_argument("--seed", type=int, default=42, help="sampling seed")
    parser.add_argument("--max-side", type=int, default=1024, help="downscale long side to this")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    add_negatives(
        args.source,
        args.dataset,
        args.heldout,
        exclude_file=args.exclude,
        n_train=args.n_train,
        n_heldout=args.n_heldout,
        seed=args.seed,
        max_side=args.max_side,
    )


if __name__ == "__main__":
    main()
