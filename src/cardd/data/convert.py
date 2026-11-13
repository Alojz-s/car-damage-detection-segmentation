"""Convert CarDD (COCO format) to YOLO-seg format.

Drops the ``tire flat`` category and re-indexes the remaining classes to 0..4
(see ``cardd.CLASS_NAMES``). Uses the official CarDD train/val/test split.
"""

from __future__ import annotations

from pathlib import Path


def convert(raw_dir: Path, out_dir: Path) -> None:
    """Convert the dataset in ``raw_dir`` into YOLO-seg layout in ``out_dir``."""
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit("Not implemented yet")
