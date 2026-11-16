"""Tests for adding undamaged-car negatives to the dataset (synthetic images only)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from cardd.data.add_negatives import (
    NEG_PREFIX,
    add_negatives,
    list_candidates,
    read_exclude_list,
)


def _make_image(path: Path, size: tuple[int, int] = (40, 30)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (120, 120, 120)).save(path)


@pytest.fixture
def env(tmp_path: Path) -> dict[str, Path]:
    source = tmp_path / "source"
    for i in range(1, 11):
        _make_image(source / "F_Normal" / f"FN_{i}.jpg", (2000, 1000))
    for i in range(1, 6):
        _make_image(source / "R_Normal" / f"RN_{i}.png")
    _make_image(source / "F_Crushed" / "FC_1.jpg")  # a damaged class, must never be used
    _make_image(source / "review_sheets" / "review_01.jpg")  # not a source class

    dataset = tmp_path / "dataset"
    _make_image(dataset / "images" / "train" / "real.jpg")
    (dataset / "labels" / "train").mkdir(parents=True)
    (dataset / "labels" / "train" / "real.txt").write_text("0 0.5 0.5 0.1 0.1 0.2 0.2\n")
    (dataset / "labels" / "train.cache").write_bytes(b"cache")

    exclude = tmp_path / "exclude.txt"
    exclude.write_text("# comment\n\nF_Normal/FN_1.jpg\nR_Normal\\RN_2.png\n", encoding="utf-8")
    return {
        "source": source,
        "dataset": dataset,
        "heldout": tmp_path / "heldout",
        "exclude": exclude,
    }


def _run(env: dict[str, Path], **kwargs: int) -> None:
    add_negatives(
        env["source"], env["dataset"], env["heldout"], exclude_file=env["exclude"], **kwargs
    )


def test_read_exclude_list_ignores_comments_and_normalises_slashes(env: dict[str, Path]) -> None:
    assert read_exclude_list(env["exclude"]) == {"F_Normal/FN_1.jpg", "R_Normal/RN_2.png"}
    assert read_exclude_list(None) == set()


def test_candidates_use_only_normal_classes_and_skip_excluded(env: dict[str, Path]) -> None:
    names = {p.name for p in list_candidates(env["source"], read_exclude_list(env["exclude"]))}
    assert "FN_1.jpg" not in names and "RN_2.png" not in names
    assert "FC_1.jpg" not in names and "review_01.jpg" not in names
    assert len(names) == 13  # 10 + 5 - 2 excluded


def test_negatives_get_empty_labels_and_original_data_is_kept(env: dict[str, Path]) -> None:
    _run(env, n_train=4, n_heldout=3, max_side=64)

    train_imgs = sorted((env["dataset"] / "images" / "train").glob(f"{NEG_PREFIX}*.jpg"))
    assert len(train_imgs) == 4
    for img in train_imgs:
        label = env["dataset"] / "labels" / "train" / f"{img.stem}.txt"
        assert label.is_file() and label.read_text() == ""
        with Image.open(img) as im:
            assert max(im.size) <= 64
    assert len(list((env["heldout"] / "images").glob("*.jpg"))) == 3
    assert (env["dataset"] / "images" / "train" / "real.jpg").is_file()
    assert (env["dataset"] / "labels" / "train" / "real.txt").read_text().startswith("0 ")
    assert not (env["dataset"] / "labels" / "train.cache").exists()


def test_train_and_heldout_are_disjoint_and_manifest_is_written(env: dict[str, Path]) -> None:
    _run(env, n_train=6, n_heldout=5)
    manifest = json.loads((env["dataset"] / "negatives_manifest.json").read_text())
    assert set(manifest["train"]).isdisjoint(manifest["heldout"])
    assert len(manifest["train"]) == 6 and len(manifest["heldout"]) == 5
    assert not set(manifest["train"] + manifest["heldout"]) & read_exclude_list(env["exclude"])


def test_sampling_is_deterministic_and_rerun_replaces_old_negatives(env: dict[str, Path]) -> None:
    _run(env, n_train=5, n_heldout=2, seed=7)
    first = json.loads((env["dataset"] / "negatives_manifest.json").read_text())
    _run(env, n_train=3, n_heldout=2, seed=7)
    second = json.loads((env["dataset"] / "negatives_manifest.json").read_text())

    assert second["train"] == first["train"][:3]
    assert len(list((env["dataset"] / "images" / "train").glob(f"{NEG_PREFIX}*"))) == 3
    assert len(list((env["dataset"] / "labels" / "train").glob(f"{NEG_PREFIX}*"))) == 3


def test_too_few_images_raises(env: dict[str, Path]) -> None:
    with pytest.raises(ValueError, match="usable"):
        _run(env, n_train=10, n_heldout=10)


def test_missing_dataset_mentions_converter(env: dict[str, Path], tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"cardd\.data\.convert"):
        add_negatives(env["source"], tmp_path / "nope", env["heldout"])
