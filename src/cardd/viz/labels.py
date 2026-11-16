"""Shared Polish/English strings for figures. Keys are stable; later figure code reuses them."""

from __future__ import annotations

from typing import Literal

Lang = Literal["pl", "en"]

LABELS: dict[str, dict[str, str]] = {
    "pl": {
        "class_dist_title": "Rozkład liczby instancji uszkodzeń w klasach i podzbiorach",
        "area_dist_title": "Rozkład względnej powierzchni masek uszkodzeń",
        "axis_count": "Liczba instancji",
        "axis_class": "Klasa uszkodzenia",
        "axis_area_frac": "Udział powierzchni maski w obrazie (skala log.)",
        "legend_split": "Podzbiór",
        "split_train": "treningowy",
        "split_val": "walidacyjny",
        "split_test": "testowy",
        "class_dent": "wgniecenie",
        "class_scratch": "zarysowanie",
        "class_crack": "pęknięcie",
        "class_glass_shatter": "rozbite szkło",
        "class_lamp_broken": "uszkodzona lampa",
        "class_tire_flat": "przebita opona",
    },
    "en": {
        "class_dist_title": "Damage instance counts per class and split",
        "area_dist_title": "Relative mask area of damage instances",
        "axis_count": "Number of instances",
        "axis_class": "Damage class",
        "axis_area_frac": "Mask area as fraction of image (log scale)",
        "legend_split": "Split",
        "split_train": "train",
        "split_val": "validation",
        "split_test": "test",
        "class_dent": "dent",
        "class_scratch": "scratch",
        "class_crack": "crack",
        "class_glass_shatter": "glass shatter",
        "class_lamp_broken": "lamp broken",
        "class_tire_flat": "tire flat",
    },
}


def tr(lang: Lang, key: str) -> str:
    """Return the string for ``key`` in ``lang``, falling back to the key itself."""
    return LABELS[lang].get(key, key)


def class_label(lang: Lang, class_name: str) -> str:
    """Translate a project class name (underscored) for display."""
    return tr(lang, f"class_{class_name}")


def split_label(lang: Lang, split: str) -> str:
    """Translate a logical split name for display."""
    return tr(lang, f"split_{split}")
