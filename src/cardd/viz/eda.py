"""EDA figures (thesis-ready, Polish or English labels).

Figures are built with ``matplotlib.figure.Figure`` directly (no pyplot), so no global
backend is touched and nothing leaks between calls; saving uses the Agg canvas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from cardd.data.stats import class_distribution, class_order
from cardd.viz.labels import Lang, class_label, split_label, tr

DPI = 300
SPLIT_COLORS = {"train": "#0072B2", "val": "#E69F00", "test": "#009E73"}  # Okabe-Ito
FIG_SIZE = (7.5, 4.5)


def _new_figure() -> tuple[Figure, Any]:
    fig = Figure(figsize=FIG_SIZE, dpi=DPI)
    FigureCanvasAgg(fig)
    return fig, fig.add_subplot(111)


def _save(fig: Figure, out: Path | None) -> None:
    if out is None:
        return
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")


def plot_class_distribution(df: pd.DataFrame, lang: Lang = "pl", out: Path | None = None) -> Figure:
    """Grouped bar chart of instance counts per class, one bar per split.

    ``df`` is either the output of ``class_distribution`` or the raw ``instances`` table.
    """
    dist = df if "n_instances" in df.columns else class_distribution(df)
    classes = list(dict.fromkeys(dist["class_name"]))
    splits = list(dict.fromkeys(dist["split"]))
    width = 0.8 / len(splits)
    x = np.arange(len(classes))

    fig, ax = _new_figure()
    for i, split in enumerate(splits):
        sub = dist[dist["split"] == split].set_index("class_name")["n_instances"]
        values = [int(sub.get(c, 0)) for c in classes]
        bars = ax.bar(
            x + (i - (len(splits) - 1) / 2) * width,
            values,
            width,
            label=split_label(lang, split),
            color=SPLIT_COLORS.get(split),
        )
        ax.bar_label(bars, fontsize=7, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels([class_label(lang, c) for c in classes])
    ax.set_xlabel(tr(lang, "axis_class"))
    ax.set_ylabel(tr(lang, "axis_count"))
    ax.set_title(tr(lang, "class_dist_title"))
    ax.legend(title=tr(lang, "legend_split"))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, out)
    return fig


def plot_area_distribution(df: pd.DataFrame, lang: Lang = "pl", out: Path | None = None) -> Figure:
    """Log-scale boxplot of ``area_frac`` per class; ``df`` is the ``instances`` table."""
    classes = [c for c in class_order(df) if (df["class_name"] == c).any()]
    # Non-positive areas cannot be drawn on a log axis.
    data = [df.loc[(df["class_name"] == c) & (df["area_frac"] > 0), "area_frac"] for c in classes]

    fig, ax = _new_figure()
    ax.boxplot(
        data,
        patch_artist=True,
        boxprops={"facecolor": "#9ecae1", "edgecolor": "#08519c"},
        medianprops={"color": "#08306b"},
        flierprops={"marker": ".", "markersize": 2, "alpha": 0.4},
    )
    ax.set_yscale("log")
    ax.set_xticks(range(1, len(classes) + 1))
    ax.set_xticklabels([class_label(lang, c) for c in classes])
    ax.set_xlabel(tr(lang, "axis_class"))
    ax.set_ylabel(tr(lang, "axis_area_frac"))
    ax.set_title(tr(lang, "area_dist_title"))
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, out)
    return fig
