"""Forecast plots (needs matplotlib: pip install 'areaFluForecast[plot]')."""

from __future__ import annotations

import math
from pathlib import Path


def plot_forecast(result, *, model: str | None = None, areas=None, weeks_of_history: int = 26,
                  path: str | Path | None = None):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Plotting needs matplotlib: pip install 'areaFluForecast[plot]'") from exc

    table = result.table
    model = model or table["model"].iloc[0]
    table = table[table["model"] == model]
    areas = areas or list(result.history.columns)
    history = result.history.iloc[-weeks_of_history:]
    cols = min(4, len(areas))
    rows = math.ceil(len(areas) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.5 * rows), squeeze=False,
                             sharex=True)
    for ax, area in zip(axes.ravel(), areas):
        part = table[table["area"] == area].sort_values("target_date")
        ax.plot(history.index, history[area], color="#333333", lw=1.2, label="observed")
        ax.fill_between(part["target_date"], part["lower"], part["upper"], color="#4C78A8",
                        alpha=0.2, lw=0, label="95% band")
        ax.fill_between(part["target_date"], part["q0.250"], part["q0.750"], color="#4C78A8",
                        alpha=0.35, lw=0, label="50% band")
        ax.plot(part["target_date"], part["predicted"], color="#4C78A8", lw=1.6, marker="o",
                ms=3, label="forecast")
        ax.set_title(area, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(areas):]:
        ax.set_visible(False)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"{result.city}: {model} forecast from {result.meta['last_observed_week']}",
                 fontsize=11)
    fig.autofmt_xdate()
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig
