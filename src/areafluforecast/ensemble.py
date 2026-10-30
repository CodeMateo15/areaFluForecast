"""A combined forecast: the plain average of several models' forecasts.

Each predictive quantile is averaged across models (Vincentization), as the CDC
forecast hubs combine teams (Ray et al. 2023, Int J Forecast 39:1366).
Averaging monotone, non-negative quantile vectors keeps them monotone and
non-negative, so no re-sorting is needed. The point forecast and the 95% band are
averaged the same way.

The average is built from the members' finished forecasts, so it needs no
training and no saved weights, and it is scored like any other model.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from .backtest import BacktestResult, _summarize
from .configs import DEFAULT_MODELS
from .intervals import QUANTILE_COLUMNS

ENSEMBLE = "avg"
CELL = ["origin_date", "target_date", "horizon", "area"]
VALUES = ["predicted", "lower", "upper", *QUANTILE_COLUMNS]


def average(table: pd.DataFrame, members: Sequence[str] = DEFAULT_MODELS,
            name: str = ENSEMBLE) -> pd.DataFrame:
    """Rows for `name`, one per cell every member forecast; cells missing a member
    are left out rather than averaged over fewer models. Works on a forecast table
    or backtest predictions (whose `season` and `actual` are carried over)."""
    members = list(members)
    keys = [c for c in ("season", *CELL) if c in table.columns]
    rows = table[table["model"].isin(members)]
    rows = rows[rows.groupby(keys)["model"].transform("nunique") == len(members)]
    how = {c: "mean" for c in VALUES} | ({"actual": "first"} if "actual" in rows else {})
    out = rows.groupby(keys, sort=False, as_index=False).agg(how).assign(model=name)
    return out[[c for c in table.columns if c in out.columns]]


def with_average(table: pd.DataFrame, members: Sequence[str] = DEFAULT_MODELS,
                 name: str = ENSEMBLE) -> pd.DataFrame:
    """`table` plus the average's rows (replacing any it already had)."""
    table = table[table["model"] != name]
    return pd.concat([table, average(table, members, name)], ignore_index=True)


def backtest_with_average(bt: BacktestResult, members: Sequence[str] = DEFAULT_MODELS,
                          name: str = ENSEMBLE) -> BacktestResult:
    """`bt` rescored with the average as one more model, so it gets a verdict like
    the others (and joins their Holm family). Unchanged if a member is missing."""
    if not set(members) <= set(bt.predictions["model"]):
        return bt
    result = _summarize(bt.city, with_average(bt.predictions, members, name), configs=bt.configs,
                        tuning=bt.tuning, meta=bt.meta)
    result.path = bt.path
    return result
