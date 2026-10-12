import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from areafluforecast.timing import describe, timing_row, timing_table


def _frame(shift: int, *, weeks: int = 60, areas=("a", "b", "c"), model="m", horizon=1):
    """Forecast = truth from `shift` weeks earlier, on a smooth seasonal curve."""
    dates = pd.date_range("2023-06-04", periods=weeks + shift, freq="7D")
    t = np.arange(len(dates))
    rows = []
    for i, area in enumerate(areas):
        curve = 5 + 40 * np.exp(-0.5 * ((t - 30 - i) / 5.0) ** 2)
        for k in range(shift, len(dates)):
            rows.append({"model": model, "horizon": horizon, "season": "s", "area": area,
                         "target_date": dates[k], "actual": curve[k], "predicted": curve[k - shift]})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("shift", [0, 1, 2, 3])
def test_shift_is_measured(shift):
    row = timing_row(_frame(shift))
    assert row["weeks_late"] == shift
    assert row["peak_weeks_late"] == shift
    if shift:
        assert row["lag_fingerprint"] < -0.9
        assert row["timing_share"] > 0.99


def test_table_and_description():
    frames = pd.concat([_frame(1, model="persistence"), _frame(0, model="good")])
    table = timing_table(frames)
    assert set(table["model"]) == {"persistence", "good"}
    persistence = table[table.model == "persistence"].iloc[0]
    assert "exactly 1 by construction" in describe(persistence)
    good = table[table.model == "good"].iloc[0]
    assert "recovers 1 of the 1" in describe(good)


REPO = Path(os.environ.get("AREAFLU_RESEARCH_REPO", "~/GNN-Influenza-Boston")).expanduser()


@pytest.mark.research
def test_matches_research_compare_timing():
    """Same numbers as the research script on a committed research predictions file."""
    path = REPO / "Code/results/backtest_2024_25/horizon_02/gnn_st_tuned/full/predictions.csv"
    if not path.exists():
        pytest.skip("research predictions not available")
    sys.path.insert(0, str(REPO / "Code"))
    import argparse

    import compare_timing

    frame = pd.read_csv(path, parse_dates=["origin_date", "target_date"])
    args = argparse.Namespace(max_lead=4, slope_threshold=5.0)
    research, _ = compare_timing.score_one(frame, "gnn_st_tuned", args)
    ours = timing_row(frame.rename(columns={"neighborhood": "area"}))
    assert ours["weeks_late"] == research["weeks_late"]
    assert ours["timing_share"] == pytest.approx(research["phase_share_of_mse"], abs=1e-12)
    assert ours["lag_fingerprint"] == pytest.approx(research["lag_fingerprint"], abs=1e-12)
    assert ours["peak_weeks_late"] == pytest.approx(research["median_peak_weeks_late"], abs=1e-12)
