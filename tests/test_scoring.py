import numpy as np
import pandas as pd

from areafluforecast.intervals import QUANTILE_COLUMNS, QUANTILE_LEVELS, fit_intervals
from areafluforecast.scoring import (add_cell_scores, compare_to_reference, diebold_mariano,
                                     holm, interval_score, wis_from_quantiles)


def test_wis_equals_interval_form():
    """2 x mean pinball == the Bracher et al. interval form with the same levels."""
    rng = np.random.default_rng(1)
    actual = rng.uniform(0, 50, 200)
    q = np.sort(rng.uniform(0, 60, (200, len(QUANTILE_LEVELS))), axis=1)
    levels = np.asarray(QUANTILE_LEVELS)
    median = q[:, levels.tolist().index(0.5)]
    k = (len(levels) - 1) // 2
    total = 0.5 * np.abs(actual - median)
    for i in range(k):
        alpha = 2 * levels[i]
        total += alpha / 2 * interval_score(actual, q[:, i], q[:, -1 - i], alpha)
    interval_form = total / (k + 0.5)
    np.testing.assert_allclose(wis_from_quantiles(actual, q), interval_form, rtol=1e-12)


def test_perfect_forecast_scores_zero():
    actual = np.array([3.0, 7.0])
    q = np.repeat(actual[:, None], len(QUANTILE_LEVELS), axis=1)
    np.testing.assert_allclose(wis_from_quantiles(actual, q), 0.0)


def test_diebold_mariano_and_holm():
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 0.1, 60)
    b = a + 0.5
    result = diebold_mariano(a, b, horizon=1)
    assert result.mean_difference < 0 and result.p_value < 1e-6
    same = diebold_mariano(a, a, horizon=2)
    assert same.p_value == 1.0
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_intervals_cover_95_on_validation():
    rng = np.random.default_rng(3)
    predicted = rng.uniform(1, 60, 2000)
    actual = predicted + rng.normal(0, 1 + 0.1 * predicted)
    model = fit_intervals(predicted, actual, np.ones(2000, int))
    lower, upper = model.bounds(predicted, np.ones(2000, int))
    coverage = ((actual >= lower) & (actual <= upper)).mean()
    assert abs(coverage - 0.95) < 0.01
    q = model.quantiles(predicted, np.ones(2000, int))
    assert np.all(np.diff(q, axis=1) >= 0)


def _frame(model, offset, weeks=40):
    dates = pd.date_range("2023-01-01", periods=weeks, freq="7D")
    rows = []
    for d in dates:
        for area in "ab":
            actual = 10.0
            row = {"model": model, "season": "s", "target_date": d, "area": area, "horizon": 1,
                   "actual": actual, "predicted": actual + offset, "lower": actual + offset - 2,
                   "upper": actual + offset + 2}
            row.update({c: actual + offset for c in QUANTILE_COLUMNS})
            rows.append(row)
    return pd.DataFrame(rows)


def test_compare_to_reference():
    rng = np.random.default_rng(0)
    frames = [_frame("persistence", 3.0), _frame("good", 0.5)]
    scored = add_cell_scores(pd.concat(frames, ignore_index=True))
    scored.loc[scored["model"] == "good", "wis"] += rng.normal(0, 0.05, 80)
    table = compare_to_reference(scored)
    row = table.iloc[0]
    assert row["model"] == "good" and row["relative_wis"] < 0.3 and row["p_holm"] < 0.001
