"""How good was a forecast, and is one model's lead bigger than chance?

* **Weighted interval score (WIS)** -- the score CDC FluSight ranks teams on
  (Bracher et al. 2021, PLoS Comput Biol 17:e1008618). It judges the whole
  forecast, band as well as point, in the target's own units. Lower is better.
* **Relative WIS** -- a model's WIS divided by persistence's ("next week looks
  like this week") on the same weeks. Below 1 means the model beat the
  simplest forecast a department could make by hand.
* **Diebold-Mariano test** -- is the weekly gap between two models' scores
  bigger than week-to-week noise? Tested over target WEEKS, with every area
  kept together inside a week: areas in one city move together and consecutive
  weeks are correlated, so treating every area-week as independent would make
  every difference look significant.
* **Holm correction** -- many models x horizons are compared at once, so the
  p-values are adjusted for the number of comparisons.

Same formulas as the research code (influenza/scoring.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .intervals import QUANTILE_COLUMNS, QUANTILE_LEVELS

KEYS = ["target_date", "area", "horizon"]


def interval_score(actual, lower, upper, alpha: float) -> np.ndarray:
    """Width of a central (1 - alpha) band plus 2/alpha per unit the truth fell outside."""
    actual = np.asarray(actual, float)
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    below = np.clip(lower - actual, 0.0, None)
    above = np.clip(actual - upper, 0.0, None)
    return (upper - lower) + (2.0 / alpha) * (below + above)


def wis_from_quantiles(actual, quantiles, levels=QUANTILE_LEVELS) -> np.ndarray:
    """WIS per row from a full set of predictive quantiles (2 x mean pinball loss)."""
    actual = np.asarray(actual, float)[:, None]
    values = np.asarray(quantiles, float)
    tau = np.asarray(levels, float)[None, :]
    if values.shape[1] != tau.shape[1]:
        raise ValueError(f"{values.shape[1]} quantile columns for {tau.shape[1]} levels")
    diff = actual - values
    pinball = np.maximum(tau * diff, (tau - 1.0) * diff)
    return 2.0 * pinball.mean(axis=1)


def pit_values(actual, quantiles, levels=QUANTILE_LEVELS) -> np.ndarray:
    """Probability integral transform; uniform on [0, 1] for a calibrated forecast."""
    actual = np.asarray(actual, float)
    values = np.asarray(quantiles, float)
    tau = np.asarray(levels, float)
    return np.array([np.interp(y, np.maximum.accumulate(row), tau, left=0.0, right=1.0)
                     for y, row in zip(actual, values)])


def add_cell_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Add `wis`, `abs_error`, `covered95` and `covered50` to a predictions frame (NaN where
    unobserved). The 50% range is the 25th to 75th percentile."""
    out = frame.copy()
    observed = np.isfinite(out["actual"].to_numpy(float))
    out["abs_error"] = (out["predicted"] - out["actual"]).abs()
    wis = np.full(len(out), np.nan)
    if observed.any():
        wis[observed] = wis_from_quantiles(out.loc[observed, "actual"],
                                           out.loc[observed, list(QUANTILE_COLUMNS)])
    out["wis"] = wis
    out["covered95"] = np.where(observed,
                                (out["actual"] >= out["lower"]) & (out["actual"] <= out["upper"]),
                                np.nan)
    if {"q0.250", "q0.750"} <= set(out.columns):
        out["covered50"] = np.where(observed,
                                    (out["actual"] >= out["q0.250"]) & (out["actual"] <= out["q0.750"]),
                                    np.nan)
    return out


@dataclass
class DMResult:
    statistic: float
    p_value: float
    mean_difference: float
    n_weeks: int


def diebold_mariano(loss_a, loss_b, horizon: int) -> DMResult:
    """Two-sided test that two forecasts have equal expected weekly loss.

    Negative mean difference means A is better. Newey-West variance over h - 1
    lags (h-step errors overlap by construction), Harvey-Leybourne-Newbold
    small-sample correction, Student t with n - 1 df.
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[np.isfinite(d)]
    n = d.size
    lag = max(int(horizon) - 1, 0)
    if n < 5:
        return DMResult(np.nan, np.nan, float(d.mean()) if n else np.nan, n)
    centred = d - d.mean()
    variance = centred @ centred / n
    for k in range(1, lag + 1):
        weight = 1.0 - k / (lag + 1.0)
        variance += 2.0 * weight * (centred[k:] @ centred[:-k]) / n
    if variance <= 0:
        same = bool(np.allclose(d, 0))
        return DMResult(0.0 if same else np.nan, 1.0 if same else np.nan, float(d.mean()), n)
    statistic = d.mean() / np.sqrt(variance / n)
    h = lag + 1
    statistic *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p_value = 2.0 * stats.t.sf(abs(statistic), df=n - 1)
    return DMResult(float(statistic), float(p_value), float(d.mean()), n)


def holm(p_values) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, same order as the input."""
    p = np.asarray(p_values, float)
    out = np.full_like(p, np.nan)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    values = p[finite]
    order = np.argsort(values)
    m = values.size
    adjusted = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * values[order]))
    back = np.empty(m)
    back[order] = adjusted
    out[finite] = back
    return out


def weekly_loss(frame: pd.DataFrame, column: str = "wis") -> pd.Series:
    """One number per (season, target week): the loss averaged over that week's areas."""
    keys = [k for k in ("season", "target_date") if k in frame.columns]
    return frame.groupby(keys)[column].mean().sort_index()


def compare_to_reference(scored: pd.DataFrame, reference: str = "persistence") -> pd.DataFrame:
    """Relative WIS and a week-level DM test of every model against `reference`.

    `scored` is a long frame with `model`, `horizon`, `season`, `target_date`,
    `area` and the per-cell `wis` from `add_cell_scores`. Each model is compared
    on exactly the cells both it and the reference scored. Seasons are pooled:
    the weekly series is concatenated season by season.
    """
    rows = []
    observed = scored[np.isfinite(scored["wis"])]
    for horizon, by_h in observed.groupby("horizon"):
        ref = by_h[by_h["model"] == reference]
        if ref.empty:
            continue
        ref_cells = ref.set_index(["season", "target_date", "area"])
        for model, part in by_h.groupby("model"):
            if model == reference:
                continue
            cells = part.set_index(["season", "target_date", "area"])
            common = cells.index.intersection(ref_cells.index)
            if common.empty:
                continue
            a = cells.loc[common].reset_index()
            b = ref_cells.loc[common].reset_index()
            loss_a, loss_b = weekly_loss(a), weekly_loss(b)
            dm = diebold_mariano(loss_a.to_numpy(), loss_b.to_numpy(), int(horizon))
            wis_a, wis_b = float(a["wis"].mean()), float(b["wis"].mean())
            rows.append({"model": model, "horizon": int(horizon), "n_weeks": dm.n_weeks,
                         "wis": wis_a, f"wis_{reference}": wis_b,
                         "relative_wis": wis_a / wis_b if wis_b > 0 else np.nan,
                         "dm_statistic": dm.statistic, "p_value": dm.p_value})
    table = pd.DataFrame(rows)
    if not table.empty:
        table["p_holm"] = holm(table["p_value"].to_numpy())
    return table
