"""Is the forecast late? Timing diagnostics for backtest forecasts.

Every model here tends to run behind the epidemic curve: it rises after the
truth rises and peaks after the truth peaks. Two reasons:

* **It only sees past rates.** A turn in a weekly flu curve is not visible in
  the data until after it has happened.
* **Squared-error-type losses make lateness cheap.** When the turn is
  uncertain, a smooth, slightly late forecast has a lower expected error than a
  guess at the turn, so the model hedges toward last week's level.

So the backtest measures timing explicitly, per model and horizon:

* **weeks late** -- how many weeks earlier the forecast would have to be slid to
  fit the observed curve best. Persistence is exactly h weeks late at h weeks
  ahead, by construction; a model that is 1 week late at 2 weeks ahead has
  recovered one week of timing.
* **timing share** -- the fraction of the squared error that disappears with
  that slide, i.e. how much of the error is timing rather than height.
* **lag fingerprint** -- the correlation between the miss and the week's change
  in the truth. Near -1 means the misses are a pure phase shift (low while the
  curve climbs, high while it falls); near 0 means they are unrelated to timing.
* **peak weeks late** -- median, over areas, of forecast peak week minus
  observed peak week.

Ported from the research code (compare_timing.py) so the numbers match it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MAX_LEAD = 4


def weekly_grid(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """(week x area) table on a complete weekly index, so a shift is a shift in weeks."""
    table = frame.pivot_table(index="target_date", columns="area", values=column, aggfunc="mean")
    weeks = pd.date_range(table.index.min(), table.index.max(), freq="7D")
    return table.reindex(weeks)


def shifted(table: pd.DataFrame, lead: int) -> pd.DataFrame:
    """`table` moved `lead` weeks earlier, re-indexed to line up with the truth."""
    if lead == 0:
        return table
    moved = table.iloc[lead:]
    moved.index = table.index[:len(moved)]
    return moved


def _rmse(truth: pd.DataFrame, forecast: pd.DataFrame) -> tuple[float, int]:
    error = (truth - forecast).to_numpy(dtype=float)
    observed = np.isfinite(error)
    if not observed.any():
        return float("nan"), 0
    return float(np.sqrt(np.mean(error[observed] ** 2))), int(observed.sum())


def lead_sweep(truth: pd.DataFrame, forecast: pd.DataFrame, max_lead: int = MAX_LEAD) -> pd.DataFrame:
    """RMSE of the forecast slid 0..max_lead weeks earlier, all on the same weeks."""
    window = truth.index[:len(truth.index) - max_lead]
    rows = []
    for lead in range(max_lead + 1):
        rmse, n = _rmse(truth.reindex(window), shifted(forecast, lead).reindex(window))
        rows.append({"lead_weeks": lead, "RMSE": rmse, "n_obs": n})
    return pd.DataFrame(rows)


def lag_fingerprint(truth: pd.DataFrame, forecast: pd.DataFrame) -> float:
    """Correlation between the miss and the truth's week-over-week change."""
    change = truth.diff().to_numpy(dtype=float).ravel()
    residual = (forecast - truth).to_numpy(dtype=float).ravel()
    usable = np.isfinite(change) & np.isfinite(residual)
    if usable.sum() < 10:
        return float("nan")
    return float(np.corrcoef(change[usable], residual[usable])[0, 1])


def peak_offsets(truth: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    """Forecast peak week minus observed peak week, per area (positive = late)."""
    rows = []
    for area in truth.columns:
        observed, predicted = truth[area].dropna(), forecast[area].dropna()
        if observed.empty or predicted.empty:
            continue
        rows.append({"area": area, "peak_week_actual": observed.idxmax(),
                     "peak_week_forecast": predicted.idxmax(),
                     "peak_weeks_late": (predicted.idxmax() - observed.idxmax()).days / 7.0})
    return pd.DataFrame(rows)


def timing_row(frame: pd.DataFrame, max_lead: int = MAX_LEAD) -> dict:
    """Timing scores for one model x horizon x season of backtest forecasts."""
    truth = weekly_grid(frame, "actual")
    forecast = weekly_grid(frame, "predicted")
    sweep = lead_sweep(truth, forecast, max_lead)
    at_zero = sweep.loc[sweep["lead_weeks"].eq(0)].iloc[0]
    best = sweep.loc[sweep["RMSE"].idxmin()]
    share = (1.0 - best["RMSE"] ** 2 / at_zero["RMSE"] ** 2) if at_zero["RMSE"] > 0 else np.nan
    peaks = peak_offsets(truth, forecast)
    return {"weeks_late": int(best["lead_weeks"]), "timing_share": float(share),
            "lag_fingerprint": lag_fingerprint(truth, forecast),
            "peak_weeks_late": float(peaks["peak_weeks_late"].median()) if len(peaks) else np.nan}


def timing_table(predictions: pd.DataFrame, max_lead: int = MAX_LEAD) -> pd.DataFrame:
    """Per model and horizon, averaged over seasons (median for the week counts)."""
    rows = []
    keys = [k for k in ("model", "horizon", "season") if k in predictions.columns]
    for key, part in predictions.groupby(keys):
        rows.append({**dict(zip(keys, key if isinstance(key, tuple) else (key,))),
                     **timing_row(part, max_lead)})
    per_season = pd.DataFrame(rows)
    if per_season.empty:
        return per_season
    return (per_season.groupby(["model", "horizon"])
            .agg(weeks_late=("weeks_late", "median"), timing_share=("timing_share", "mean"),
                 lag_fingerprint=("lag_fingerprint", "mean"),
                 peak_weeks_late=("peak_weeks_late", "median"),
                 seasons=("weeks_late", "size"))
            .reset_index())


def describe(row: pd.Series) -> str:
    """One plain-language line for the report."""
    late = row["weeks_late"]
    h = int(row["horizon"])
    if row["model"] == "persistence":
        return f"persistence, {h} wk ahead: {late:g} week(s) behind the curve (exactly {h} by construction)"
    if row["model"] == "seasonal_naive":
        return (f"{row['model']}, {h} wk ahead: does not follow the recent curve (it copies an "
                f"earlier season), so 'weeks behind' does not apply; its peak is a median "
                f"{row['peak_weeks_late']:g} week(s) off")
    gained = h - late
    verdict = (f"recovers {gained:g} of the {h} week(s) persistence loses" if gained > 0 else
               "no earlier than persistence" if gained == 0 else
               f"{-gained:g} week(s) later than persistence")
    return (f"{row['model']}, {h} wk ahead: {late:g} week(s) behind the curve ({verdict}); "
            f"{row['timing_share']:.0%} of its squared error is timing; "
            f"peaks a median {row['peak_weeks_late']:g} week(s) late")
