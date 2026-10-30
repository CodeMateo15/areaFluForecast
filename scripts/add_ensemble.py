"""Add "avg" to forecasts archived before it existed. A one-off, logged exception to
"archives are never rewritten": it only appends rows, so every existing row stays
byte for byte what it was, and it writes ensemble.json in each folder it touches.

The average is computed from the members' archived forecasts, so it is exactly what
`live_forecast.py run` would have archived that week. Folders missing a member
(e.g. before arima joined) are skipped.

    python scripts/add_ensemble.py --city boston --dry-run
    python scripts/add_ensemble.py --city boston
    python scripts/live_forecast.py score --city boston
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

import areafluforecast as af
from areafluforecast.backtest import BacktestResult
from areafluforecast.configs import DEFAULT_MODELS
from areafluforecast.ensemble import ENSEMBLE, average, backtest_with_average
from areafluforecast.forecast import _trust

HORIZONS = (1, 2, 3, 4)
SEASON_GLOB = "[0-9][0-9][0-9][0-9]_[0-9][0-9]"
EVERY_MODEL_BEAT = "- Every model beat persistence in the backtest."


def append(path: Path, rows: pd.DataFrame) -> None:
    """Append rows in the file's own column order, without touching what is there."""
    columns = pd.read_csv(path, nrows=0).columns
    with path.open("a") as f:
        rows[list(columns)].to_csv(f, header=False, index=False)


def add(folder: Path, dry_run: bool) -> str:
    fc = pd.read_csv(folder / "forecast.csv", dtype={"area": str, "origin_date": str,
                                                    "target_date": str})
    if ENSEMBLE in set(fc["model"]):
        return "already has avg"
    avg = average(fc)
    if avg.empty:
        missing = sorted(set(DEFAULT_MODELS) - set(fc["model"]))
        return f"skipped: no {', '.join(missing)}"
    meta = json.loads((folder / "meta.json").read_text())
    bt = backtest_with_average(BacktestResult.load(meta["backtest"]))
    trust, notes = _trust(bt, [ENSEMBLE], HORIZONS)
    avg = avg.merge(trust, on=["model", "horizon"], how="left")
    verdicts = ", ".join(f"{h} wk {t}" for h, t in zip(trust["horizon"], trust["trust"]))
    if dry_run:
        return f"would add {len(avg)} rows ({verdicts})"

    append(folder / "forecast.csv", avg)
    dated = avg.assign(origin_date=pd.to_datetime(avg["origin_date"]),
                       target_date=pd.to_datetime(avg["target_date"]))
    append(folder / "forecast_flusight.csv",
           af.ForecastResult(city=meta["city"], table=dated, history=pd.DataFrame(),
                             trust=trust).to_flusight())
    if notes:
        text = (folder / "trust.md").read_text().replace(EVERY_MODEL_BEAT + "\n", "")
        (folder / "trust.md").write_text(text + "".join(f"- {n}\n" for n in notes))
    for name, frame in (("backtest_comparison.csv", bt.comparison), ("backtest_scores.csv", bt.scores)):
        append(folder / name, frame[frame["model"] == ENSEMBLE])
    (folder / "ensemble.json").write_text(json.dumps({
        "name": ENSEMBLE, "members": list(DEFAULT_MODELS), "method": "quantile mean",
        "rows": len(avg), "added_after_archiving": True, "by": "scripts/add_ensemble.py",
        "added": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2) + "\n")
    return f"added {len(avg)} rows ({verdicts})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--city", required=True)
    parser.add_argument("--root", default="live")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    folders = sorted(p.parent for p in Path(args.root).glob(f"{SEASON_GLOB}/{args.city}/*/forecast.csv"))
    if not folders:
        raise SystemExit(f"No archived forecasts for {args.city} under {args.root}.")
    for folder in folders:
        print(f"{folder}: {add(folder, args.dry_run)}", flush=True)


if __name__ == "__main__":
    main()
