"""Gate B, part 1 verdict: package forecasts vs the paper's, season by season.

For each (season, horizon, model) it asks two questions:

* **Are the forecasts the same numbers?** Largest absolute difference in the
  point forecast. Training is deterministic, so on the same machine type this
  should be ~0; small float drift across CPUs is possible.
* **Would the paper's conclusion change?** Week-level Diebold-Mariano test on
  the weighted interval score (WIS), Holm-adjusted over all 27 comparisons.

A row passes when it is tied (no significant difference) OR practically
identical (mean WIS within 0.5%). The second clause matters because when two
forecasts are nearly the same, the weekly differences have almost no variance,
so even a 0.01% gap can come out "significant" -- a statement about float
rounding, not about the model.

    python scripts/parity/compare_parity.py --package parity_out \
        --research ~/GNN-Influenza-Boston/Code/results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_parity import tasks  # noqa: E402

from areafluforecast.intervals import QUANTILE_COLUMNS  # noqa: E402
from areafluforecast.scoring import diebold_mariano, holm, wis_from_quantiles  # noqa: E402

KEYS = ["origin_date", "target_date", "neighborhood"]
PRACTICAL_PCT = 0.5


def _load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["origin_date", "target_date"])
    return frame[np.isfinite(frame["actual"])]


def compare(package: Path, research: Path) -> pd.DataFrame:
    rows = []
    for season, horizon, arm in tasks():
        leaf = Path(f"backtest_{season}") / f"horizon_{horizon:02d}" / arm
        ours = package / leaf / "predictions.csv"
        theirs = research / leaf / "full" / "predictions.csv"
        if not ours.exists() or not theirs.exists():
            rows.append({"season": season, "horizon": horizon, "model": arm,
                         "status": "missing " + ("package" if not ours.exists() else "research")})
            continue
        a, b = _load(ours), _load(theirs)
        pair = a.merge(b, on=KEYS, suffixes=("_p", "_r"))
        wis_p = wis_from_quantiles(pair["actual_p"], pair[[f"{c}_p" for c in QUANTILE_COLUMNS]])
        wis_r = wis_from_quantiles(pair["actual_r"], pair[[f"{c}_r" for c in QUANTILE_COLUMNS]])
        pair["wis_p"], pair["wis_r"] = wis_p, wis_r
        weekly = pair.groupby("target_date")[["wis_p", "wis_r"]].mean()
        dm = diebold_mariano(weekly["wis_p"], weekly["wis_r"], horizon)
        rows.append({"season": season, "horizon": horizon, "model": arm, "status": "ok",
                     "cells": len(pair), "cells_research": len(b),
                     "wis_package": float(wis_p.mean()), "wis_research": float(wis_r.mean()),
                     "max_abs_diff_point": float((pair["predicted_p"] - pair["predicted_r"]).abs().max()),
                     "p_value": dm.p_value})
    table = pd.DataFrame(rows)
    if "p_value" in table:
        table["p_holm"] = holm(table["p_value"].to_numpy())
        table["tied"] = table["p_holm"].isna() | (table["p_holm"] >= 0.05)
        table["wis_gap_pct"] = 100 * (table["wis_package"] / table["wis_research"] - 1)
        table["practically_identical"] = table["wis_gap_pct"].abs() < PRACTICAL_PCT
        table["passes"] = table["tied"] | table["practically_identical"]
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package", type=Path, default=Path("parity_out"))
    parser.add_argument("--research", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    table = compare(args.package, args.research)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.round(4).to_string(index=False))
    ok = table[table["status"] == "ok"]
    passed = len(ok) == len(table) and bool(ok["passes"].all())
    exact = int((ok["max_abs_diff_point"] < 1e-3).sum()) if len(ok) else 0
    verdict = (f"PASS: all {len(table)} season x horizon x model rows tied or practically "
               f"identical; {exact} are numerically identical."
               if passed else "NOT PASSED: see missing or failing rows above.")
    print("\n" + verdict)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
