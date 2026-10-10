"""Which weeks should the live graph model learn from? A held-out comparison.

Two questions, answered on the live track's three trust-check seasons (2023-24,
2024-25, 2025-26), with the live data and the live gnn_st settings:

1. COVID: train on every week ("full", what the live track uses), on every week
   except 2020-03-01..2022-06-30 ("exclude_covid"), or only from 2022-07-01 on
   ("post_covid"). post_covid skips 2023-24: it would train on 27 weeks.
2. Refit: after the usual fit, retrain each seed on training + validation weeks
   for its best epoch count (GNNConfig.refit), so the season right before the
   test season is learned from, not only held out.

Each task fits one (variant, season, horizon) with refit on, which yields both
arms: the usual fit ("standard") and the refit. 8 variant-seasons x 4 horizons
= 32 tasks.

    python scripts/training_window.py tasks                    # list them
    python scripts/training_window.py run-part --index 0..31   # Explorer array task
    python scripts/training_window.py collect                  # local, after pull

Scored against "same as last week" from the live backtest of the same seasons
(live/runs/boston/backtest_*), and against the current live setting
(full, standard) on exactly the same cells.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import areafluforecast as af
from areafluforecast.engine import run_model, units
from areafluforecast.progress import Progress
from areafluforecast.scoring import add_cell_scores, compare_to_reference

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_bridge import maybe_quick  # noqa: E402

COVID_START, COVID_END = pd.Timestamp("2020-03-01"), pd.Timestamp("2022-06-30")
POST_COVID_START = pd.Timestamp("2022-07-01")
SEASONS = {"full": (2023, 2024, 2025), "exclude_covid": (2023, 2024, 2025),
           "post_covid": (2024, 2025)}
HORIZONS = (1, 2, 3, 4)
CURRENT = "gnn_st full standard"
TASKS = [(v, s, h) for v, seasons in SEASONS.items() for s in seasons for h in HORIZONS]


def variant_rates(rates: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Rows dropped, not blanked: valid_origins then skips every window that
    crosses the gap, as the research repo's exclude_covid does."""
    if variant == "exclude_covid":
        return rates.loc[(rates.index < COVID_START) | (rates.index > COVID_END)]
    if variant == "post_covid":
        return rates.loc[rates.index >= POST_COVID_START]
    return rates


def out_dir(root: Path) -> Path:
    return root / "runs" / "boston" / "training_window"


def run_part(args) -> None:
    variant, season, horizon = TASKS[args.index]
    city = af.load_city(Path(args.root) / "boston", name="boston")
    config = maybe_quick(replace(af.resolve("gnn_st"), refit=True))
    start, end = city.season_window(season)
    label = f"{variant} {city.season_label(season)} {horizon}wk"
    print(f"Task {args.index}: {label}", flush=True)
    bar = Progress(units(config)).start()
    out = run_model(city, config, horizon=horizon, test_start=start, test_end=end,
                    rates=variant_rates(city.rates, variant),
                    on_unit=lambda m: bar.step(f"{label} {m}"))
    frames = [out.standard.assign(model=f"gnn_st {variant} standard"),
              out.test.assign(model=f"gnn_st {variant} refit")]
    path = out_dir(Path(args.root)) / "parts" / f"{variant}_{season}_h{horizon}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames).assign(season=city.season_label(season)).to_csv(path, index=False)
    split = out.meta.get("split", {})
    (path.with_suffix(".json")).write_text(json.dumps(
        {"task": args.index, "variant": variant, "season": season, "horizon": horizon,
         "split": split, "best_epochs": out.meta.get("best_epochs"),
         "quick": config != replace(af.resolve("gnn_st"), refit=True)}, indent=2) + "\n")
    bar.finish(path)


def collect(args) -> None:
    root = Path(args.root)
    parts = sorted((out_dir(root) / "parts").glob("*.csv"))
    missing = len(TASKS) - len(parts)
    if missing:
        print(f"Warning: {missing} of {len(TASKS)} parts missing")
    arms = pd.concat([pd.read_csv(p, parse_dates=["origin_date", "target_date"], dtype={"area": str})
                      for p in parts], ignore_index=True)
    backtest = sorted((root / "runs" / "boston").glob("backtest_*/predictions.csv"))[-1]
    live = pd.read_csv(backtest, parse_dates=["origin_date", "target_date"], dtype={"area": str})
    # The full/standard arm re-runs the live backtest's gnn_st: they should match.
    check = live[live["model"] == "gnn_st"].merge(
        arms[arms["model"] == CURRENT], on=["season", "horizon", "origin_date", "area"],
        suffixes=("_live", "_arm"))
    drift = float((check["predicted_live"] - check["predicted_arm"]).abs().max()) if len(check) else np.nan
    scored = add_cell_scores(pd.concat([arms, live[live["model"] == "persistence"]],
                                       ignore_index=True))
    vs_persistence = compare_to_reference(scored, "persistence")
    vs_current = compare_to_reference(scored[scored["model"] != "persistence"], CURRENT)
    cover = (scored.groupby(["model", "horizon"])[["covered50", "covered95"]].mean()
             .rename(columns={"covered50": "coverage50", "covered95": "coverage95"}).reset_index())
    by_season = (scored.groupby(["model", "season", "horizon"])["wis"].mean().unstack("season")
                 .reset_index())
    out = out_dir(root)
    vs_persistence.merge(cover, on=["model", "horizon"], how="left").to_csv(
        out / "vs_persistence.csv", index=False)
    vs_current.to_csv(out / "vs_current.csv", index=False)
    by_season.to_csv(out / "wis_by_season.csv", index=False)
    pd.set_option("display.width", 160)
    print(f"Parts: {len(parts)}/{len(TASKS)} | full/standard vs live backtest gnn_st: "
          f"max |diff| {drift:.4f} over {len(check)} cells")
    print("\nRelative WIS vs same-as-last-week (lower is better), all seasons each arm has:")
    print(vs_persistence.pivot(index="model", columns="horizon", values="relative_wis").round(3))
    print(f"\nAgainst the current live setting ({CURRENT}), same cells:")
    print(vs_current[["model", "horizon", "n_weeks", "relative_wis", "p_value"]].round(3)
          .to_string(index=False))
    print("\nCoverage (50% / 95% ranges):")
    print(cover.pivot(index="model", columns="horizon", values=["coverage50", "coverage95"]).round(3))
    print(f"Outputs: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("tasks", "run-part", "collect"):
        p = sub.add_parser(name)
        p.add_argument("--root", default="live")
        if name == "run-part":
            p.add_argument("--index", type=int, required=True, choices=range(len(TASKS)))
    args = parser.parse_args()
    if args.command == "tasks":
        for i, t in enumerate(TASKS):
            print(i, *t)
    elif args.command == "run-part":
        run_part(args)
    else:
        collect(args)


if __name__ == "__main__":
    main()
