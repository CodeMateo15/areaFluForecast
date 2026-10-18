"""Does loading the season's saved graph model each week lose anything? A check.

The live track can train gnn_st once, on the weeks before the season opens, and
load it every week after (live_forecast.py train / --models-dir) instead of
retraining weekly. Two questions, on Boston with the live gnn_st settings:

1. Same as the backtest? For each trust-check season (2023-24, 2024-25,
   2025-26) and horizon, train the season-start model exactly as the backtest
   does, save it, then replay every week of the season through the weekly
   loading path (forecast(frozen=...)), one week of data at a time. Its
   training should reproduce the backtest's gnn_st exactly; the weekly replay
   differs only by which random dropout masks each week's 100 draws get.
   12 tasks.
2. Same as retraining every week (what the live track did until now)? For each
   week of 2025-26 flu season (Oct-Mar), retrain gnn_st on every week before it,
   all four horizons, and compare with (1)'s replay on the same cells.
   One task per week.

    python scripts/frozen_check.py tasks                       # list them
    python scripts/frozen_check.py run-part --index 0..N-1     # Explorer array task
    python scripts/frozen_check.py collect                     # local, after pull
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import areafluforecast as af
from areafluforecast.backtest import data_fingerprint
from areafluforecast.checkpoint import save_checkpoint
from areafluforecast.engine import run_model, units
from areafluforecast.progress import Progress
from areafluforecast.scoring import add_cell_scores, compare_to_reference

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_bridge import maybe_quick  # noqa: E402

SEASONS = (2023, 2024, 2025)
HORIZONS = (1, 2, 3, 4)
WEEKLY_SEASON = 2025
FLU_MONTHS = (10, 11, 12, 1, 2, 3)
FROZEN, WEEKLY, BACKTEST = "gnn_st frozen weekly", "gnn_st retrained weekly", "gnn_st backtest"


def load(root: Path) -> af.City:
    return af.load_city(root / "boston", name="boston")


def as_of(city: af.City, week) -> af.City:
    weather = {k: v.loc[:week] for k, v in city.weather.items()} if city.weather else None
    return replace(city, rates=city.rates.loc[:week], weather=weather)


def weekly_origins(city: af.City) -> list[pd.Timestamp]:
    start, end = city.season_window(WEEKLY_SEASON)
    weeks = city.rates.index[(city.rates.index >= start) & (city.rates.index <= end)]
    return [w for w in weeks if w.month in FLU_MONTHS]


def tasks(city: af.City) -> list[tuple]:
    return ([("frozen", s, h) for s in SEASONS for h in HORIZONS]
            + [("weekly", w, None) for w in weekly_origins(city)])


def out_dir(root: Path) -> Path:
    return root / "runs" / "boston" / "frozen_check"


def run_frozen(root: Path, city: af.City, season: int, horizon: int, index: int) -> None:
    config = maybe_quick(af.resolve("gnn_st"))
    start, end = city.season_window(season)
    label = f"{city.season_label(season)} {horizon}wk"
    before = city.rates.index[city.rates.index < start]
    replays = [w for w in city.rates.index
               if w >= before[-1] and start <= w + pd.Timedelta(weeks=horizon) <= end]
    print(f"Task {index}: frozen {label}, {len(replays)} weekly replays", flush=True)
    bar = Progress(units(config) + len(replays)).start()
    # 1. The backtest's own training for this season and horizon.
    trained = run_model(city, config, horizon=horizon, test_start=start, test_end=end,
                        on_unit=lambda m: bar.step(f"{label} train {m}"))
    parts = out_dir(root) / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    frames = [trained.test.assign(model=BACKTEST)]
    with tempfile.TemporaryDirectory() as tmp:
        trained.meta.update({"anchor": str(start.date()),
                             "data_fingerprint": data_fingerprint(city, start - pd.Timedelta(days=1))})
        save_checkpoint(Path(tmp) / f"gnn_st_h{horizon}.pt", trained, city_name=city.name,
                        areas=city.areas)
        # 2. Each week, as the live track would see it, loading those weights.
        for week in replays:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", af.TrustWarning)
                fc = af.forecast(as_of(city, week), ("gnn_st",), horizons=(horizon,),
                                 configs={"gnn_st": _overrides(config)}, frozen=tmp,
                                 backtest=None, out_dir=None, progress=False)
            frames.append(fc.table.assign(model=FROZEN))
            bar.step(f"{label} replay {week.date()}")
    path = parts / f"frozen_{season}_h{horizon}.csv"
    pd.concat(frames, ignore_index=True).assign(season=city.season_label(season)).to_csv(
        path, index=False)
    path.with_suffix(".json").write_text(json.dumps(
        {"task": index, "season": season, "horizon": horizon, "replays": len(replays),
         "best_epochs": trained.meta.get("best_epochs"),
         "quick": config != af.resolve("gnn_st")}, indent=2) + "\n")
    bar.finish(path)


def run_weekly(root: Path, city: af.City, week: pd.Timestamp, index: int) -> None:
    config = maybe_quick(af.resolve("gnn_st"))
    print(f"Task {index}: retrain gnn_st on the weeks before {week.date()}, 1-4 wk", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", af.TrustWarning)
        fc = af.forecast(as_of(city, week), ("gnn_st",), horizons=HORIZONS,
                         configs={"gnn_st": _overrides(config)}, backtest=None, out_dir=None,
                         progress=True)
    path = out_dir(root) / "parts" / f"weekly_{week.date()}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fc.table.assign(model=WEEKLY, season=city.season_label(WEEKLY_SEASON)).to_csv(path, index=False)
    print(f"Outputs: {path}", flush=True)


def _overrides(config) -> dict:
    full = af.resolve("gnn_st")
    return {k: getattr(config, k) for k in ("n_seeds", "epochs", "mc_samples")
            if getattr(config, k) != getattr(full, k)}


def run_part(args) -> None:
    root = Path(args.root)
    city = load(root)
    kind, what, horizon = tasks(city)[args.index]
    if kind == "frozen":
        run_frozen(root, city, what, horizon, args.index)
    else:
        run_weekly(root, city, what, args.index)


def list_tasks(args) -> None:
    city = load(Path(args.root))
    for i, (kind, what, horizon) in enumerate(tasks(city)):
        name = (f"{city.season_label(what)} {horizon}wk" if kind == "frozen"
                else str(pd.Timestamp(what).date()))
        print(f"{i:3d}  {kind:7s} {name}")
    print(f"{len(tasks(city))} tasks: --array=0-{len(tasks(city)) - 1}")


def collect(args) -> None:
    root = Path(args.root)
    city = load(root)
    parts = sorted((out_dir(root) / "parts").glob("*.csv"))
    expected = len(tasks(city))
    if len(parts) < expected:
        print(f"Warning: {expected - len(parts)} of {expected} parts missing")
    arms = pd.concat([pd.read_csv(p, parse_dates=["origin_date", "target_date"], dtype={"area": str})
                      for p in parts], ignore_index=True)
    actual = city.rates.stack().rename("actual").rename_axis(["target_date", "area"]).reset_index()
    arms = arms.drop(columns=["actual"], errors="ignore").merge(actual, on=["target_date", "area"],
                                                                 how="left")
    backtest = Path(args.backtest) / "predictions.csv" if args.backtest else sorted(
        (root / "runs" / "boston").glob("backtest_*/predictions.csv"))[-1]
    print(f"Backtest: {backtest.parent}")
    live = pd.read_csv(backtest, parse_dates=["origin_date", "target_date"], dtype={"area": str})
    live = live[live["season"].isin([city.season_label(s) for s in SEASONS])]

    # Training reproduces the backtest's gnn_st: same weights, same draws.
    check = live[live["model"] == "gnn_st"].merge(
        arms[arms["model"] == BACKTEST], on=["season", "horizon", "origin_date", "area"],
        suffixes=("_live", "_arm"))
    same = float((check["predicted_live"] - check["predicted_arm"]).abs().max()) if len(check) else np.nan
    # The weekly replay differs from it only by its dropout draws.
    replay = arms[arms["model"] == BACKTEST].merge(
        arms[arms["model"] == FROZEN], on=["season", "horizon", "origin_date", "area"],
        suffixes=("_season", "_week"))
    gap = (replay["predicted_week"] - replay["predicted_season"]).abs()
    scale = replay["predicted_season"].abs().mean()

    scored = add_cell_scores(pd.concat([arms, live[live["model"] == "persistence"]],
                                       ignore_index=True))
    frozen_seasons = scored[scored["model"] != WEEKLY]
    vs_persistence = compare_to_reference(frozen_seasons, "persistence")
    cover = (frozen_seasons.groupby(["model", "horizon"])[["covered50", "covered95"]].mean()
             .rename(columns={"covered50": "coverage50", "covered95": "coverage95"}).reset_index())
    # Weekly retraining against the weekly frozen path, on the cells both have.
    label = city.season_label(WEEKLY_SEASON)
    pair = scored[(scored["season"] == label) & scored["model"].isin([FROZEN, WEEKLY])]
    weeks = set(pair.loc[pair["model"] == WEEKLY, "origin_date"])
    pair = pair[pair["origin_date"].isin(weeks)]
    vs_weekly = compare_to_reference(pair, FROZEN)
    pair_cover = (pair.groupby(["model", "horizon"])[["covered50", "covered95"]].mean()
                  .reset_index())

    out = out_dir(root)
    vs_persistence.merge(cover, on=["model", "horizon"], how="left").to_csv(
        out / "vs_persistence.csv", index=False)
    vs_weekly.to_csv(out / "retrain_vs_frozen.csv", index=False)
    pair_cover.to_csv(out / "retrain_vs_frozen_coverage.csv", index=False)
    pd.set_option("display.width", 160)
    print(f"Parts: {len(parts)}/{expected}")
    print(f"Season-start training vs the cached backtest's gnn_st: max |diff| {same:.2e} "
          f"over {len(check)} cells (0 = the same model)")
    print(f"Weekly replay vs the season-at-once forecast, same weights: median |diff| "
          f"{gap.median():.3f}, 99th pct {gap.quantile(0.99):.3f} (mean forecast {scale:.2f}) "
          f"over {len(replay)} cells")
    print("\nHow much better than 'same as last week' (WIS ratio, lower is better), 3 seasons:")
    print(vs_persistence.pivot(index="model", columns="horizon", values="relative_wis").round(3))
    print("\nShare of weeks inside the 95% range:")
    print(cover.pivot(index="model", columns="horizon", values="coverage95").round(3))
    print(f"\nRetraining every week vs loading the season-start model, {label} Oct-Mar "
          f"(ratio > 1: retraining is worse):")
    if vs_weekly.empty:
        print("  (no overlapping cells yet: needs the 2025-26 frozen parts and weekly parts)")
    else:
        print(vs_weekly[["model", "horizon", "n_weeks", "wis", f"wis_{FROZEN}", "relative_wis",
                         "p_value"]].round(3).to_string(index=False))
        print(pair_cover.pivot(index="model", columns="horizon", values="covered95").round(3))
    print(f"Outputs: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in {"tasks": list_tasks, "run-part": run_part, "collect": collect}.items():
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--root", default="live")
        if name == "run-part":
            p.add_argument("--index", type=int, required=True)
        if name == "collect":
            p.add_argument("--backtest", help="the trust-check backtest folder (default: newest)")
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
