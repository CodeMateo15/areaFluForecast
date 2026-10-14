"""The live track: forecast 1-4 weeks past the newest week, archive it, score it later.

    # refresh the data first (BPHC export, weather scrape), then
    python scripts/export_sample.py --out live/boston

    # all in one (local, or AREAFLU_QUICK=1 smoke test)
    python scripts/live_forecast.py run --city boston

    # on Explorer, as array tasks (scripts/explorer/areaflu.sbatch MODE=live_*):
    python scripts/live_forecast.py backtest-part --city boston --index 0..11
    python scripts/live_forecast.py forecast-part --city boston --horizon 1..4
    python scripts/live_forecast.py assemble --city boston

    python scripts/live_forecast.py score --city boston     # light: once new weeks arrive

Layout under --root (default live/):
    <city>/                         the data, as export_sample.py --out writes it
    runs/<city>/backtest_*          the trust-check backtest, reused until the seasons change
    runs/<city>/parts/              per-task pieces before `assemble`
    <season>/<city>/<origin>/       one frozen folder per forecast: forecast.csv,
                                    forecast_flusight.csv, trust.md, rates_asof.csv, meta.json,
                                    runtime.csv (seconds per model and horizon)
    <season>/<city>/scores.csv      every archived forecast joined to what happened

This track is separate from the paper's backtest: its forecasts are written once and
never re-run, so the scores are a true prospective record. Persistence is forecast
alongside the models so each week has its "same as last week" reference.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import warnings
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

import areafluforecast as af
from areafluforecast.backtest import BacktestResult, combine
from areafluforecast.configs import DEFAULT_MODELS
from areafluforecast.engine import make_split, make_window
from areafluforecast.forecast import _trust
from areafluforecast.scoring import add_cell_scores

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_bridge import maybe_quick  # noqa: E402

HORIZONS = (1, 2, 3, 4)
SEASON_GLOB = "[0-9][0-9][0-9][0-9]_[0-9][0-9]"


# --- helpers -------------------------------------------------------------------

def load(args) -> af.City:
    city = af.load_city(args.data or Path(args.root) / args.city, name=args.city)
    if getattr(args, "as_of", None):
        as_of = pd.Timestamp(args.as_of)
        weather = {k: v.loc[:as_of] for k, v in city.weather.items()} if city.weather else None
        city = replace(city, rates=city.rates.loc[:as_of], weather=weather)
    return city


def season_of(city: af.City, week: pd.Timestamp) -> str:
    year = week.year if week >= city.season_window(week.year)[0] else week.year - 1
    return city.season_label(year).replace("-", "_")


def trust_seasons(args, city: af.City) -> list[int]:
    return list(args.seasons or city.complete_seasons()[-3:])


def quick_configs(models) -> dict:
    """AREAFLU_QUICK=1: tiny configs to smoke-test the script before the cluster."""
    out = {}
    for m in models:
        full, small = af.resolve(m), maybe_quick(af.resolve(m))
        if small != full:
            out[m] = {k: v for k, v in asdict(small).items() if k in ("n_seeds", "epochs", "mc_samples")}
    return out


def cached_backtest(runs: Path, city: str, seasons, models) -> BacktestResult | None:
    root = runs / city
    for meta in sorted(root.glob("backtest_*/backtest.json"), reverse=True) if root.exists() else []:
        info = json.loads(meta.read_text())
        if (info.get("seasons") == list(seasons) and set(HORIZONS) <= set(info.get("horizons", []))
                and set(models) <= set(info.get("models", []))):
            return BacktestResult.load(meta.parent)
    return None


def machine() -> str:
    """Where a forecast ran, in words that say nothing about who ran it."""
    name = {"Darwin": "macOS"}.get(platform.system(), platform.system())
    return f"{name} {platform.machine()}, {os.cpu_count()} CPUs"


def timed_forecast(city: af.City, models, horizons, configs: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """af.forecast one model and horizon at a time, timing each. Each config carries
    its own seed, so the numbers match a single call with every model. The trust
    check is left to archive(), which has the backtest."""
    frames, rows = [], []
    for horizon in horizons:
        for model in models:
            started = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", af.TrustWarning)
                fc = af.forecast(city, (model,), horizons=(horizon,), configs=configs, backtest=None,
                                 out_dir=None)
            seconds = time.perf_counter() - started
            frames.append(fc.table)
            config = af.resolve(model, configs.get(model))
            rows.append({"model": model, "horizon": horizon, "seconds": round(seconds, 2),
                         "n_seeds": getattr(config, "n_seeds", None),
                         "epochs": getattr(config, "epochs", None), "machine": machine()})
            print(f"{model} {horizon} wk: {seconds:.1f}s", flush=True)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(rows)


def training_window(city: af.City, model: str = "gnn_st", horizon: int = 1) -> dict:
    """The target weeks one model learned from and held out, by the same split
    forecast() uses, so the app can say which seasons are training seasons."""
    last = city.rates.index.max()
    future = pd.date_range(last + pd.Timedelta(days=7), periods=max(HORIZONS), freq="7D")
    index = city.rates.index.append(future)
    window = make_window(af.resolve(model), horizon, future[0], future[-1])
    split = make_split(index, window, only_origin=len(city.rates) - 1)
    span = lambda origins: [str(index[origins[0] + horizon].date()),
                            str(index[origins[-1] + horizon].date())]
    return {"model": model, "horizon": horizon, "train_targets": span(split.train),
            "validation_targets": span(split.val)}


def archive(args, city: af.City, table: pd.DataFrame, bt: BacktestResult, quick: bool,
            runtime: pd.DataFrame | None = None) -> Path:
    """Write the frozen forecast folder. Never overwrites without --force."""
    origin = city.rates.index.max()
    season = season_of(city, origin)
    folder = Path(args.root) / season / args.city / str(origin.date())
    if folder.exists() and not args.force:
        raise SystemExit(f"{folder} exists. Archived forecasts are never overwritten; "
                         "pass --force only to replace a broken run.")
    ref = af.forecast(city, ("persistence",), horizons=HORIZONS, backtest=None, out_dir=None,
                      progress=False)
    trust, notes = _trust(bt, list(args.models), HORIZONS)
    table = pd.concat([table.drop(columns=["trust"], errors="ignore").merge(
                           trust, on=["model", "horizon"], how="left"),
                       ref.table.assign(trust="reference")], ignore_index=True)
    folder.mkdir(parents=True, exist_ok=True)
    table.to_csv(folder / "forecast.csv", index=False)
    af.ForecastResult(city=args.city, table=table, history=city.rates,
                      trust=trust).to_flusight().to_csv(folder / "forecast_flusight.csv", index=False)
    (folder / "trust.md").write_text("# Trust check\n\n" + "\n".join(
        f"- {n}" for n in notes or ["Every model beat persistence in the backtest."]) + "\n")
    city.rates.to_csv(folder / "rates_asof.csv", date_format="%Y-%m-%d")
    bt.comparison.to_csv(folder / "backtest_comparison.csv", index=False)
    bt.scores.to_csv(folder / "backtest_scores.csv", index=False)
    if runtime is not None and not runtime.empty:
        runtime.to_csv(folder / "runtime.csv", index=False)
    (folder / "meta.json").write_text(json.dumps({
        "city": args.city, "label": city.label, "season": season,
        "origin": str(origin.date()), "horizons": list(HORIZONS), "models": list(args.models),
        "backtest": str(bt.path), "backtest_seasons": bt.meta.get("season_labels"),
        "training": training_window(city),
        "issued": time.strftime("%Y-%m-%dT%H:%M:%S"), "quick": quick}, indent=2) + "\n")
    print(f"Outputs: {folder}", flush=True)
    return folder


# --- commands --------------------------------------------------------------------

def run(args) -> None:
    city = load(args)
    print(city.summary(), flush=True)
    configs = quick_configs(args.models)
    seasons = trust_seasons(args, city)
    runs = Path(args.root) / "runs"
    bt = cached_backtest(runs, args.city, seasons, args.models)
    if bt is None:
        print(f"Backtesting seasons {seasons} for the trust check", flush=True)
        bt = af.backtest(city, args.models, seasons=seasons, horizons=HORIZONS, configs=configs,
                         out_dir=runs)
    else:
        print(f"Reusing backtest {bt.path}", flush=True)
    table, runtime = timed_forecast(city, args.models, HORIZONS, configs)
    archive(args, city, table, bt, bool(configs), runtime)


def backtest_part(args) -> None:
    """One (season, horizon) of the trust-check backtest. --index i maps to
    seasons[i // 4], horizon HORIZONS[i % 4]."""
    city = load(args)
    seasons = trust_seasons(args, city)
    season, horizon = seasons[args.index // len(HORIZONS)], HORIZONS[args.index % len(HORIZONS)]
    out = Path(args.root) / "runs" / args.city / "parts" / f"bt_{season}_h{horizon}"
    print(f"Backtest part {args.index}: season {city.season_label(season)}, {horizon} wk", flush=True)
    bt = af.backtest(city, args.models, seasons=[season], horizons=(horizon,),
                     configs=quick_configs(args.models), out_dir=None)
    bt.save(out)
    print(f"Outputs: {out}", flush=True)


def forecast_part(args) -> None:
    city = load(args)
    origin = str(city.rates.index.max().date())
    out = Path(args.root) / "runs" / args.city / "parts" / origin / f"fc_h{args.horizon}.csv"
    table, runtime = timed_forecast(city, args.models, (args.horizon,), quick_configs(args.models))
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    runtime.to_csv(out.parent / f"runtime_h{args.horizon}.csv", index=False)
    print(f"Outputs: {out}", flush=True)


def assemble(args) -> None:
    city = load(args)
    seasons = trust_seasons(args, city)
    runs = Path(args.root) / "runs"
    parts_dir = runs / args.city / "parts"
    bt = cached_backtest(runs, args.city, seasons, args.models)
    if bt is None:
        wanted = [parts_dir / f"bt_{s}_h{h}" for s in seasons for h in HORIZONS]
        missing = [p.name for p in wanted if not (p / "backtest.json").exists()]
        if missing:
            raise SystemExit(f"Backtest parts missing: {missing}. Run MODE=live_backtest first.")
        parts = [BacktestResult.load(p) for p in wanted]
        stale = [p.name for p, part in zip(wanted, parts)
                 if not set(args.models) <= set(part.predictions["model"])]
        if stale:
            raise SystemExit(f"Backtest parts {stale} predate models {list(args.models)}; delete "
                             "them and rerun MODE=live_backtest.")
        bt = combine(parts)
        bt.save(runs / args.city / f"backtest_{time.strftime('%Y%m%d-%H%M%S')}")
        print(f"Combined {len(wanted)} backtest parts into {bt.path}", flush=True)
    origin = str(city.rates.index.max().date())
    files = [parts_dir / origin / f"fc_h{h}.csv" for h in HORIZONS]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"Forecast parts for {origin} missing: {missing}. Run MODE=live_forecast.")
    table = pd.concat([pd.read_csv(f, parse_dates=["origin_date", "target_date"], dtype={"area": str})
                       for f in files], ignore_index=True)
    timings = [parts_dir / origin / f"runtime_h{h}.csv" for h in HORIZONS]
    runtime = (pd.concat([pd.read_csv(f) for f in timings], ignore_index=True)
               if all(f.exists() for f in timings) else None)
    archive(args, city, table, bt, bool(quick_configs(args.models)), runtime)
    print(bt.report(show=False))


def score(args) -> None:
    root = Path(args.root)
    truth = af.load_city(args.data or root / args.city, name=args.city, weather=False).rates
    actual = truth.stack().rename("actual").rename_axis(["target_date", "area"]).reset_index()
    frames = []
    for path in sorted(root.glob(f"{SEASON_GLOB}/{args.city}/*/forecast.csv")):
        fc = pd.read_csv(path, parse_dates=["origin_date", "target_date"], dtype={"area": str})
        frames.append(fc.merge(actual, on=["target_date", "area"], how="left")
                      .assign(season=path.parents[2].name))
    if not frames:
        raise SystemExit(f"No archived forecasts for {args.city} under {root}.")
    scored = add_cell_scores(pd.concat(frames, ignore_index=True))
    keep = ["season", "model", "origin_date", "target_date", "horizon", "area", "predicted",
            "lower", "upper", "actual", "wis", "abs_error", "covered95", "covered50"]
    for season, part in scored.groupby("season"):
        out = root / season / args.city / "scores.csv"
        part[keep].to_csv(out, index=False)
        done = part[np.isfinite(part["wis"])]
        if done.empty:
            print(f"{season}: no forecast has reached its target week yet ({out})")
            continue
        summary = done.groupby(["horizon", "model"]).agg(
            weeks=("origin_date", "nunique"), wis=("wis", "mean"), coverage95=("covered95", "mean"),
            coverage50=("covered50", "mean"))
        if "persistence" in done["model"].values:
            ref = summary.xs("persistence", level="model")["wis"]
            summary["relative_wis"] = summary["wis"] / summary.index.get_level_values("horizon").map(ref)
        print(f"{season} ({args.city}), scored through {truth.index.max().date()}:")
        print(summary.round(3).to_string())
        print(f"Outputs: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    commands = {"run": run, "backtest-part": backtest_part, "forecast-part": forecast_part,
                "assemble": assemble, "score": score}
    for name in commands:
        p = sub.add_parser(name)
        p.add_argument("--city", required=True)
        p.add_argument("--root", default="live")
        p.add_argument("--data", help="city folder (default <root>/<city>)")
        if name == "score":
            continue
        p.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
        p.add_argument("--seasons", nargs="+", type=int,
                       help="trust-check seasons (default: the last 3 complete)")
        p.add_argument("--as-of", help="pretend the data ends this week (sanity checks)")
        if name in ("run", "assemble"):
            p.add_argument("--force", action="store_true")
        if name == "backtest-part":
            p.add_argument("--index", type=int, required=True)
        if name == "forecast-part":
            p.add_argument("--horizon", type=int, required=True, choices=HORIZONS)
    args = parser.parse_args()
    commands[args.command](args)


if __name__ == "__main__":
    main()
