"""Forecast the weeks after the last observation, with a trust check.

Every model is trained on all weeks whose targets are already observed, then
forecasts from the last observed week. Before handing the numbers over, the
forecast looks up the most recent backtest for this city and says, per model and
horizon, whether that model actually beat persistence there.
"""

from __future__ import annotations

import json
import time
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .backtest import DEFAULT_RUN_DIR, BacktestResult, latest_backtest
from .checkpoint import InitSource, describe_init, load_init, save_checkpoint
from .city import City
from .configs import DEFAULT_MODELS, ModelConfig, is_learned, resolve, to_json, with_seed
from .engine import run_model, units
from .errors import DataError, TrustWarning
from .formats import check, stamp
from .intervals import QUANTILE_COLUMNS, QUANTILE_LEVELS
from .progress import Progress
from .tune import tune_config, tuning_units

if TYPE_CHECKING:
    from matplotlib.figure import Figure

NO_BACKTEST = "not backtested"
OTHER_SETTINGS = "backtested with other settings"
SEED_FIELDS = ("seed", "mc_seed")


@dataclass(kw_only=True)
class ForecastResult:
    city: str
    table: pd.DataFrame          # one row per model x horizon x area
    history: pd.DataFrame        # the observed rates
    trust: pd.DataFrame          # model, horizon, verdict
    warnings: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    path: Path | None = None

    def to_csv(self, path: str | Path) -> Path:
        path = Path(path)
        self.table.to_csv(path, index=False)
        return path

    def save(self, path: str | Path) -> Path:
        """Write the forecast folder: forecast.csv, forecast_flusight.csv, history.csv,
        trust.md and forecast.json. `ForecastResult.load` reads it back."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.to_csv(path / "forecast.csv")
        self.to_flusight().to_csv(path / "forecast_flusight.csv", index=False)
        history = self.history.copy()
        history.index.name = "week"
        history.to_csv(path / "history.csv", date_format="%Y-%m-%d")
        (path / "trust.md").write_text(
            "# Trust check\n\n" + "\n".join(f"- {n}" for n in self.warnings or [
                "Every model beat persistence in the backtest."]) + "\n")
        meta = {k: v for k, v in self.meta.items() if k != "package_version"}
        (path / "forecast.json").write_text(json.dumps(
            {**stamp("forecast"), "city": self.city, "warnings": self.warnings, **meta},
            indent=2, default=str) + "\n")
        self.path = path
        return path

    @classmethod
    def load(cls, path: str | Path) -> ForecastResult:
        """A forecast folder written by `save` (or by `forecast()` with an out_dir)."""
        path = Path(path)
        meta = check(json.loads((path / "forecast.json").read_text()), "forecast", path)
        table = pd.read_csv(path / "forecast.csv", parse_dates=["origin_date", "target_date"])
        table["area"] = table["area"].astype(str)
        history = pd.read_csv(path / "history.csv", parse_dates=["week"]).set_index("week")
        history.columns = [str(c) for c in history.columns]
        trust = (table[["model", "horizon", "trust"]].drop_duplicates()
                 .reset_index(drop=True) if "trust" in table else pd.DataFrame())
        return cls(city=meta.pop("city"), table=table, history=history, trust=trust,
                   warnings=meta.pop("warnings", []), meta=meta, path=path)

    def to_flusight(self, model: str | None = None) -> pd.DataFrame:
        """FluSight hub layout: one row per quantile."""
        table = self.table if model is None else self.table[self.table["model"] == model]
        long = table.melt(id_vars=["model", "origin_date", "target_date", "horizon", "area"],
                          value_vars=list(QUANTILE_COLUMNS), var_name="column", value_name="value")
        levels = dict(zip(QUANTILE_COLUMNS, QUANTILE_LEVELS))
        return pd.DataFrame({
            "model": long["model"], "reference_date": long["origin_date"].dt.date,
            "target": "wk inc flu rate", "horizon": long["horizon"],
            "target_end_date": long["target_date"].dt.date, "location": long["area"],
            "output_type": "quantile", "output_type_id": long["column"].map(levels),
            "value": long["value"]}).sort_values(
                ["model", "location", "horizon", "output_type_id"]).reset_index(drop=True)

    def summary(self) -> pd.DataFrame:
        """Median and 95% band per model, horizon and area."""
        return self.table[["model", "horizon", "target_date", "area", "predicted", "lower",
                           "upper", "trust"]].sort_values(["model", "horizon", "area"])

    def plot(self, model: str | None = None, *, areas: list[str] | None = None,
             weeks_of_history: int = 26, path: str | Path | None = None) -> Figure:
        from .plots import plot_forecast
        return plot_forecast(self, model=model, areas=areas, weeks_of_history=weeks_of_history,
                             path=path)


def _settings_differ(backtest: BacktestResult, model: str, horizon: int,
                     settings: dict) -> str | None:
    """Why the backtest's verdict for `model` does not carry over to this forecast, if so."""
    meta = backtest.meta
    base = (meta.get("base_configs") or {}).get(model)
    ours = settings["configs"].get(model) or {}
    # Seeds pick one draw of the same model; they don't change what was tested.
    changed = sorted(k for k in set(base or {}) | set(ours)
                     if k not in SEED_FIELDS and (base or {}).get(k) != ours.get(k))
    if base is not None and changed:
        return f"its settings differ from the backtest's ({', '.join(changed)})"
    if model in settings["learned"] and bool(meta.get("tuned")) != settings["tuned"]:
        return ("the backtest tuned its settings and this forecast does not" if meta.get("tuned")
                else "this forecast tunes its settings and the backtest did not")
    if model == "pretrained":
        if meta.get("zero_shot") is not None and meta["zero_shot"] != settings["zero_shot"]:
            return ("the backtest applied it without training and this forecast trains it"
                    if meta["zero_shot"] else
                    "the backtest trained it and this forecast applies it without training")
        theirs = (meta.get("init_digest") or {}).get(str(horizon))
        ours = (settings.get("init_digest") or {}).get(horizon)
        if theirs is not None and theirs != ours:
            return "the backtest started from a different saved model"
    return None


def _trust(backtest: BacktestResult | None, models, horizons,
           settings: dict | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Per model and horizon, the backtest's verdict, or why there is none.

    `settings` (this forecast's configs, tuning and pretrained source) is checked
    against what the backtest ran; a verdict for other settings doesn't carry over.
    """
    rows, notes = [], []
    if backtest is None:
        notes.append("No backtest found for this city. These forecasts have not been checked "
                     "against persistence on your data; run areafluforecast.backtest(city) first.")
    elif backtest.meta.get("unverified"):
        notes.append(f"The backtest used ({backtest.path}) was saved by an older version and "
                     "cannot be checked against this data or these settings; run "
                     "areafluforecast.backtest(city) again.")
    check = settings is not None and backtest is not None and "base_configs" in backtest.meta
    for model in models:
        for horizon in horizons:
            verdict = backtest.verdict(model, horizon) if backtest is not None else None
            reason = _settings_differ(backtest, model, horizon, settings) if (
                check and verdict is not None) else None
            if reason:
                rows.append({"model": model, "horizon": horizon, "trust": OTHER_SETTINGS})
                notes.append(f"{model} at {horizon} weeks ahead: the backtest's verdict does "
                             f"not apply, because {reason}.")
                continue
            rows.append({"model": model, "horizon": horizon, "trust": verdict or NO_BACKTEST})
            if backtest is not None and verdict is None:
                notes.append(f"{model} was not backtested at {horizon} weeks ahead.")
            elif verdict and verdict != "better than persistence":
                notes.append(f"{model} at {horizon} weeks ahead: {verdict} in the backtest "
                             f"({', '.join(backtest.meta.get('season_labels', []))}).")
    return pd.DataFrame(rows), notes


def forecast(city: City, models: Sequence[str | ModelConfig] = DEFAULT_MODELS, *,
             horizons: Iterable[int] = (1, 2, 3, 4),
             configs: dict | None = None, tune: bool = False,
             backtest: BacktestResult | str | Path | None = "latest",
             init_from: InitSource | None = None, zero_shot: bool = False,
             seed: int | None = None, out_dir: str | Path | None = DEFAULT_RUN_DIR, save_models: bool = False,
             progress: bool = True, device: str | None = None) -> ForecastResult:
    """Forecast 1..max(horizons) weeks past the last observed week.

    backtest   "latest" looks up the newest saved backtest of this city's data in
               out_dir (nothing is looked up when out_dir is None); pass a
               BacktestResult or a saved backtest folder; None skips the check.
    init_from  a saved checkpoint to start the graph model from (transfer from a
               city with more history): a file or Checkpoint for a single horizon,
               or {horizon: file or Checkpoint}, such as af.pretrained("boston").
               zero_shot=True applies it without training.
    seed       derive every model's random seeds from this one number.
    out_dir    where the run is saved, as <out_dir>/<city name>/forecast_<time>/;
               None saves nothing (and looks up no backtest).
    save_models  also save each trained model there, as models/<model>_h<h>.pt.
    device     "cpu" (the default) or e.g. "cuda".
    """
    configs = configs or {}
    resolved = {}
    for model in models:
        config = resolve(model, configs.get(model if isinstance(model, str) else model.kind))
        resolved[config.kind] = config
    horizons = tuple(sorted(int(h) for h in horizons))
    init = None
    if init_from is not None:
        init = load_init(init_from, horizons)
        # Labelled "pretrained", as backtest() labels it, so the trust check
        # matches this forecast to the pretrained model's own backtest verdict.
        resolved["pretrained"] = resolve(init.config, configs.get("pretrained"))
    resolved = {name: with_seed(config, seed) for name, config in resolved.items()}

    last = city.rates.index.max()
    if city.rates.loc[last].isna().all():
        raise DataError(f"The last week ({last.date()}) has no data in any area; drop it so the "
                         "forecast starts from an observed week.")
    future = pd.date_range(last + pd.Timedelta(days=7), periods=max(horizons), freq="7D")
    extended = pd.concat([city.rates, pd.DataFrame(np.nan, index=future, columns=city.areas)])
    extended.index.name = city.rates.index.name
    origin = len(city.rates) - 1

    total = sum(units(c, city.n_areas) + (tuning_units(c) if tune else 0)
                for c in resolved.values()) * len(horizons)
    bar = Progress(total, enabled=progress).start()
    frames, chosen, saved = [], {}, []
    started = time.localtime()
    run_path = (Path(out_dir) / city.name / f"forecast_{time.strftime('%Y%m%d-%H%M%S', started)}"
                if out_dir is not None else None)
    for horizon in horizons:
        for name, config in resolved.items():
            use = config
            if tune and is_learned(config) and name != "pretrained":
                use, _ = tune_config(city, config, horizon=horizon, test_start=future[0],
                                     test_end=future[-1], rates=extended, on_unit=bar.step)
            pretrained = init is not None and name == "pretrained"
            output = run_model(city, use, horizon=horizon, test_start=future[0], test_end=future[-1],
                               rates=extended, only_origin=origin,
                               init_members=init.members[horizon] if pretrained else None,
                               init_features=init.feature_names[horizon] if pretrained else None,
                               zero_shot=zero_shot and pretrained,
                               on_unit=lambda m, h=horizon: bar.step(f"{h}wk {m}"), device=device)
            frames.append(output.test.assign(model=name))
            chosen[f"{name}|{horizon}"] = to_json(use)
            if save_models and run_path is not None and is_learned(use):
                saved.append(str(save_checkpoint(run_path / "models" / f"{name}_h{horizon}.pt",
                                                 output, city_name=city.name, areas=city.areas)))

    table = pd.concat(frames, ignore_index=True).drop(columns=["actual"])
    if isinstance(backtest, str) and backtest == "latest":
        backtest = latest_backtest(city, out_dir) if out_dir is not None else None
    elif isinstance(backtest, (str, Path)):
        backtest = BacktestResult.load(backtest)
    settings = {"configs": {name: to_json(c) for name, c in resolved.items()},
                "learned": {name for name, c in resolved.items()
                            if is_learned(c) and name != "pretrained"},
                "tuned": bool(tune), "zero_shot": bool(zero_shot),
                "init_digest": init.digests if init is not None else None}
    trust, notes = _trust(backtest, list(resolved), horizons, settings)
    table = table.merge(trust, on=["model", "horizon"], how="left")
    for note in notes:
        warnings.warn(note, TrustWarning, stacklevel=2)

    result = ForecastResult(
        city=city.name, table=table, history=city.rates, trust=trust, warnings=notes,
        meta={"last_observed_week": str(last.date()), "horizons": list(horizons),
              "configs": chosen, "backtest": str(backtest.path) if backtest is not None else None,
              "saved_models": saved, "created": time.strftime("%Y-%m-%dT%H:%M:%S", started),
              "seed": seed, "tuned": bool(tune),
              "init_from": describe_init(init_from)})
    if run_path is not None:
        result.save(run_path)
        bar.finish(run_path)
    return result
