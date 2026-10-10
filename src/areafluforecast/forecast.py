"""Forecast the weeks after the last observation, with a trust check.

Every model is trained on all weeks whose targets are already observed, then
forecasts from the last observed week. Before handing the numbers over, the
forecast looks up the most recent backtest for this city and says, per model and
horizon, whether that model actually beat persistence there.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import DEFAULT_RUN_DIR, BacktestResult, latest_backtest
from .checkpoint import load_checkpoint, save_checkpoint
from .city import City
from .configs import DEFAULT_MODELS, is_learned, resolve, to_json
from .engine import run_model, units
from .intervals import QUANTILE_COLUMNS, QUANTILE_LEVELS
from .progress import Progress
from .tune import tune_config, tuning_units

NO_BACKTEST = "not backtested"


class TrustWarning(UserWarning):
    """A forecast comes from a model that has not shown it beats persistence here."""


@dataclass
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
             weeks_of_history: int = 26, path: str | Path | None = None):
        from .plots import plot_forecast
        return plot_forecast(self, model=model, areas=areas, weeks_of_history=weeks_of_history,
                             path=path)


def _trust(backtest: BacktestResult | None, models, horizons) -> tuple[pd.DataFrame, list[str]]:
    rows, notes = [], []
    if backtest is None:
        notes.append("No backtest found for this city. These forecasts have not been checked "
                     "against persistence on your data; run areafluforecast.backtest(city) first.")
    for model in models:
        for horizon in horizons:
            verdict = backtest.verdict(model, horizon) if backtest is not None else None
            rows.append({"model": model, "horizon": horizon, "trust": verdict or NO_BACKTEST})
            if backtest is not None and verdict is None:
                notes.append(f"{model} was not backtested at {horizon} weeks ahead.")
            elif verdict and verdict != "better than persistence":
                notes.append(f"{model} at {horizon} weeks ahead: {verdict} in the backtest "
                             f"({', '.join(backtest.meta.get('season_labels', []))}).")
    return pd.DataFrame(rows), notes


def forecast(city: City, models=DEFAULT_MODELS, *, horizons=(1, 2, 3, 4),
             configs: dict | None = None, tune: bool = False,
             backtest: BacktestResult | str | Path | None = "latest",
             init_from: str | Path | None = None, zero_shot: bool = False,
             out_dir: str | Path | None = DEFAULT_RUN_DIR, save_models: bool = False,
             progress: bool = True, device: str | None = None) -> ForecastResult:
    """Forecast 1..max(horizons) weeks past the last observed week.

    backtest   "latest" looks up the newest saved backtest for this city; pass a
               BacktestResult or a saved backtest folder; None skips the check.
    init_from  a saved checkpoint to start the graph model from (transfer from a
               city with more history). zero_shot=True applies it without training.
    """
    configs = configs or {}
    resolved = {}
    for model in models:
        config = resolve(model, configs.get(model if isinstance(model, str) else model.kind))
        resolved[config.kind] = config
    init_members = None
    if init_from is not None:
        loaded_config, init_members, _ = load_checkpoint(init_from)
        # Labelled "pretrained", as backtest() labels it, so the trust check
        # matches this forecast to the pretrained model's own backtest verdict.
        resolved["pretrained"] = resolve(loaded_config, configs.get("pretrained"))
    horizons = tuple(sorted(int(h) for h in horizons))

    last = city.rates.index.max()
    if city.rates.loc[last].isna().all():
        raise ValueError(f"The last week ({last.date()}) has no data in any area; drop it so the "
                         "forecast starts from an observed week.")
    future = pd.date_range(last + pd.Timedelta(days=7), periods=max(horizons), freq="7D")
    extended = pd.concat([city.rates, pd.DataFrame(np.nan, index=future, columns=city.areas)])
    extended.index.name = city.rates.index.name
    origin = len(city.rates) - 1

    total = sum(units(c) + (tuning_units(c) if tune else 0) for c in resolved.values()) * len(horizons)
    bar = Progress(total, enabled=progress).start()
    frames, chosen, saved = [], {}, []
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_path = Path(out_dir) / city.name / f"forecast_{stamp}" if out_dir is not None else None
    for horizon in horizons:
        for name, config in resolved.items():
            use = config
            if tune and is_learned(config) and name != "pretrained":
                use, _ = tune_config(city, config, horizon=horizon, test_start=future[0],
                                     test_end=future[-1], rates=extended, on_unit=bar.step)
            pretrained = name == "pretrained"
            output = run_model(city, use, horizon=horizon, test_start=future[0], test_end=future[-1],
                               rates=extended, only_origin=origin,
                               init_members=init_members if pretrained else None,
                               zero_shot=zero_shot and pretrained,
                               on_unit=lambda m, h=horizon: bar.step(f"{h}wk {m}"), device=device)
            frames.append(output.test.assign(model=name))
            chosen[f"{name}|{horizon}"] = to_json(use)
            if save_models and run_path is not None and is_learned(use):
                saved.append(str(save_checkpoint(run_path / "models" / f"{name}_h{horizon}.pt",
                                                 output, city_name=city.name, areas=city.areas)))

    table = pd.concat(frames, ignore_index=True).drop(columns=["actual"])
    if backtest == "latest":
        backtest = latest_backtest(city, out_dir or DEFAULT_RUN_DIR)
    elif isinstance(backtest, (str, Path)):
        backtest = BacktestResult.load(backtest)
    trust, notes = _trust(backtest, list(resolved), horizons)
    table = table.merge(trust, on=["model", "horizon"], how="left")
    for note in notes:
        warnings.warn(note, TrustWarning, stacklevel=2)

    result = ForecastResult(
        city=city.name, table=table, history=city.rates, trust=trust, warnings=notes,
        meta={"last_observed_week": str(last.date()), "horizons": list(horizons),
              "configs": chosen, "backtest": str(backtest.path) if backtest is not None else None,
              "saved_models": saved, "created": stamp})
    if run_path is not None:
        run_path.mkdir(parents=True, exist_ok=True)
        result.to_csv(run_path / "forecast.csv")
        result.to_flusight().to_csv(run_path / "forecast_flusight.csv", index=False)
        (run_path / "trust.md").write_text(
            "# Trust check\n\n" + "\n".join(f"- {n}" for n in notes or ["Every model beat "
                                                                   "persistence in the backtest."]) + "\n")
        result.path = run_path
        bar.finish(run_path)
    return result
