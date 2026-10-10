"""Would these models have worked here? Rolling-origin backtest on past seasons.

For each past season the models are trained only on weeks before it, then
forecast every week of it, exactly as they would have in real time (minus
reporting delays; see the README). Every model is compared with persistence
("next week looks like this week") on the same weeks.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .city import City
from .configs import BASELINES, DEFAULT_MODELS, ModelConfig, is_learned, resolve, to_json
from .engine import run_model, units
from .progress import Progress
from .scoring import add_cell_scores, compare_to_reference
from .timing import describe as describe_timing, timing_table
from .tune import tune_config, tuning_units

SIGNIFICANCE = 0.05
DEFAULT_RUN_DIR = Path("areaflu_runs")


def _verdict(row) -> str:
    if not np.isfinite(row["relative_wis"]):
        return "not enough data"
    clear = np.isfinite(row["p_holm"]) and row["p_holm"] < SIGNIFICANCE
    if row["relative_wis"] < 1.0:
        return "better than persistence" if clear else "not clearly better than persistence"
    return "worse than persistence" if clear else "no better than persistence"


@dataclass
class BacktestResult:
    city: str
    predictions: pd.DataFrame
    comparison: pd.DataFrame
    scores: pd.DataFrame
    timing: pd.DataFrame = field(default_factory=pd.DataFrame)
    configs: dict = field(default_factory=dict)
    tuning: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    path: Path | None = None

    def verdict(self, model: str, horizon: int) -> str | None:
        rows = self.comparison[(self.comparison["model"] == model)
                               & (self.comparison["horizon"] == horizon)]
        return None if rows.empty else str(rows.iloc[0]["verdict"])

    def report(self, *, show: bool = True) -> str:
        lines = [f"# Backtest: {self.city}", "",
                 f"Seasons: {', '.join(self.meta.get('season_labels', []))}. "
                 "Every number is from weeks the models never trained on.", "",
                 "**Relative WIS** is the model's weighted interval score divided by "
                 "persistence's on the same weeks: below 1 means it beat \"next week looks "
                 "like this week\". **p (Holm)** asks whether the weekly gap is bigger "
                 "than week-to-week noise, adjusted for the number of comparisons.", ""]
        table = self.comparison.copy()
        if not table.empty:
            table = table.sort_values(["horizon", "relative_wis"])
            lines.append("| weeks ahead | model | WIS | relative WIS | p (Holm) | verdict |")
            lines.append("|---|---|---|---|---|---|")
            for _, r in table.iterrows():
                lines.append(f"| {r['horizon']} | {r['model']} | {r['wis']:.2f} | "
                             f"{r['relative_wis']:.2f} | {r['p_holm']:.3f} | {r['verdict']} |")
        if not self.timing.empty:
            lines += ["", "**Timing.** Is the forecast late? *Weeks behind the curve* is how far "
                      "the forecast would have to be slid earlier to fit best; persistence is exactly "
                      "h weeks behind by construction. See areafluforecast.timing for why every "
                      "model here tends to run late.", ""]
            ordered = self.timing.sort_values(["horizon", "model"])
            lines += [f"- {describe_timing(r)}" for _, r in ordered.iterrows()]
        lines += ["", "Coverage of the 95% band (should be near 95%):", ""]
        cover = (self.scores.groupby(["horizon", "model"])["coverage95"].mean() * 100).round(1)
        lines += [f"- {m}, {h} wk: {v}%" for (h, m), v in cover.items()]
        text = "\n".join(lines)
        if show:
            print(text)
        return text

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.predictions.to_csv(path / "predictions.csv", index=False)
        self.comparison.to_csv(path / "comparison.csv", index=False)
        self.scores.to_csv(path / "scores.csv", index=False)
        self.timing.to_csv(path / "timing.csv", index=False)
        if self.tuning:
            pd.DataFrame(self.tuning).to_csv(path / "tuning.csv", index=False)
        (path / "report.md").write_text(self.report(show=False) + "\n")
        (path / "backtest.json").write_text(json.dumps(
            {"city": self.city, "configs": self.configs, **self.meta}, indent=2, default=str))
        self.path = path
        return path

    @classmethod
    def load(cls, path: str | Path) -> "BacktestResult":
        path = Path(path)
        meta = json.loads((path / "backtest.json").read_text())
        predictions = pd.read_csv(path / "predictions.csv",
                                  parse_dates=["origin_date", "target_date"])
        tuning_file = path / "tuning.csv"
        timing_file = path / "timing.csv"
        timing = pd.read_csv(timing_file) if timing_file.exists() else pd.DataFrame()
        return cls(city=meta.pop("city"), predictions=predictions,
                   comparison=pd.read_csv(path / "comparison.csv"),
                   scores=pd.read_csv(path / "scores.csv"), timing=timing,
                   configs=meta.pop("configs", {}),
                   tuning=pd.read_csv(tuning_file).to_dict("records") if tuning_file.exists() else [],
                   meta=meta, path=path)


def _seasons(city: City, seasons) -> list[int]:
    available = city.complete_seasons()
    if seasons is None:
        seasons = 2
    if isinstance(seasons, int):
        if not available:
            raise ValueError(f"{city.label} has no complete season with two years of history "
                             "before it, so there is nothing to backtest yet.")
        return available[-seasons:]
    chosen = [int(s) for s in seasons]
    return chosen


def backtest(city: City, models=DEFAULT_MODELS, *, seasons=None, horizons=(1, 2, 3, 4),
             configs: dict | None = None, tune: bool = False, out_dir: str | Path | None = DEFAULT_RUN_DIR,
             init_from: str | Path | None = None, zero_shot: bool = True,
             progress: bool = True, device: str | None = None) -> BacktestResult:
    """Train on the weeks before each past season and forecast that season.

    models   preset names ("gnn_st", "lstm", "colagnn", ...) or config objects.
             Persistence and the same-week-last-year baseline are always added.
    seasons  an int (the last n complete seasons, default 2) or season years
             (2023 = the season scored June 2023 - May 2024 for an Oct-Mar city).
    configs  {model name: {setting: value}} overrides, e.g. {"gnn_st": {"n_seeds": 3}}.
    tune     pick each learned model's settings per season from 12 candidates on
             validation weeks (12x the training time), as the paper did.
    init_from  a saved graph-model checkpoint (e.g. one trained in another city) to
             backtest alongside the other models, as "pretrained". zero_shot=True
             applies it with no training here; False fine-tunes it on each season's
             training weeks.
    """
    configs = configs or {}
    resolved, init_for = {}, {}
    for model in [*models, *BASELINES]:
        config = resolve(model, configs.get(model if isinstance(model, str) else model.kind))
        resolved[config.kind] = config
    if init_from is not None:
        from .checkpoint import load_checkpoint
        loaded, members, _ = load_checkpoint(init_from)
        resolved["pretrained"] = resolve(loaded, configs.get("pretrained"))
        init_for["pretrained"] = members
    years = _seasons(city, seasons)
    horizons = tuple(int(h) for h in horizons)

    total = sum((units(c) + (tuning_units(c) if tune else 0)) for c in resolved.values())
    total *= len(years) * len(horizons)
    bar = Progress(total, enabled=progress).start()
    if progress:
        print(city.summary(), flush=True)

    frames, tuning_rows, chosen = [], [], {}
    for year in years:
        start, end = city.season_window(year)
        label = city.season_label(year)
        for horizon in horizons:
            for name, config in resolved.items():
                use = config
                if tune and is_learned(config) and name not in init_for:
                    use, rows = tune_config(city, config, horizon=horizon, test_start=start,
                                            test_end=end, on_unit=bar.step)
                    tuning_rows += [{"season": label, "horizon": horizon, "model": name, **r}
                                    for r in rows]
                output = run_model(city, use, horizon=horizon, test_start=start, test_end=end,
                                   init_members=init_for.get(name),
                                   zero_shot=zero_shot and name in init_for,
                                   on_unit=lambda m, l=label, h=horizon: bar.step(f"{l} {h}wk {m}"),
                                   device=device)
                frame = output.test.assign(model=name, season=label)
                frames.append(frame)
                chosen[f"{name}|{label}|{horizon}"] = to_json(use)

    predictions = pd.concat(frames, ignore_index=True)
    result = _summarize(
        city.name, predictions, configs=chosen, tuning=tuning_rows,
        meta={"seasons": years, "season_labels": [city.season_label(y) for y in years],
              "horizons": list(horizons), "models": list(resolved), "tuned": tune,
              "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "last_week": str(city.rates.index.max().date())})
    if out_dir is not None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        result.save(Path(out_dir) / city.name / f"backtest_{stamp}")
        bar.finish(result.path)
    if progress:
        result.report()
    return result


def _summarize(city: str, predictions: pd.DataFrame, *, configs: dict, tuning: list,
               meta: dict) -> BacktestResult:
    scored = add_cell_scores(predictions)
    comparison = compare_to_reference(scored, "persistence")
    if not comparison.empty:
        comparison["verdict"] = comparison.apply(_verdict, axis=1)
    scores = (scored.groupby(["model", "season", "horizon"])
              .agg(n_cells=("wis", lambda s: int(np.isfinite(s).sum())), wis=("wis", "mean"),
                   mae=("abs_error", "mean"), coverage95=("covered95", "mean"),
                   coverage50=("covered50", "mean"))
              .reset_index())
    return BacktestResult(city=city, predictions=predictions, comparison=comparison,
                          scores=scores, timing=timing_table(predictions), configs=configs,
                          tuning=tuning, meta=meta)


def combine(parts: list[BacktestResult]) -> BacktestResult:
    """One backtest from several run separately (e.g. one cluster task per season and
    horizon). The comparison with persistence is recomputed on the pooled weeks, so
    it matches what a single backtest() over all of them would report."""
    if not parts:
        raise ValueError("Nothing to combine.")
    if len({p.city for p in parts}) != 1:
        raise ValueError(f"Parts come from different cities: {sorted({p.city for p in parts})}.")
    seasons = sorted({s for p in parts for s in p.meta.get("seasons", [])})
    labels = sorted({s for p in parts for s in p.meta.get("season_labels", [])})
    horizons = sorted({h for p in parts for h in p.meta.get("horizons", [])})
    models = list(dict.fromkeys(m for p in parts for m in p.meta.get("models", [])))
    configs = {k: v for p in parts for k, v in p.configs.items()}
    meta = {**parts[0].meta, "seasons": seasons, "season_labels": labels,
            "horizons": horizons, "models": models, "combined_from": len(parts),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    predictions = pd.concat([p.predictions for p in parts], ignore_index=True)
    return _summarize(parts[0].city, predictions, configs=configs,
                      tuning=[r for p in parts for r in p.tuning], meta=meta)


def latest_backtest(city: City, out_dir: str | Path = DEFAULT_RUN_DIR) -> BacktestResult | None:
    root = Path(out_dir) / city.name
    runs = sorted(root.glob("backtest_*/backtest.json")) if root.exists() else []
    return BacktestResult.load(runs[-1].parent) if runs else None
