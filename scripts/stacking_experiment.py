"""Would weighting the four models beat averaging them? An offline experiment.

Feature-weighted linear stacking (FWLS; Sill, Takacs, Mackey & Lin 2009,
arXiv:0911.0460) blends models with weights that are themselves linear in a few
"meta-features" of each forecast, w_i(x) = sum_j v_ij f_j(x), so the blend is
sum_ij v_ij f_j(x) g_i(x): one ridge regression on the products f_j g_i (their
eq. 5). With only the constant feature it is ordinary linear stacking.

Here g_i are the four models' point forecasts in the trust-check backtest, whose
every prediction is out of sample (each season's models trained only on the
weeks before it, as the frozen live models are). The fitted weights are applied
to every quantile, so a blend is a weighted quantile average like "avg":
  raw      weights as fitted (any sign, any sum); quantiles re-sorted
  simplex  weights clipped at 0 and rescaled to sum to 1 (Breiman 1996)
  wis      constant simplex weights chosen to minimise WIS directly

Scored by nested leave-one-season-out: for each held-out season the ridge
penalty and the meta-features (added greedily, one at a time, as the paper did)
are chosen by leave-one-season-out within the other two, then fitted on both
and scored on the held-out season, which none of the choices saw.

    python scripts/stacking_experiment.py --city boston
    python scripts/stacking_experiment.py --city chicago

Writes runs/<city>/stacking/report.md and scores.csv under --root; changes nothing
the live track reads.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import areafluforecast as af
from areafluforecast.backtest import BacktestResult
from areafluforecast.configs import DEFAULT_MODELS
from areafluforecast.intervals import QUANTILE_COLUMNS, QUANTILE_LEVELS
from areafluforecast.scoring import wis_from_quantiles

MEMBERS = list(DEFAULT_MODELS)
Q = list(QUANTILE_COLUMNS)
KEYS = ["season", "origin_date", "target_date", "horizon", "area"]
LAMBDAS = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
CANDIDATES = ("level", "trend", "phase", "disagreement", "noise")
I25, I75 = Q.index("q0.250"), Q.index("q0.750")
I025, I975 = Q.index("q0.025"), Q.index("q0.975")


# --- data ----------------------------------------------------------------------

def load_backtest(root: Path, city: str) -> BacktestResult:
    for meta in sorted((root / "runs" / city).glob("backtest_*/backtest.json"), reverse=True):
        if set(MEMBERS) <= set(json.loads(meta.read_text()).get("models", [])):
            return BacktestResult.load(meta.parent)
    raise SystemExit(f"No backtest of {city} has all of {MEMBERS}.")


def cells(bt: BacktestResult) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cells every member and persistence forecast and that have an actual:
    (keys + actual, point forecasts n x M, member quantiles n x M x 23,
    persistence quantiles n x 23, persistence point n)."""
    p = bt.predictions.assign(area=lambda d: d["area"].astype(str))
    p = p[np.isfinite(p["actual"])]
    by = {m: p[p["model"] == m].set_index(KEYS) for m in [*MEMBERS, "persistence"]}
    index = by["persistence"].index
    for m in MEMBERS:
        index = index.intersection(by[m].index)
    index = index.sort_values()
    frame = by["persistence"].loc[index, ["actual"]].reset_index()
    point = np.stack([by[m].loc[index, "predicted"].to_numpy(float) for m in MEMBERS], axis=1)
    quant = np.stack([by[m].loc[index, Q].to_numpy(float) for m in MEMBERS], axis=1)
    ref = by["persistence"].loc[index]
    return frame, point, quant, ref[Q].to_numpy(float), ref["predicted"].to_numpy(float)


def meta_features(frame: pd.DataFrame, point: np.ndarray, last: np.ndarray,
                  rates: pd.DataFrame) -> dict[str, np.ndarray]:
    """Each candidate as n x k columns, known at the forecast origin."""
    logs = np.log1p(rates.ffill())
    at = pd.MultiIndex.from_arrays([frame["origin_date"], frame["area"]])

    def look(table: pd.DataFrame) -> np.ndarray:
        return table.stack().reindex(at).to_numpy(float)

    level = np.log1p(last)
    week = frame["origin_date"].dt.isocalendar().week.to_numpy(float)
    angle = 2 * np.pi * week / 52.18
    return {
        "level": level[:, None],                                  # log of last reported week
        "trend": (level - look(logs.shift(3)))[:, None],          # 3-week change, log scale
        "phase": np.column_stack([np.sin(angle), np.cos(angle)]),  # time of year
        "disagreement": np.log1p(point.std(axis=1))[:, None],     # spread of the members
        "noise": look(logs.diff().rolling(26, min_periods=8).std())[:, None],  # recent volatility
    }


# --- blends --------------------------------------------------------------------

def to_simplex(w: np.ndarray) -> np.ndarray:
    w = np.clip(w, 0.0, None)
    total = w.sum(axis=1, keepdims=True)
    return np.where(total > 0, w / np.where(total > 0, total, 1.0), 1.0 / w.shape[1])


def apply(w: np.ndarray, quant: np.ndarray) -> np.ndarray:
    return np.sort(np.einsum("nm,nmq->nq", w, quant), axis=1)


def wis(y: np.ndarray, q: np.ndarray) -> float:
    return float(wis_from_quantiles(y, q, QUANTILE_LEVELS).mean())


class FWLS:
    """Ridge on the products of standardised meta-features and member forecasts."""

    def __init__(self, features: list[str], lam: float, simplex: bool):
        self.features, self.lam, self.simplex = features, lam, simplex

    def matrix(self, meta: dict, rows) -> np.ndarray:
        cols = [meta[f][rows] for f in self.features]
        return np.column_stack([np.ones(len(rows)), *cols]) if cols else np.ones((len(rows), 1))

    def fit(self, meta, point, y, rows):
        F = self.matrix(meta, rows)
        self.mu = np.nanmean(F[:, 1:], axis=0)
        self.sd = np.nanstd(F[:, 1:], axis=0) + 1e-9
        F = self.standardise(F)
        A = (F[:, :, None] * point[rows][:, None, :]).reshape(len(rows), -1)
        scale = np.sqrt((A ** 2).mean(axis=0)) + 1e-12
        As = A / scale
        # macOS Accelerate raises spurious FP warnings in matmul; the results are finite.
        with np.errstate(all="ignore"):
            v = np.linalg.solve(As.T @ As + self.lam * len(rows) * np.eye(As.shape[1]),
                                As.T @ y[rows])
        self.v = (v / scale).reshape(F.shape[1], point.shape[1])
        return self

    def standardise(self, F):
        F = F.copy()
        F[:, 1:] = np.nan_to_num((F[:, 1:] - self.mu) / self.sd)   # unknown -> training mean
        return F

    def weights(self, meta, rows):
        with np.errstate(all="ignore"):
            w = self.standardise(self.matrix(meta, rows)) @ self.v
        return to_simplex(w) if self.simplex else w


def fit_wis_simplex(quant, y, rows) -> np.ndarray:
    """Constant weights on the simplex minimising WIS (softmax-parametrised)."""
    def loss(z):
        w = np.exp(z - z.max())
        return wis(y[rows], np.einsum("m,nmq->nq", w / w.sum(), quant[rows]))
    m = quant.shape[1]
    # Start from equal weights with unit steps: the default simplex around zero is
    # too small to leave it.
    start = np.vstack([np.zeros(m), np.eye(m)])
    z = minimize(loss, np.zeros(m), method="Nelder-Mead",
                 options={"initial_simplex": start, "xatol": 1e-4, "fatol": 1e-7,
                          "maxiter": 4000}).x
    w = np.exp(z - z.max())
    return w / w.sum()


# --- nested leave-one-season-out ------------------------------------------------

def inner_cv(features, lam, simplex, meta, point, quant, y, season, train) -> float:
    """Mean held-out WIS over leave-one-season-out folds inside `train`."""
    scores = []
    for s in np.unique(season[train]):
        fit_rows, test_rows = train[season[train] != s], train[season[train] == s]
        model = FWLS(features, lam, simplex).fit(meta, point, y, fit_rows)
        scores.append(wis(y[test_rows], apply(model.weights(meta, test_rows), quant[test_rows])))
    return float(np.mean(scores))


def choose(simplex, meta, point, quant, y, season, train, greedy: bool):
    """Ridge penalty, and (greedy) meta-features added while they lower inner-CV WIS."""
    def best_lam(features):
        return min((inner_cv(features, lam, simplex, meta, point, quant, y, season, train), lam)
                   for lam in LAMBDAS)
    chosen, (score, lam) = [], best_lam([])
    while greedy:
        trials = [(*best_lam(chosen + [c]), c) for c in CANDIDATES if c not in chosen]
        if not trials:
            break
        trial_score, trial_lam, feature = min(trials)
        if trial_score >= score:
            break
        chosen, score, lam = chosen + [feature], trial_score, trial_lam
    return chosen, lam


def run_horizon(frame, point, quant, ref_q, meta, y) -> tuple[list[dict], list[dict]]:
    season = frame["season"].to_numpy()
    rows, notes = [], []
    for held in np.unique(season):
        train, test = np.flatnonzero(season != held), np.flatnonzero(season == held)
        blends = {"persistence": ref_q[test], "avg": quant[test].mean(axis=1)}
        blends |= {m: quant[test, i] for i, m in enumerate(MEMBERS)}
        w = fit_wis_simplex(quant, y, train)
        blends["stack-wis"] = apply(np.tile(w, (len(test), 1)), quant[test])
        notes.append({"season": held, "method": "stack-wis", "features": [],
                      "lambda": None, "mean_weights": dict(zip(MEMBERS, w.round(3)))})
        for simplex in (False, True):
            for greedy, name in ((False, "stack"), (True, "fwls")):
                label = f"{name}-{'simplex' if simplex else 'raw'}"
                features, lam = choose(simplex, meta, point, quant, y, season, train, greedy)
                model = FWLS(features, lam, simplex).fit(meta, point, y, train)
                weights = model.weights(meta, test)
                blends[label] = apply(weights, quant[test])
                notes.append({"season": held, "method": label, "features": features, "lambda": lam,
                              "mean_weights": dict(zip(MEMBERS, weights.mean(axis=0).round(3)))})
        for method, q in blends.items():
            per = wis_from_quantiles(y[test], q, QUANTILE_LEVELS)
            rows.append({"season": held, "method": method, "n_cells": len(test),
                         "n_weeks": frame["target_date"].iloc[test].nunique(),
                         "wis": float(per.mean()),
                         "coverage50": float(np.mean((q[:, I25] <= y[test]) & (y[test] <= q[:, I75]))),
                         "coverage95": float(np.mean((q[:, I025] <= y[test]) & (y[test] <= q[:, I975])))})
    return rows, notes


# --- report --------------------------------------------------------------------

ORDER = ["persistence", *MEMBERS, "avg", "stack-wis", "stack-raw", "stack-simplex", "fwls-raw",
         "fwls-simplex"]


def report(city: str, bt: BacktestResult, scores: pd.DataFrame, notes: pd.DataFrame,
           seconds: float) -> str:
    pooled = (scores.assign(total=scores["wis"] * scores["n_cells"])
              .groupby(["horizon", "method"])[["total", "n_cells"]].sum())
    pooled = (pooled["total"] / pooled["n_cells"]).unstack("method")[ORDER]
    rel = pooled.div(pooled["persistence"], axis=0)
    cov = scores.groupby(["horizon", "method"])[["coverage50", "coverage95"]].mean()
    lines = [f"# Stacking experiment: {city}", "",
             f"Backtest `{bt.path}` ({', '.join(bt.meta.get('season_labels', []))}); "
             f"every number is from a season the blend's weights, penalty and features never saw "
             f"(nested leave-one-season-out). {seconds:.0f} s.", "",
             "## Relative WIS (vs persistence, pooled over held-out seasons; lower is better)", "",
             "| weeks ahead | " + " | ".join(ORDER) + " |", "|---" * (len(ORDER) + 1) + "|"]
    for h, r in rel.iterrows():
        best = r.drop("persistence").idxmin()
        lines.append(f"| {h} | " + " | ".join(
            f"**{v:.3f}**" if m == best else f"{v:.3f}" for m, v in r.items()) + " |")
    lines += ["", "## Does weighting beat the average? Held-out seasons won (WIS lower than avg)", "",
              "| weeks ahead | " + " | ".join(m for m in ORDER if m not in ("persistence", "avg"))
              + " |", "|---" * (len(ORDER) - 1) + "|"]
    wide = scores.pivot_table(index=["horizon", "season"], columns="method", values="wis")
    for h, part in wide.groupby(level="horizon"):
        n = len(part)
        lines.append(f"| {h} | " + " | ".join(
            f"{int((part[m] < part['avg']).sum())}/{n}" for m in ORDER
            if m not in ("persistence", "avg")) + " |")
    lines += ["", "## Coverage of the 50% and 95% ranges (held-out, averaged over seasons)", "",
              "| weeks ahead | method | 50% | 95% |", "|---|---|---|---|"]
    for (h, m), r in cov.iterrows():
        if m in ("avg", "gnn_st", "stack-wis", "fwls-simplex", "fwls-raw"):
            lines.append(f"| {h} | {m} | {r['coverage50']:.3f} | {r['coverage95']:.3f} |")
    lines += ["", "## What each blend chose, per held-out season", "",
              "| weeks ahead | held out | method | meta-features | ridge λ | mean weights "
              f"({', '.join(MEMBERS)}) |", "|---|---|---|---|---|---|"]
    for _, r in notes[notes["method"] != "stack-raw"].iterrows():
        lam = "" if r["lambda"] is None else f"{r['lambda']:g}"
        lines.append(f"| {r['horizon']} | {r['season']} | {r['method']} | "
                     f"{', '.join(r['features']) or 'constant only'} | {lam} | "
                     f"{', '.join(f'{v:.2f}' for v in r['mean_weights'].values())} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--city", required=True)
    parser.add_argument("--root", default="live")
    args = parser.parse_args()
    root = Path(args.root)
    started = time.perf_counter()
    bt = load_backtest(root, args.city)
    rates = af.load_city(root / args.city, name=args.city, weather=False).rates
    frame, point, quant, ref_q, last = cells(bt)
    meta_all = meta_features(frame, point, last, rates)
    y_all = frame["actual"].to_numpy(float)
    scores, notes = [], []
    for h in sorted(frame["horizon"].unique()):
        rows = np.flatnonzero(frame["horizon"].to_numpy() == h)
        meta = {k: v[rows] for k, v in meta_all.items()}
        r, n = run_horizon(frame.iloc[rows].reset_index(drop=True), point[rows], quant[rows],
                           ref_q[rows], meta, y_all[rows])
        scores += [{"horizon": int(h), **x} for x in r]
        notes += [{"horizon": int(h), **x} for x in n]
        print(f"{args.city} {h} wk done ({time.perf_counter() - started:.0f} s)", flush=True)
    scores, notes = pd.DataFrame(scores), pd.DataFrame(notes)
    out = root / "runs" / args.city / "stacking"
    out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out / "scores.csv", index=False)
    notes.to_json(out / "choices.json", orient="records", indent=1)
    text = report(args.city, bt, scores, notes, time.perf_counter() - started)
    (out / "report.md").write_text(text)
    print(text)
    print(f"Outputs: {out}")


if __name__ == "__main__":
    main()
