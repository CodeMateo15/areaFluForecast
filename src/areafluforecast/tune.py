"""Give every learned model the same tuning budget, chosen on validation weeks only.

For each model: 12 candidate configurations -- its default plus 11 random draws
from a search space of comparable breadth -- each trained once (one seed).
The candidate with the lowest error on the validation weeks (which come before
the scored window, never inside it) wins, and is then trained as the full
ensemble. Same search spaces and candidates as the research behind the package.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .city import City
from .configs import K_CONFIGS, SEARCH_SPACES, ModelConfig, candidate_flags, is_learned, with_flags
from .engine import run_model


def tuning_units(config: ModelConfig, k: int = K_CONFIGS) -> int:
    return k if is_learned(config) and _arm(config) in SEARCH_SPACES else 0


def _arm(config: ModelConfig) -> str:
    return config.kind


def tune_config(city: City, config: ModelConfig, *, horizon: int, test_start, test_end,
                rates=None, k: int = K_CONFIGS, on_unit=None) -> tuple[ModelConfig, list[dict]]:
    """The best of `k` candidates by validation mean absolute error, and every score."""
    if not tuning_units(config, k):
        return config, []
    scores = []
    best, best_mae = config, np.inf
    for index, flags in enumerate(candidate_flags(_arm(config), k)):
        candidate = replace(with_flags(config, flags), n_seeds=1)
        note = ""
        try:
            output = run_model(city, candidate, horizon=horizon, test_start=test_start,
                               test_end=test_end, rates=rates)
            val = output.val
            observed = np.isfinite(val["actual"]) & np.isfinite(val["predicted"])
            mae = float((val.loc[observed, "predicted"] - val.loc[observed, "actual"]).abs().mean())
        except ValueError as exc:      # e.g. a lookback too long for the history
            mae, note = np.inf, str(exc)
        scores.append({"candidate": index, "flags": " ".join(flags) or "(default)",
                       "val_mae": mae, "note": note})
        if mae < best_mae:
            best, best_mae = with_flags(config, flags), mae
        if on_unit:
            on_unit(f"tuning {config.kind} candidate {index + 1}/{k}")
    return best, scores
