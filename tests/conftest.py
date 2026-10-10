import numpy as np
import pandas as pd
import pytest


def synthetic_rates(n_weeks: int = 260, areas=("a", "b", "c", "d"), seed: int = 0) -> pd.DataFrame:
    """Seasonal curves peaking each January, with noise, on Sunday-start weeks."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2019-06-02", periods=n_weeks, freq="7D")
    phase = 2 * np.pi * (index.dayofyear.to_numpy() - 20) / 365.25
    base = 5 + 40 * np.clip(np.cos(phase), 0, None) ** 3
    data = {a: np.maximum(0, base * (1 + 0.2 * i) + rng.normal(0, 2, n_weeks))
            for i, a in enumerate(areas)}
    return pd.DataFrame(data, index=index)


@pytest.fixture
def rates():
    return synthetic_rates()


@pytest.fixture
def line_edges():
    return [("a", "b"), ("b", "c"), ("c", "d")]
