import numpy as np
import pandas as pd

from areafluforecast.configs import DEFAULT_MODELS
from areafluforecast.ensemble import ENSEMBLE, average, with_average
from areafluforecast.intervals import QUANTILE_COLUMNS, QUANTILE_LEVELS
from areafluforecast.scoring import add_cell_scores, wis_from_quantiles


def member_table(seed: int = 0, areas=("A", "B"), horizons=(1, 2)) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for model in DEFAULT_MODELS:
        for h in horizons:
            for area in areas:
                q = np.sort(rng.uniform(0, 30, len(QUANTILE_LEVELS)))
                rows.append({"origin_date": pd.Timestamp("2026-10-18"),
                             "target_date": pd.Timestamp("2026-10-18") + pd.Timedelta(weeks=h),
                             "horizon": h, "area": area, "predicted": q[11], "lower": q[1],
                             "upper": q[-2], **dict(zip(QUANTILE_COLUMNS, q)), "model": model,
                             "trust": "better than persistence"})
    return pd.DataFrame(rows)


def test_average_of_identical_members_is_the_member():
    one = member_table().query("model == 'gnn_st'")
    same = pd.concat([one.assign(model=m) for m in DEFAULT_MODELS], ignore_index=True)
    avg = average(same).sort_values(["horizon", "area"]).reset_index(drop=True)
    expect = one.sort_values(["horizon", "area"]).reset_index(drop=True)
    cols = ["predicted", "lower", "upper", *QUANTILE_COLUMNS]
    np.testing.assert_allclose(avg[cols].to_numpy(), expect[cols].to_numpy())
    assert set(avg["model"]) == {ENSEMBLE}
    assert "trust" not in avg


def test_average_is_the_mean_and_stays_monotone():
    table = member_table(seed=3)
    avg = average(table)
    q = avg[list(QUANTILE_COLUMNS)].to_numpy()
    assert (np.diff(q, axis=1) >= 0).all() and (q >= 0).all()
    cell = table[(table["horizon"] == 2) & (table["area"] == "B")]
    row = avg[(avg["horizon"] == 2) & (avg["area"] == "B")].iloc[0]
    np.testing.assert_allclose(row[list(QUANTILE_COLUMNS)].to_numpy(float),
                               cell[list(QUANTILE_COLUMNS)].mean().to_numpy())


def test_cells_missing_a_member_are_left_out():
    table = member_table()
    table = table[~((table["model"] == "arima") & (table["area"] == "A"))]
    avg = average(table)
    assert set(avg["area"]) == {"B"}
    assert average(table[table["model"] != "arima"]).empty


def test_with_average_replaces_and_scores():
    table = with_average(with_average(member_table()))
    assert (table["model"] == ENSEMBLE).sum() == 4
    scored = add_cell_scores(average(member_table()).assign(actual=10.0))
    q = scored[list(QUANTILE_COLUMNS)].to_numpy()
    np.testing.assert_allclose(scored["wis"], wis_from_quantiles(np.full(len(q), 10.0), q))


def test_backtest_rows_keep_season_and_actual():
    table = member_table().assign(season="2025-26", actual=5.0)
    avg = average(table)
    assert list(avg["season"].unique()) == ["2025-26"]
    assert (avg["actual"] == 5.0).all()
