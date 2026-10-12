"""check -> backtest -> forecast on the bundled Boston sample, with tiny models."""

import warnings

import numpy as np
import pandas as pd
import pytest

import areafluforecast as af
from areafluforecast.intervals import QUANTILE_COLUMNS

pytestmark = pytest.mark.slow
TINY = {"gnn_st": {"n_seeds": 1, "epochs": 2, "mc_samples": 4},
        "lstm": {"n_seeds": 1, "epochs": 2}, "colagnn": {"n_seeds": 1, "epochs": 2}}


@pytest.fixture(scope="module")
def city():
    return af.load_sample("boston")


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    return tmp_path_factory.mktemp("runs")


@pytest.fixture(scope="module")
def backtest(city, runs):
    return af.backtest(city, seasons=[2024], horizons=(1, 2), configs=TINY, out_dir=runs,
                       progress=False)


def test_sample_loads(city):
    assert city.n_areas == 14 and len(city.anchors) == 7 and city.has_weather
    assert city.complete_seasons()[-1] == 2024


def test_backtest(backtest, city):
    p = backtest.predictions
    assert set(p["model"]) == {"gnn_st", "lstm", "colagnn", "persistence", "seasonal_naive"}
    start, end = city.season_window(2024)
    assert p["target_date"].between(start, end).all()
    assert p[list(QUANTILE_COLUMNS)].notna().all().all()
    assert set(backtest.comparison["horizon"]) == {1, 2}
    assert backtest.comparison["verdict"].notna().all()
    assert (backtest.path / "report.md").exists()
    loaded = af.BacktestResult.load(backtest.path)
    assert loaded.verdict("gnn_st", 1) == backtest.verdict("gnn_st", 1)


def test_forecast_uses_latest_backtest(city, runs, backtest):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = af.forecast(city, horizons=(1, 2, 3), configs=TINY, out_dir=runs,
                             progress=False, save_models=True)
    table = result.table
    last = city.rates.index.max()
    assert table["origin_date"].eq(last).all()
    assert (table["target_date"] > last).all()
    assert len(table) == 3 * 3 * city.n_areas
    assert np.all(np.diff(table[list(QUANTILE_COLUMNS)].to_numpy(), axis=1) >= 0)
    # Horizon 3 was not in the backtest, so its trust says so.
    assert set(table.loc[table["horizon"] == 3, "trust"]) == {"not backtested"}
    assert any(issubclass(w.category, af.TrustWarning) for w in caught)
    flusight = result.to_flusight("gnn_st")
    assert len(flusight) == 3 * city.n_areas * len(QUANTILE_COLUMNS)
    assert (result.path / "forecast.csv").exists() and result.meta["saved_models"]


def test_no_backtest_warns(city, tmp_path):
    with pytest.warns(af.TrustWarning, match="No backtest"):
        af.forecast(city, models=["lstm"], horizons=(1,), configs=TINY, out_dir=tmp_path,
                    progress=False)


def test_zero_shot_from_checkpoint(city, tmp_path):
    """Train a transferable model, save it, apply it untrained (same city here)."""
    config = af.resolve("gnn_st_noadapt", {"n_seeds": 1, "epochs": 2, "mc_samples": 4})
    first = af.forecast(city, models=[config], horizons=(1,), out_dir=tmp_path, progress=False,
                        save_models=True, backtest=None)
    path = first.meta["saved_models"][0]
    again = af.forecast(city, models=[], horizons=(1,), init_from=path, zero_shot=True,
                        out_dir=tmp_path, progress=False, backtest=None)
    assert set(again.table["model"]) == {"pretrained"}
    assert again.table["predicted"].notna().all()


@pytest.fixture(scope="module")
def checkpoints(city, tmp_path_factory):
    """One saved transferable model per horizon, 1 and 2 weeks ahead."""
    config = af.resolve("gnn_st_noadapt", {"n_seeds": 1, "epochs": 2, "mc_samples": 4})
    first = af.forecast(city, models=[config], horizons=(1, 2), progress=False, save_models=True,
                        out_dir=tmp_path_factory.mktemp("ckpt"), backtest=None)
    return {int(p[-4]): p for p in first.meta["saved_models"]}


def test_backtest_pretrained(city, tmp_path, checkpoints):
    bt = af.backtest(city, models=[], seasons=[2024], horizons=(1, 2), init_from=checkpoints,
                     out_dir=tmp_path, progress=False)
    assert "pretrained" in set(bt.comparison["model"])
    fc = af.forecast(city, models=[], horizons=(1, 2), init_from=checkpoints, zero_shot=True,
                     out_dir=tmp_path, progress=False)
    assert fc.table.groupby("horizon")["trust"].first().to_dict() == {
        h: bt.verdict("pretrained", h) for h in (1, 2)}
    # The backtest applied the weights untrained, so it says nothing about fine-tuning them.
    with pytest.warns(af.TrustWarning, match="applied it without training"):
        tuned = af.forecast(city, models=[], horizons=(1,), init_from={1: checkpoints[1]},
                            out_dir=tmp_path, progress=False)
    assert set(tuned.table["trust"]) == {"backtested with other settings"}


def test_one_horizon_checkpoint_is_not_reused(city, checkpoints):
    with pytest.raises(ValueError, match="trained to forecast 1 weeks ahead, not 2"):
        af.forecast(city, models=[], horizons=(1, 2), init_from=checkpoints[1], zero_shot=True,
                    out_dir=None, progress=False, backtest=None)


def test_pretrained_inputs_must_match(city, checkpoints):
    with pytest.raises(ValueError, match="different input features"):
        af.forecast(city, models=[], horizons=(1,), init_from=checkpoints[1], zero_shot=True,
                    configs={"pretrained": {"use_weather": False}}, out_dir=None,
                    progress=False, backtest=None)


def test_no_out_dir_looks_nothing_up(city, monkeypatch, runs, backtest):
    monkeypatch.chdir(runs.parent)
    with pytest.warns(af.TrustWarning, match="No backtest"):
        af.forecast(city, models=["persistence"], horizons=(1,), out_dir=None, progress=False)


def test_plot(city, runs, backtest, tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    result = af.forecast(city, models=["lstm"], horizons=(1, 2), configs=TINY, out_dir=runs,
                         progress=False)
    result.plot(path=tmp_path / "f.png")
    assert (tmp_path / "f.png").stat().st_size > 1000


def test_cli_check(capsys):
    from areafluforecast.cli import main
    main(["check", "--sample", "boston"])
    assert "14 areas" in capsys.readouterr().out


def test_combine_matches_one_backtest(city, backtest):
    """Per-horizon parts, combined, report what one backtest over both reports."""
    from areafluforecast.backtest import combine
    parts = [af.backtest(city, ["gnn_st", "lstm", "colagnn"], seasons=[2024], horizons=(h,),
                         configs=TINY, out_dir=None, progress=False) for h in (1, 2)]
    merged = combine(parts)
    key = ["model", "horizon"]
    a = merged.comparison.sort_values(key).reset_index(drop=True)
    b = backtest.comparison.sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(a[key + ["verdict"]], b[key + ["verdict"]])
    assert merged.meta["horizons"] == [1, 2] and merged.meta["seasons"] == [2024]


def test_save_and_load_city_round_trip(city, tmp_path):
    af.save_city(city, tmp_path / "boston")
    again = af.load_city(tmp_path / "boston")
    pd.testing.assert_frame_equal(again.rates, city.rates, check_freq=False)
    assert again.edges == city.edges and again.anchor_edges == city.anchor_edges
    assert again.has_weather and again.label == city.label
