"""The public API's promises: what is exported, saved-folder formats, seeding, the CLI."""

import json
import random

import numpy as np
import pandas as pd
import pytest
import torch

import areafluforecast as af
from areafluforecast.cli import EXIT_DATA, EXIT_USAGE, main


@pytest.fixture
def city(rates, line_edges):
    return af.City(name="t", rates=rates, edges=line_edges)


def test_public_names_resolve():
    for name in af.__all__:
        assert getattr(af, name) is not None
    assert issubclass(af.DataError, ValueError) and issubclass(af.DataWarning, UserWarning)


def test_results_take_keywords_only(rates, line_edges):
    with pytest.raises(TypeError):
        af.City("t", rates, line_edges)


def test_city_folder_is_stamped(city, tmp_path):
    folder = af.save_city(city, tmp_path / "t")
    info = json.loads((folder / "city.json").read_text())
    assert info["format"] == "areafluforecast-city-1" and info["package_version"] == af.__version__
    assert af.load_city(folder).areas == city.areas
    (folder / "city.json").write_text(json.dumps({**info, "format": "areafluforecast-city-9"}))
    with pytest.raises(ValueError, match="Upgrade the package"):
        af.load_city(folder)


def test_forecast_folder_round_trip(city, tmp_path):
    with pytest.warns(af.TrustWarning):
        result = af.forecast(city, models=["persistence"], horizons=(1, 2), out_dir=tmp_path,
                             progress=False)
    info = json.loads((result.path / "forecast.json").read_text())
    assert info["format"] == "areafluforecast-forecast-1"
    again = af.ForecastResult.load(result.path)
    pd.testing.assert_frame_equal(again.table, result.table, check_dtype=False)
    pd.testing.assert_frame_equal(again.history, result.history, check_freq=False)
    assert again.warnings == result.warnings and again.city == "t"
    assert again.meta["horizons"] == [1, 2]
    (result.path / "forecast.json").write_text(json.dumps({**info, "format": "areafluforecast-forecast-2"}))
    with pytest.raises(ValueError, match="Upgrade the package"):
        af.ForecastResult.load(result.path)


def test_backtest_folder_is_stamped(city, tmp_path):
    bt = af.backtest(city, models=[], seasons=1, horizons=(1,), out_dir=tmp_path, progress=False)
    info = json.loads((bt.path / "backtest.json").read_text())
    assert info["format"] == "areafluforecast-backtest-1"
    (bt.path / "backtest.json").write_text(json.dumps({**info, "format": "areafluforecast-backtest-2"}))
    with pytest.raises(ValueError, match="Upgrade the package"):
        af.BacktestResult.load(bt.path)


@pytest.mark.slow
def test_training_leaves_the_session_alone_and_seed_applies(city):
    random.seed(5)
    np.random.seed(5)
    torch.manual_seed(5)
    threads = torch.get_num_threads()
    expected = (random.random(), np.random.rand(), torch.rand(1).item())
    random.seed(5)
    np.random.seed(5)
    torch.manual_seed(5)
    bt = af.backtest(city, models=["lstm"], seasons=1, horizons=(1,), seed=7, out_dir=None,
                     progress=False, configs={"lstm": {"n_seeds": 1, "epochs": 1}})
    assert (random.random(), np.random.rand(), torch.rand(1).item()) == expected
    assert torch.get_num_threads() == threads
    assert {c["seed"] for k, c in bt.configs.items() if k.startswith("lstm")} == {7}
    assert bt.meta["seed"] == 7


def test_seed_reaches_every_seed_field():
    from areafluforecast.configs import with_seed
    config = with_seed(af.resolve("gnn_st"), 7)
    assert (config.seed, config.mc_seed) == (7, 88)
    assert with_seed(af.resolve("persistence"), 7) == af.resolve("persistence")
    assert with_seed(af.resolve("arima"), 7) == af.resolve("arima")


# --- command line ----------------------------------------------------------------

def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0 and af.__version__ in capsys.readouterr().out


def write_city(folder, rates, edges):
    folder.mkdir()
    rates.rename_axis("week").to_csv(folder / "rates.csv", date_format="%Y-%m-%d")
    pd.DataFrame(edges, columns=["a", "b"]).to_csv(folder / "edges.csv", index=False)
    (folder / "city.json").write_text(json.dumps({"rates": "rates.csv", "edges": "edges.csv",
                                                  "name": "t"}))


def test_cli_city_json_paths_are_relative_to_it(rates, line_edges, tmp_path, monkeypatch, capsys):
    write_city(tmp_path / "data", rates, line_edges)
    monkeypatch.chdir(tmp_path)
    assert main(["check", "--city", "data/city.json"]) == 0
    assert "4 areas" in capsys.readouterr().out


def test_cli_data_errors_exit_cleanly(rates, line_edges, tmp_path, capsys):
    write_city(tmp_path / "data", rates, [*line_edges, ("d", "zz")])
    assert main(["check", "--city", str(tmp_path / "data" / "city.json")]) == EXIT_DATA
    err = capsys.readouterr().err
    assert "areaflu: error:" in err and "zz" in err and "Traceback" not in err


def test_cli_usage_errors(capsys):
    assert main(["backtest", "--sample", "boston", "--init-from", "x.pt", "y.pt"]) == EXIT_USAGE
    assert "H=PATH" in capsys.readouterr().err
    assert main(["backtest", "--sample", "boston", "--models", "nope"]) == EXIT_USAGE
    assert "Unknown model" in capsys.readouterr().err


# --- weather ---------------------------------------------------------------------

def test_weather_does_not_retry_a_rejected_request(monkeypatch):
    import requests

    from areafluforecast import weather

    calls = []

    def fake_get(*args, **kwargs):
        calls.append(1)
        response = requests.Response()
        response.status_code = 400
        return response

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(weather.time, "sleep", lambda s: pytest.fail("should not wait"))
    with pytest.raises(RuntimeError, match="rejected the request"):
        weather.fetch_daily(42.0, -71.0, "2024-01-01", "2024-01-31")
    assert len(calls) == 1
