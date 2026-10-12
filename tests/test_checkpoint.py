"""Checkpoint safety, horizon matching, and matching backtests to the data and settings."""

from types import SimpleNamespace

import pandas as pd
import pytest
import torch

import areafluforecast as af
from areafluforecast.backtest import BacktestResult, data_fingerprint, latest_backtest
from areafluforecast.checkpoint import load_init
from areafluforecast.configs import to_json
from areafluforecast.forecast import NO_BACKTEST, OTHER_SETTINGS, _trust


class NotAllowed:
    """A class torch's weights-only loader refuses to rebuild."""


def save(path, horizon, *, channels=24):
    config = af.resolve("gnn_st_noadapt", {"channels": channels})
    output = SimpleNamespace(config=config, members=[{"w": torch.ones(2, 2) * horizon}],
                             meta={"horizon": horizon, "feature_names": ["flu_lag0"]})
    return af.save_checkpoint(path, output, city_name="t", areas=["a", "b"])


def test_round_trip_records_versions(tmp_path):
    config, members, payload = af.load_checkpoint(save(tmp_path / "h1.pt", 1))
    assert config.kind == "gnn_st_noadapt" and torch.equal(members[0]["w"], torch.ones(2, 2))
    assert payload["package_version"] == af.__version__
    assert isinstance(payload["torch_version"], str)


def test_newer_format_asks_for_upgrade(tmp_path):
    torch.save({"format": "areafluforecast-checkpoint-2", "config": {}}, tmp_path / "x.pt")
    with pytest.raises(ValueError, match="Upgrade the package"):
        af.load_checkpoint(tmp_path / "x.pt")


def test_not_a_checkpoint(tmp_path):
    torch.save({"something": 1}, tmp_path / "x.pt")
    with pytest.raises(ValueError, match="not an areaFluForecast checkpoint"):
        af.load_checkpoint(tmp_path / "x.pt")


def test_files_that_need_unpickling_are_refused(tmp_path):
    torch.save({"model_state_dict": {"w": torch.zeros(1)}, "extra": NotAllowed()},
               tmp_path / "research.pt")
    with pytest.raises(ValueError, match="could run code"):
        af.load_checkpoint(tmp_path / "research.pt")
    config, members, _ = af.load_checkpoint(tmp_path / "research.pt", trust_research_file=True)
    assert config.kind == "gnn_st_noadapt" and len(members) == 1


def test_one_horizon_file_is_not_reused_for_others(tmp_path):
    path = save(tmp_path / "h1.pt", 1)
    assert set(load_init(path, (1,)).members) == {1}
    with pytest.raises(ValueError, match="trained to forecast 1 weeks ahead, not 2"):
        load_init(path, (1, 2))


def test_horizon_map(tmp_path):
    paths = {h: save(tmp_path / f"h{h}.pt", h) for h in (1, 2)}
    init = load_init(paths, (2, 1))
    assert torch.equal(init.members[2][0]["w"], torch.ones(2, 2) * 2)
    assert init.digests[1] != init.digests[2] and init.feature_names[1] == ["flu_lag0"]
    with pytest.raises(ValueError, match=r"no checkpoint for \[3\]"):
        load_init(paths, (1, 2, 3))
    with pytest.raises(ValueError, match="trained to forecast 2 weeks ahead, not 1"):
        load_init({1: paths[2], 2: paths[2]}, (1, 2))
    other = save(tmp_path / "other.pt", 2, channels=8)
    with pytest.raises(ValueError, match="different model settings"):
        load_init({1: paths[1], 2: other}, (1, 2))


# --- matching a saved backtest to the data -------------------------------------

def saved_backtest(city, root, **meta):
    end = city.season_window(city.complete_seasons()[-1])[1]
    comparison = pd.DataFrame({"model": ["gnn_st"], "horizon": [1], "wis": [1.0],
                               "relative_wis": [0.8], "p_holm": [0.01],
                               "verdict": ["better than persistence"]})
    scores = pd.DataFrame({"model": ["gnn_st"], "horizon": [1], "coverage95": [0.95]})
    result = BacktestResult(city=city.name, predictions=pd.DataFrame(
        columns=["origin_date", "target_date"]), comparison=comparison, scores=scores,
        meta={"data_end": str(end.date()), "fingerprint": data_fingerprint(city, end), **meta})
    return result.save(root / city.name / "backtest_20260101-000000")


def test_latest_backtest_matches_the_data(rates, line_edges, tmp_path):
    city = af.City(name="rates", rates=rates, edges=line_edges)
    saved_backtest(city, tmp_path)
    assert latest_backtest(city, tmp_path).path is not None
    # Weeks added after the backtested seasons don't matter.
    more = pd.concat([rates, pd.DataFrame(1.0, columns=rates.columns,
                                          index=[rates.index[-1] + pd.Timedelta(days=7)])])
    assert latest_backtest(af.City(name="rates", rates=more, edges=line_edges), tmp_path)
    # Another dataset that happens to share the name does not get its verdicts.
    other = af.City(name="rates", rates=rates * 2, edges=line_edges)
    assert latest_backtest(other, tmp_path) is None


def test_old_backtests_are_marked_unverified(rates, line_edges, tmp_path):
    city = af.City(name="t", rates=rates, edges=line_edges)
    folder = saved_backtest(city, tmp_path)
    meta = (folder / "backtest.json").read_text().replace('"fingerprint"', '"old_fingerprint"')
    (folder / "backtest.json").write_text(meta)
    found = latest_backtest(city, tmp_path)
    assert found.meta["unverified"]
    _, notes = _trust(found, ["gnn_st"], (1,))
    assert any("older version" in n for n in notes)


def test_trust_needs_the_same_settings(rates, line_edges, tmp_path):
    city = af.City(name="t", rates=rates, edges=line_edges)
    full = to_json(af.resolve("gnn_st"))
    bt = BacktestResult.load(saved_backtest(city, tmp_path, base_configs={"gnn_st": full},
                                            tuned=False))

    def trust(config, tuned=False):
        settings = {"configs": {"gnn_st": config}, "learned": {"gnn_st"}, "tuned": tuned,
                    "zero_shot": False, "init_digest": None}
        return _trust(bt, ["gnn_st"], (1, 2), settings)

    table, notes = trust(full)
    assert list(table["trust"]) == ["better than persistence", NO_BACKTEST]
    quick = to_json(af.resolve("gnn_st", {"n_seeds": 1}))
    table, notes = trust(quick)
    assert table["trust"].iloc[0] == OTHER_SETTINGS and any("n_seeds" in n for n in notes)
    table, _ = trust(full, tuned=True)
    assert table["trust"].iloc[0] == OTHER_SETTINGS
