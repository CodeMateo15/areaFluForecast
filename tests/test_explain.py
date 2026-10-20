"""explain(): Owen values for a frozen graph model add up to its forecast."""

import numpy as np
import pandas as pd
import pytest
import torch

import areafluforecast as af
from areafluforecast import drivers as ex

pytestmark = pytest.mark.slow


def _as_of(city, week):
    from dataclasses import replace
    weather = {k: v.loc[:week] for k, v in city.weather.items()} if city.weather else None
    return replace(city, rates=city.rates.loc[:week], weather=weather)


def _frozen(city, folder, tiny):
    year = city.complete_seasons()[-1]
    anchor = city.season_window(year)[0]
    weeks = city.rates.index[city.rates.index >= anchor]
    early, late = _as_of(city, weeks[3]), _as_of(city, weeks[8])
    first = af.forecast(early, ("gnn_st",), horizons=(1, 2), configs={"gnn_st": tiny},
                        anchor=anchor, save_models=True, out_dir=folder, backtest=None,
                        progress=False)
    return late, first.path / "models"


@pytest.fixture(scope="module")
def city():
    return af.load_sample("boston")


@pytest.fixture(scope="module")
def tiny():
    return {"n_seeds": 2, "epochs": 2, "mc_samples": 4}


@pytest.fixture(scope="module")
def drivers(city, tiny, tmp_path_factory):
    late, models = _frozen(city, tmp_path_factory.mktemp("explain"), tiny)
    origins = late.rates.index[-2:]
    table = ex.explain(late, models, origins=origins, horizons=(1, 2), n_perm=16,
                       config=tiny)
    return late, models, table


def test_values_add_up_to_the_forecast(drivers, city):
    late, _, table = drivers
    total = table.groupby(["origin_date", "horizon", "area"]).agg(
        value=("value", "sum"), base=("base", "first"), output=("output", "first"))
    np.testing.assert_allclose(total["base"] + total["value"], total["output"], atol=1e-4)
    assert len(total) == 2 * 2 * late.n_areas
    players = table.groupby(["origin_date", "horizon", "area"]).size()
    assert (players == len(ex.OWN) + late.n_areas - 1).all()
    neighbors = table[table["driver"] == "neighbor"]
    assert set(neighbors["source_area"]) == set(late.areas)          # never an anchor node
    assert (neighbors["source_area"] != neighbors["area"]).all()


def test_output_is_the_ensembles_eval_forecast(drivers, tiny):
    """The explained forecast is the frozen members' eval-mode mean, through the
    model's own forward pass."""
    late, models, table = drivers
    config = af.resolve("gnn_st", tiny)
    checkpoint = af.Checkpoint.load(models / "gnn_st_h1.pt")
    last = late.rates.index.max()
    future = pd.date_range(last + pd.Timedelta(days=7), periods=2, freq="7D")
    rates = pd.concat([late.rates, pd.DataFrame(np.nan, index=future, columns=late.areas)])
    ens = ex._ensemble(late, checkpoint, config, rates, 1, future)
    row = [rates.index[p] for p in ens.positions].index(last)
    with torch.no_grad():
        mean = sum(m(ens.X[row], ens.relations, torch.zeros(0), anchor=ens.anchors[row],
                     clim=ens.clim[row])[:late.n_areas, 0] for m in ens.members) / len(ens.members)
    expected = mean.numpy() * ens.std + ens.mean
    got = (table[(table["origin_date"] == last) & (table["horizon"] == 1)]
           .drop_duplicates("area").set_index("area")["output"].reindex(late.areas))
    np.testing.assert_allclose(got.to_numpy(), expected, atol=1e-4)


def test_an_input_the_model_lacks_gets_nothing(city, tmp_path_factory):
    tiny = {"n_seeds": 1, "epochs": 2, "mc_samples": 2, "use_weather": False}
    late, models = _frozen(city, tmp_path_factory.mktemp("noweather"), tiny)
    table = ex.explain(late, models, horizons=(1,), n_perm=8, config=tiny)
    assert (table.loc[table["driver"] == "weather", "value"] == 0).all()


def test_unknown_origin_is_refused(drivers, tiny):
    late, models, _ = drivers
    with pytest.raises(af.DataError, match="not a scored origin"):
        ex.explain(late, models, origins=["2019-01-06"], horizons=(1,), n_perm=4, config=tiny)
