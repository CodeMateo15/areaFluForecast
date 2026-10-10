"""The package must compute what the research code computed.

Runs only when the research repo is available:

    AREAFLU_RESEARCH_REPO=~/GNN-Influenza-Boston pytest -m research

Checks, on Boston's 2022-23 fold at 1 and 4 weeks ahead: identical model inputs,
identical graph, identical initial weights, and identical forecasts after a few
epochs of training, for the graph model, the LSTM and Cola-GNN. The full-length
comparison (10 seeds, tuned settings, three seasons) is the Explorer parity run
in scripts/parity/.
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.research

REPO = Path(os.environ.get("AREAFLU_RESEARCH_REPO", "~/GNN-Influenza-Boston")).expanduser()
if not (REPO / "Code" / "influenza").exists():
    pytest.skip("research repo not available", allow_module_level=True)
sys.path.insert(0, str(REPO / "Code"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import torch  # noqa: E402

from areafluforecast import configs, engine, graphs, samples, training  # noqa: E402
from areafluforecast.models import SpatioTemporalGNN, blend_init  # noqa: E402
from areafluforecast.windows import Window  # noqa: E402

START, END = pd.Timestamp("2022-06-01"), pd.Timestamp("2023-05-31")
EPOCHS = 3


@pytest.fixture(scope="module")
def city():
    from research_bridge import boston_from_research
    return boston_from_research()


@pytest.fixture(scope="module")
def research():
    from influenza import config as rconfig, graphs as rgraphs, samples as rsamples
    from influenza import training as rtraining, windows as rwindows, models as rmodels
    from influenza.cities import get
    return SimpleNamespace(config=rconfig, graphs=rgraphs, samples=rsamples, training=rtraining,
                           windows=rwindows, models=rmodels, city=get("boston"))


def _research_gnn(research, horizon, epochs=EPOCHS, n_seeds=1):
    r = research
    exp = r.config.EXPERIMENTS["gnn_st"] if hasattr(r.config, "EXPERIMENTS") else None
    if exp is None:
        exp = r.config._experiments()["gnn_st"]
    window = r.windows.Window(lookback=16, horizons=(horizon,), test_start=START, test_end=END,
                              val_mode="pre_test")
    rates = r.windows.variant_data(r.city.loaders.load_rates(), "full").available
    dataset = r.samples.load_dataset(exp.features, city=r.city, rates=rates)
    split = r.windows.split_origins(dataset.week_index,
                                    r.windows.valid_origins(dataset.week_index, window), window)
    history = dataset.rates.loc[dataset.rates.index < window.test_start]
    graph = r.graphs.build_graph(exp.graph, city=r.city, flu_history=history, static=dataset.static)
    smp, norm = r.samples.build_samples(dataset, split, exp.features, window, graph,
                                        target="blend", normalize="train")
    rel = graph.relation_tensors(["all"])
    spec = replace(exp.train, epochs=epochs)
    return SimpleNamespace(window=window, split=split, graph=graph, samples=smp, norm=norm,
                           relations=rel, spec=spec)


def _package_gnn(city, horizon, epochs=EPOCHS):
    config = replace(configs.PRESETS["gnn_st_paper"], epochs=epochs, n_seeds=1)
    window = engine.make_window(config, horizon, START, END)
    split = engine.make_split(city.rates.index, window)
    features = engine._features(city, config)
    dataset = samples.build_dataset(city, features)
    history = dataset.rates.loc[dataset.rates.index < window.test_start]
    graph = graphs.build_graph(engine.gnn_graph_spec(config), city=city, flu_history=history,
                               static=dataset.static)
    smp, norm = samples.build_samples(dataset, split, features, window, graph)
    return SimpleNamespace(config=config, window=window, split=split, graph=graph, samples=smp,
                           norm=norm, relations=graph.relation_tensors())


@pytest.mark.parametrize("horizon", [1, 4])
def test_gnn_inputs_and_graph_identical(city, research, horizon):
    r, p = _research_gnn(research, horizon), _package_gnn(city, horizon)
    assert r.split.train == p.split.train and r.split.val == p.split.val
    assert r.split.test == p.split.test
    np.testing.assert_array_equal(r.graph.W, p.graph.W)
    np.testing.assert_array_equal(r.samples.X, p.samples.X)
    np.testing.assert_array_equal(r.samples.y, p.samples.y)
    np.testing.assert_array_equal(r.samples.anchors, p.samples.anchors)
    np.testing.assert_array_equal(r.samples.clim, p.samples.clim)
    assert r.samples.feature_names == p.samples.feature_names


@pytest.mark.parametrize("horizon", [1, 4])
def test_gnn_training_identical(city, research, horizon):
    r, p = _research_gnn(research, horizon), _package_gnn(city, horizon)
    rm = research.models.SpatioTemporalGNN

    def research_model():
        s = r.samples
        return rm(lookback=s.lookback, temporal_channels=s.temporal_channels,
                  n_static_feat=s.n_static_feat, n_global=s.n_global, n_nodes=r.graph.n_nodes,
                  n_horizons=1, relation_names=["all"], hidden=r.spec.channels,
                  dropout=r.spec.dropout, dilations=r.spec.dilations,
                  adaptive_dim=r.spec.adaptive_dim, dropedge=r.spec.drop_edge,
                  blend_init=blend_init((horizon,)), horizon_steps=(horizon,),
                  n_blocks=r.spec.n_blocks)

    research.training.seed_everything(42)
    model_r = research_model()
    research.training.train_stgnn(model_r, r.samples.subset(_rows(r.samples, r.split.train)),
                                  r.samples.subset(_rows(r.samples, r.split.val)),
                                  relations=r.relations, spec=r.spec, device=torch.device("cpu"),
                                  n_neigh=14, blended=True, target_kind="blend", verbose=False)
    torch.manual_seed(123)
    draws_r, _ = research.training.mc_dropout_stgnn(
        model_r, r.samples.subset(_rows(r.samples, r.split.test)), relations=r.relations,
        norm=r.norm, n_neigh=14, device=torch.device("cpu"), n_mc=4, target="blend")

    training.seed_everything(42)
    s = p.samples
    model_p = SpatioTemporalGNN(lookback=s.lookback, temporal_channels=s.temporal_channels,
                                n_static_feat=s.n_static_feat, n_global=s.n_global,
                                n_nodes=p.graph.n_nodes, n_horizons=1, relation_names=["all"],
                                hidden=p.config.channels, dropout=p.config.dropout,
                                dilations=p.config.dilations, adaptive_dim=p.config.adaptive_dim,
                                dropedge=p.config.drop_edge, blend_init=blend_init((horizon,)),
                                n_blocks=p.config.n_blocks)
    training.train_stgnn(model_p, s.subset(p.split.train), s.subset(p.split.val),
                         relations=p.relations, config=p.config, device=torch.device("cpu"),
                         n_neigh=14)
    torch.manual_seed(123)
    draws_p, _ = training.mc_dropout_stgnn(model_p, s.subset(p.split.test), relations=p.relations,
                                           norm=p.norm, n_neigh=14, device=torch.device("cpu"),
                                           n_mc=4)
    for (k, a), b in zip(model_r.state_dict().items(), model_p.state_dict().values()):
        torch.testing.assert_close(a, b, rtol=0, atol=0, msg=k)
    np.testing.assert_array_equal(draws_r, draws_p)


def _rows(smp, positions):
    lookup = {p: i for i, p in enumerate(smp.positions)}
    return [lookup[p] for p in positions]


@pytest.mark.parametrize("kind", ["lstm", "colagnn"])
def test_sequence_models_identical(city, research, kind):
    """Package engine vs the research runner's own functions, end to end, 1 seed."""
    import run_lstm
    import run_colagnn

    horizon = 2
    if kind == "lstm":
        config = replace(configs.PRESETS["lstm"], epochs=EPOCHS, n_seeds=1, lookback=16,
                         target="blend", dropout=0.1)
    else:
        config = replace(configs.PRESETS["colagnn"], epochs=EPOCHS, n_seeds=1, lookback=12)
    package = engine.run_model(city, config, horizon=horizon, test_start=START, test_end=END)

    rw = research.windows
    window = rw.Window(lookback=config.lookback, horizons=(horizon,), test_start=START,
                       test_end=END, val_mode="pre_test")
    rates = rw.variant_data(research.city.loaders.load_rates(), "full").available
    split = rw.split_origins(rates.index, rw.valid_origins(rates.index, window), window)
    features, _ = research.samples.impute_causal(rates) if hasattr(research.samples, "impute_causal") \
        else __import__("influenza.data", fromlist=["impute_causal"]).impute_causal(rates)
    mean, std = rw.normalization(rates, split, mode="train")
    device = torch.device("cpu")
    import random
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if kind == "lstm":
        from influenza.climatology import fit_climatology
        clim = fit_climatology(rates, end=split.train[-1] + window.max_horizon + 1).level(rates.index)
        train = run_lstm.arrays(features, rates, split.train, mean, std, window, clim)
        val = run_lstm.arrays(features, rates, split.val, mean, std, window, clim)
        x_test, _, c_test = run_lstm.arrays(features, rates, split.test, mean, std, window, clim)
        args = SimpleNamespace(epochs=EPOCHS, batch_size=32, hidden_size=64, dropout=0.1,
                               lr=1e-3, weight_decay=1e-5)
        model, *_ = run_lstm.train_model(train, val, device, args, window, 14, 42,
                                         run_lstm._blend_init((horizon,)))
        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(x_test), torch.from_numpy(c_test)).numpy()
    else:
        train = run_lstm.arrays(features, rates, split.train, mean, std, window)
        val = run_lstm.arrays(features, rates, split.val, mean, std, window)
        x_test = run_lstm.arrays(features, rates, split.test, mean, std, window)[0]
        adjacency = run_colagnn.geographic_adjacency(research.city, rates)
        args = SimpleNamespace(epochs=EPOCHS, batch_size=32, hidden=32, conv_channels=16,
                               dropout=0.2, lr=1e-3, weight_decay=5e-4)
        model, *_ = run_colagnn.train_one(train, val, adjacency, device, args, window, 14, 42)
        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(x_test)).numpy()
    research_pred = np.maximum(0.0, pred * std[:, None, :] + mean[:, None, :])  # (S, H, N)
    ours = (package.test.sort_values(["origin_date", "area"])
            .pivot(index="origin_date", columns="area", values="predicted")
            [list(research.city.node_names)].to_numpy())
    np.testing.assert_allclose(ours, research_pred[:, 0, :], rtol=0, atol=1e-5)


def test_gnn_end_to_end_with_intervals(city, tmp_path):
    """research run_gnn.py (subprocess) vs package run_model: point, band and quantiles."""
    import subprocess

    from areafluforecast.intervals import QUANTILE_COLUMNS

    flags = ["--lr", "0.001", "--weight-decay", "0.0003", "--dropout", "0.25",
             "--channels", "32", "--epochs", "4"]
    command = [sys.executable, "-u", str(REPO / "Code" / "run_gnn.py"), "--experiment", "gnn_st",
               "--variant", "full", "--horizons", "1", "--test-start", str(START.date()),
               "--test-end", str(END.date()), "--val-mode", "pre_test", "--n-seeds", "2",
               "--no-carbon", "--output-dir", str(tmp_path / "out"),
               "--checkpoint-dir", str(tmp_path / "ckpt"), *flags]
    subprocess.run(command, check=True, capture_output=True, cwd=REPO,
                   env={**os.environ, "GNNFLU_LOG_DIR": str(tmp_path)})
    found = list((tmp_path / "out").rglob("predictions.csv"))
    assert found, "research run wrote no predictions"
    research_pred = pd.read_csv(found[0], parse_dates=["origin_date", "target_date"])

    config = configs.with_flags(replace(configs.PRESETS["gnn_st_paper"], n_seeds=2), flags)
    ours = engine.run_model(city, config, horizon=1, test_start=START, test_end=END).test
    research_pred = research_pred.rename(columns={"neighborhood": "area"})
    merged = ours.merge(research_pred, on=["origin_date", "area"], suffixes=("", "_r"))
    assert len(merged) == len(ours) == len(research_pred)
    for column in ["predicted", "lower", "upper", *QUANTILE_COLUMNS]:
        np.testing.assert_allclose(merged[column], merged[f"{column}_r"], rtol=1e-6, atol=1e-6,
                                   err_msg=column)
