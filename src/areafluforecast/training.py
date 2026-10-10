"""Training loops, MC-Dropout inference and interval calibration.

Copied from the research code (influenza/training.py, run_lstm.py,
run_colagnn.py) so that the same configuration and seed train the same model.
"""

from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .intervals import IntervalModel, fit_intervals
from .samples import Normalization, Samples

EpochCallback = Callable[[int, float], None] | None


def seed_everything(seed: int) -> None:
    """Seed at the point of use and pin torch to one thread.

    Multithreaded reductions sum in a nondeterministic order; because early
    stopping compares validation losses, a float tie broken differently selects a
    different epoch. One thread makes the seed an actual guarantee, and at this
    model size it is faster besides.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def masked_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE over observed cells only; NaN targets are missing or suppressed weeks."""
    observed = torch.isfinite(target)
    if not bool(observed.any()):
        return prediction.sum() * 0.0
    return ((prediction[observed] - target[observed]) ** 2).mean()


def masked_huber(prediction: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    observed = torch.isfinite(target)
    if not bool(observed.any()):
        return prediction.sum() * 0.0
    return F.huber_loss(prediction[observed], target[observed], delta=delta)


def soft_pearson(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - Pearson r over the observed cells of a batch."""
    observed = torch.isfinite(target)
    if int(observed.sum()) < 3:
        return prediction.sum() * 0.0
    p = prediction[observed] - prediction[observed].mean()
    y = target[observed] - target[observed].mean()
    return 1.0 - (p * y).sum() / (p.norm() * y.norm() + 1e-8)


def composite_loss(prediction, target, *, huber_delta: float = 0.0,
                   pearson_weight: float = 0.0) -> torch.Tensor:
    point = (masked_mse(prediction, target) if huber_delta <= 0
             else masked_huber(prediction, target, delta=huber_delta))
    if pearson_weight <= 0:
        return point
    return point + pearson_weight * soft_pearson(prediction, target)


@dataclass
class TrainResult:
    best_epoch: int
    best_val_loss: float
    epochs_run: int


# --- graph model -----------------------------------------------------------

def _stgnn_tensors(samples: Samples, device: torch.device) -> list[tuple]:
    return [(torch.from_numpy(samples.X[i]).to(device),
             torch.from_numpy(samples.g[i]).to(device),
             torch.from_numpy(samples.y[i]).to(device),
             torch.from_numpy(samples.anchors[i]).to(device),
             torch.from_numpy(samples.clim[i]).to(device))
            for i in range(len(samples.positions))]


def train_stgnn(model: nn.Module, train: Samples, val: Samples, *, relations: list[tuple],
                config, device: torch.device, n_neigh: int, blended: bool = True,
                on_epoch: EpochCallback = None) -> TrainResult:
    """Mini-batch training with cosine annealing over a fixed budget; the best
    validation epoch (plain masked MSE) is restored at the end. With val=None it
    trains the full budget and keeps the last epoch (the refit)."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    relations = [(name, index.to(device), weight.to(device)) for name, index, weight in relations]
    tensors = _stgnn_tensors(train, device)
    val_tensors = _stgnn_tensors(val, device) if val is not None else None

    def run_batch(rows, source):
        predictions, targets = [], []
        for idx in rows:
            X, g, y, anchor, clim = source[idx]
            kwargs = dict(anchor=anchor, clim=clim) if blended else {}
            predictions.append(model(X, relations, g, **kwargs)[:n_neigh])
            targets.append(y)
        return torch.cat(predictions, dim=0), torch.cat(targets, dim=0)

    best_val, best_state, best_epoch, patience = np.inf, None, 0, 0
    epoch = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = np.random.permutation(len(tensors))
        for start in range(0, len(order), config.batch_size):
            rows = [int(i) for i in order[start:start + config.batch_size]]
            optimizer.zero_grad()
            prediction, target = run_batch(rows, tensors)
            loss = composite_loss(prediction, target, huber_delta=config.huber_delta,
                                  pearson_weight=config.pearson_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
            optimizer.step()
        model.eval()
        scheduler.step()
        if val_tensors is None:
            best_epoch = epoch
            continue
        with torch.no_grad():
            prediction, target = run_batch(range(len(val_tensors)), val_tensors)
            val_loss = float(masked_mse(prediction, target).item())
        if val_loss < best_val:
            best_val, best_epoch, patience = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if on_epoch is not None:
            on_epoch(epoch, val_loss)
    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(best_epoch=best_epoch, best_val_loss=float(best_val), epochs_run=epoch)


def mc_dropout_stgnn(model: nn.Module, samples: Samples, *, relations: list[tuple],
                     norm: Normalization, n_neigh: int, device: torch.device, n_mc: int,
                     blended: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(draws, actual) in rate units; draws are (n_mc, S, n_areas, n_horizons)."""
    relations = [(name, index.to(device), weight.to(device)) for name, index, weight in relations]
    tensors = _stgnn_tensors(samples, device)
    model.train()                          # dropout on; the model has no BatchNorm
    draws = []
    with torch.no_grad():
        for _ in range(n_mc):
            per_sample = []
            for X, g, _y, anchor, clim in tensors:
                kwargs = dict(anchor=anchor, clim=clim) if blended else {}
                per_sample.append(model(X, relations, g, **kwargs)[:n_neigh].cpu().numpy())
            draws.append(np.stack(per_sample, axis=0))
    model.eval()
    normalized = np.stack(draws, axis=0)
    level = norm.to_level(normalized.reshape(-1, *normalized.shape[2:])).reshape(normalized.shape)
    return level, norm.to_level(samples.y)


def assert_mc_dropout_valid(model: nn.Module) -> None:
    """MC-Dropout calls model.train() at inference, which a BatchNorm would corrupt."""
    offenders = sorted({type(m).__name__ for m in model.modules()
                        if isinstance(m, (nn.Dropout, nn.BatchNorm1d, nn.BatchNorm2d))})
    if offenders:
        raise TypeError(f"MC-Dropout needs functional dropout only; found {offenders}.")


def calibrate_intervals(draws: np.ndarray, val_draws: np.ndarray, val_actual: np.ndarray,
                        horizons: tuple[int, ...], two_sided: bool = False):
    """Mean, 95% band and 23 quantiles from MC spread plus level-dependent noise."""
    mean = draws.mean(axis=0)
    epistemic = draws.var(axis=0)
    grid = np.broadcast_to(np.asarray(horizons), mean.shape)
    if val_draws is not None and val_draws.shape[1] > 1:
        val_mean = val_draws.mean(axis=0)
        val_grid = np.broadcast_to(np.asarray(horizons), val_mean.shape)
        model = fit_intervals(val_mean.ravel(), val_actual.ravel(), val_grid.ravel(),
                              epistemic=val_draws.var(axis=0).ravel(), two_sided=two_sided)
    else:
        model = IntervalModel(fitted=False, note="no validation split available")
    lower, upper = model.bounds(mean.ravel(), grid.ravel(), epistemic=epistemic.ravel())
    quantiles = model.quantiles(mean.ravel(), grid.ravel(), epistemic=epistemic.ravel())
    return (mean, lower.reshape(mean.shape), upper.reshape(mean.shape),
            quantiles.reshape(*mean.shape, -1), model)


# --- LSTM and Cola-GNN -------------------------------------------------------

SEQUENCE_PATIENCE = 25


def train_sequence_model(model: nn.Module, train, val, *, config, device: torch.device,
                         seed: int, use_clim: bool, on_epoch: EpochCallback = None):
    """Adam, masked MSE, early stopping after 25 epochs without improvement.

    The research LSTM and Cola-GNN runners each had this loop; it is one here.
    """
    x_train, y_train, c_train = train
    x_val, y_val, c_val = val
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    generator = torch.Generator().manual_seed(seed)
    tensors = [torch.from_numpy(x_train), torch.from_numpy(y_train)]
    if use_clim:
        tensors.append(torch.from_numpy(c_train))
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*tensors),
                                         batch_size=config.batch_size, shuffle=True,
                                         generator=generator)
    val_x = torch.from_numpy(x_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)
    val_c = torch.from_numpy(c_val).to(device)

    best_state, best_loss, best_epoch, patience = None, np.inf, 0, 0
    epoch = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        for batch in loader:
            batch = [b.to(device) for b in batch]
            optimizer.zero_grad()
            prediction = model(batch[0], batch[2]) if use_clim else model(batch[0])
            loss = masked_mse(prediction, batch[1])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = model(val_x, val_c) if use_clim else model(val_x)
            val_loss = float(masked_mse(prediction, val_y).item())
        if val_loss < best_loss - 1e-7:
            best_loss, best_epoch, patience = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience += 1
        if on_epoch is not None:
            on_epoch(epoch, val_loss)
        if patience >= SEQUENCE_PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(best_epoch=best_epoch, best_val_loss=float(best_loss), epochs_run=epoch)
