"""The three learned models.

* `SpatioTemporalGNN` ("gnn_st") -- the graph model from the paper: dilated causal
  convolutions over each area's history, graph mixing over the weighted
  adjacency plus a learned one, and a forecast measured from a learned blend of
  last week's level and the seasonal curve.
* `MultivariateLSTM` -- every area's series in one input vector; cross-area
  information but no graph.
* `ColaGNN` -- a reimplementation of Deng et al. (2020, CIKM) from the paper, not
  the authors' code: cross-location attention gated against the border graph.

Copied from the research code (influenza/models.py, run_lstm.py, run_colagnn.py)
so a checkpoint from either loads in the other. The graph model keeps dropout
functional and has no BatchNorm, so `model.train()` at inference turns on only
dropout -- which MC-Dropout uncertainty depends on.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedTemporalBlock(nn.Module):
    """One gated, causal (left-padded), dilated 1-D convolution with a residual."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int,
                 kernel_size: int = 2) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, 3 * out_channels, kernel_size, dilation=dilation)
        self.residual = (nn.Identity() if in_channels == out_channels
                         else nn.Conv1d(in_channels, out_channels, 1))
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padded = F.pad(x, (self.pad, 0))
        filt, gate, skip = torch.split(self.conv(padded), self.out_channels, dim=1)
        return torch.tanh(filt) * torch.sigmoid(gate) + skip + self.residual(x)


class MultiRelationalMixer(nn.Module):
    """A graph convolution per edge set plus a learned adaptive adjacency, gated."""

    def __init__(self, channels: int, relation_names: list[str], n_nodes: int,
                 *, adaptive_dim: int = 8, dropedge: float = 0.1) -> None:
        super().__init__()
        from torch_geometric.nn import GCNConv

        self.relation_names = list(relation_names)
        self.adaptive = adaptive_dim > 0
        # Self-loops are already in the relation tensors.
        self.convs = nn.ModuleList([GCNConv(channels, channels, add_self_loops=False)
                                    for _ in self.relation_names])
        if self.adaptive:
            self.adaptive_source = nn.Parameter(torch.randn(n_nodes, adaptive_dim) * 0.1)
            self.adaptive_target = nn.Parameter(torch.randn(n_nodes, adaptive_dim) * 0.1)
            self.adaptive_project = nn.Linear(channels, channels, bias=False)
        self.gate = nn.Parameter(torch.zeros(len(self.relation_names) + int(self.adaptive)))
        self.norm = nn.LayerNorm(channels)
        self.dropedge = dropedge

    def adaptive_adjacency(self) -> torch.Tensor | None:
        if not self.adaptive:
            return None
        return F.softmax(F.relu(self.adaptive_source @ self.adaptive_target.t()), dim=1)

    def forward(self, h: torch.Tensor, relations: list[tuple]) -> torch.Tensor:
        by_name = {name: (index, weight) for name, index, weight in relations}
        parts = [conv(h, *by_name[name]) for name, conv in zip(self.relation_names, self.convs)]
        adjacency = self.adaptive_adjacency()
        if adjacency is not None:
            if self.training and self.dropedge > 0:
                adjacency = adjacency * (torch.rand_like(adjacency) >= self.dropedge)
            parts.append(self.adaptive_project(adjacency @ h))
        weights = F.softmax(self.gate, dim=0)
        mixed = sum(w * part for w, part in zip(weights, parts))
        return self.norm(h + F.relu(mixed))

    def gate_weights(self) -> dict[str, float]:
        values = F.softmax(self.gate.detach(), dim=0).tolist()
        return dict(zip([*self.relation_names] + (["adaptive"] if self.adaptive else []), values))


class SpatioTemporalGNN(nn.Module):
    """Temporal conv stack -> graph mixing -> residual from a blended baseline.

        forecast_h = s_h * level_at_origin + (1 - s_h) * seasonal_curve(target week)
                     + network residual

    s_h is one learned number per horizon in [0, 1], so at worst the model
    collapses onto the better of the two fixed baselines.
    """

    def __init__(self, *, lookback: int, temporal_channels: int, n_static_feat: int,
                 n_global: int, n_nodes: int, n_horizons: int, relation_names: list[str],
                 hidden: int = 24, dropout: float = 0.2, dilations: tuple[int, ...] = (1, 2, 4, 8),
                 adaptive_dim: int = 8, dropedge: float = 0.1,
                 blend_init: tuple[float, ...] | None = None, n_blocks: int = 2) -> None:
        super().__init__()
        self.lookback = lookback
        self.temporal_channels = temporal_channels
        self.n_static_feat = n_static_feat
        self.n_horizons = n_horizons
        self.dropout = dropout
        blocks, channels = [], temporal_channels
        for dilation in dilations:
            blocks.append(DilatedTemporalBlock(channels, hidden, dilation))
            channels = hidden
        self.temporal = nn.ModuleList(blocks)
        self.static_fuse = nn.Linear(hidden + n_static_feat, hidden)
        self.mixers = nn.ModuleList([
            MultiRelationalMixer(hidden, relation_names, n_nodes,
                                 adaptive_dim=adaptive_dim, dropedge=dropedge)
            for _ in range(n_blocks)])
        self.head_fuse = nn.Linear(hidden + n_global, hidden)
        self.output = nn.Linear(hidden, n_horizons)
        if blend_init is None:
            blend_init = tuple([2.0] * n_horizons)
        self.blend_logit = nn.Parameter(torch.tensor(blend_init, dtype=torch.float32))

    def blend_weights(self) -> list[float]:
        """Learned weight on last week's level, per horizon (1 = pure persistence)."""
        return torch.sigmoid(self.blend_logit.detach()).tolist()

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        n_nodes = x.size(0)
        span = self.lookback * self.temporal_channels
        sequence = x[:, :span].view(n_nodes, self.lookback, self.temporal_channels)
        sequence = sequence.flip(1).transpose(1, 2)          # oldest first
        for block in self.temporal:
            sequence = block(sequence)
            sequence = F.dropout(sequence, p=self.dropout, training=self.training)
        h = sequence[:, :, -1]
        h = torch.cat([h, x[:, span:]] if self.n_static_feat else [h, x[:, span:span]], dim=1)
        return F.relu(self.static_fuse(h))

    def forward(self, x, relations, g_vec, anchor=None, clim=None):
        h = self._encode(x)
        for mixer in self.mixers:
            h = mixer(h, relations)
            h = F.dropout(h, p=self.dropout, training=self.training)
        if g_vec.numel():
            h = torch.cat([h, g_vec.unsqueeze(0).expand(h.size(0), -1)], dim=1)
        else:
            h = torch.cat([h, g_vec.new_zeros((h.size(0), 0))], dim=1)
        residual = self.output(F.relu(self.head_fuse(h)))
        if anchor is None or clim is None:
            return residual
        n_scored = anchor.size(0)
        share = torch.sigmoid(self.blend_logit).unsqueeze(0)
        base = share * anchor.unsqueeze(1) + (1.0 - share) * clim
        return torch.cat([base + residual[:n_scored], residual[n_scored:]], dim=0)


def blend_init(horizons) -> tuple[float, ...]:
    """Starting logits for the weight on last week's level: high at 1 week, low by 4."""
    scale = {1: 2.5, 2: 1.5, 3: 0.8, 4: 0.0}
    return tuple(scale.get(int(h), -1.5) for h in horizons)


class MultivariateLSTM(nn.Module):
    def __init__(self, n_inputs: int, n_horizons: int, hidden_size: int = 64,
                 dropout: float = 0.0, blend_init: tuple[float, ...] | None = None) -> None:
        super().__init__()
        self.n_inputs = n_inputs
        self.n_horizons = n_horizons
        self.lstm = nn.LSTM(input_size=n_inputs, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.output = nn.Linear(hidden_size, n_inputs * n_horizons)
        self.blend_logit = (None if blend_init is None else
                            nn.Parameter(torch.tensor(blend_init, dtype=torch.float32)))

    def forward(self, x: torch.Tensor, clim: torch.Tensor | None = None) -> torch.Tensor:
        encoded, _ = self.lstm(x)
        out = self.output(self.dropout(encoded[:, -1])).view(-1, self.n_horizons, self.n_inputs)
        if self.blend_logit is None:
            return out
        share = torch.sigmoid(self.blend_logit)[None, :, None]
        return share * x[:, -1, :][:, None, :] + (1.0 - share) * clim + out


class ColaGNN(nn.Module):
    def __init__(self, n_nodes: int, lookback: int, n_horizons: int, adjacency: np.ndarray,
                 hidden: int = 32, conv_channels: int = 16, gnn_hidden: int = 32,
                 dropout: float = 0.2, kernel: int = 3, long_dilation: int = 2) -> None:
        super().__init__()
        self.n_nodes, self.n_horizons = n_nodes, n_horizons
        self.register_buffer("a_geo", torch.tensor(adjacency, dtype=torch.float32))
        self.rnn = nn.GRU(input_size=1, hidden_size=hidden, batch_first=True)
        self.w_s = nn.Linear(hidden, hidden, bias=False)
        self.w_t = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, 1)
        self.gate = nn.Linear(n_nodes, n_nodes)
        self.conv_short = nn.Conv1d(1, conv_channels, kernel, dilation=1)
        self.conv_long = nn.Conv1d(1, conv_channels, kernel, dilation=long_dilation)
        short_len = lookback - (kernel - 1)
        long_len = lookback - long_dilation * (kernel - 1)
        if long_len < 1:
            raise ValueError(f"lookback {lookback} too short for dilation {long_dilation}")
        conv_features = conv_channels * 2
        self.conv_proj = nn.Linear(conv_channels * (short_len + long_len), conv_features)
        self.gnn1 = nn.Linear(conv_features, gnn_hidden)
        self.gnn2 = nn.Linear(gnn_hidden, gnn_hidden)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(gnn_hidden + hidden, n_horizons)

    def forward(self, x: torch.Tensor, clim: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, T, N) normalised rates -> (B, H, N). `clim` is accepted and ignored."""
        B, T, N = x.shape
        series = x.permute(0, 2, 1).reshape(B * N, T, 1)
        _, h_last = self.rnn(series)
        h = h_last[-1].view(B, N, -1)
        scores = self.v(F.elu(self.w_s(h)[:, :, None, :] + self.w_t(h)[:, None, :, :]))
        attention = F.normalize(scores.squeeze(-1), p=2, dim=-1)
        gate = torch.sigmoid(self.gate(attention))
        mixing = (1.0 - gate) * self.a_geo[None] + gate * attention
        raw = series.permute(0, 2, 1)
        conv = torch.cat([F.relu(self.conv_short(raw)).flatten(1),
                          F.relu(self.conv_long(raw)).flatten(1)], dim=1)
        features = F.relu(self.conv_proj(conv)).view(B, N, -1)
        g = F.elu(mixing @ self.gnn1(self.dropout(features)))
        g = F.elu(mixing @ self.gnn2(self.dropout(g)))
        out = self.out(self.dropout(torch.cat([g, h], dim=-1)))
        return out.permute(0, 2, 1)
