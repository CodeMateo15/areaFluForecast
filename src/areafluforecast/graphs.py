"""The weighted adjacency the graph model mixes over.

Edge types, all undirected, accumulated into one weight matrix W:

* **borders** -- weight 1 between areas that share a border (and between an
  area and an anchor), plus weaker multi-hop links between areas 2..max_hop
  borders apart (0.5 * decay^(hops - 1)).
* **correlation** -- areas whose flu series correlated above a threshold before
  the scored window, weighted by the correlation. Uses pre-scoring weeks only.
  Flu spreads along commuting links as well as borders, so areas far apart can
  move in step (Viboud et al. 2006, Science 312:447).
* **similarity** (off by default) -- areas with similar static characteristics.

Copied from the research code (influenza/graphs.py), minus its ablation-only
edge types.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .city import City


@dataclass(frozen=True)
class GraphSpec:
    anchors: bool = True
    geo: bool = True
    geo_max_hop: int = 3
    geo_decay: float = 0.5
    corr: bool = True
    corr_threshold: float = 0.85
    corr_binary: bool = False
    corr_coef: float = 1.0
    demo: bool = False
    demo_threshold: float = 0.5
    demo_coef: float = 1.0


@dataclass
class Graph:
    node_names: list[str]
    n_nodes: int
    n_neigh: int
    W: np.ndarray
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def n_anchors(self) -> int:
        return self.n_nodes - self.n_neigh

    def relation_tensors(self):
        """[("all", edge_index, edge_weight)] with a self-loop on every node."""
        import torch

        src, dst, weights = [], [], []
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j and self.W[i, j] > 0:
                    src.append(i); dst.append(j); weights.append(float(self.W[i, j]))
        for i in range(self.n_nodes):
            src.append(i); dst.append(i); weights.append(1.0)
        return [("all", torch.tensor([src, dst], dtype=torch.long),
                 torch.tensor(weights, dtype=torch.float32))]

    def border_adjacency(self) -> np.ndarray:
        """Row-normalised 0/1 border adjacency between scored areas, with self-loops."""
        a = (self.W[:self.n_neigh, :self.n_neigh] > 0).astype(np.float64) + np.eye(self.n_neigh)
        return a / a.sum(axis=1, keepdims=True)

    def summary(self) -> str:
        undirected = int((np.triu(self.W, 1) > 0).sum())
        parts = ", ".join(f"{k}={v}" for k, v in sorted(self.counts.items()))
        return (f"{self.n_nodes} nodes ({self.n_neigh} areas + {self.n_anchors} anchors), "
                f"{undirected} edges ({parts})")


def _hop_distances(adjacency: dict[int, set[int]], source: int) -> dict[int, int]:
    """BFS over area-area borders only, so anchors are not shortcuts."""
    distance = {source: 0}
    queue = collections.deque([source])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if neighbor not in distance:
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)
    return distance


def build_graph(spec: GraphSpec, *, city: City, flu_history: pd.DataFrame,
                static: np.ndarray | None = None) -> Graph:
    """Assemble W. `flu_history` must already be cut to weeks before the scored window."""
    use_anchors = spec.anchors and bool(city.anchors)
    names = [*city.areas, *(city.anchors if use_anchors else [])]
    n_nodes, n_neigh = len(names), city.n_areas
    index = {name: i for i, name in enumerate(names)}
    counts: dict[str, int] = {}
    W = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    anchor_pairs = list(city.anchor_edges) if use_anchors else []

    def put(i: int, j: int, weight: float) -> None:
        W[i, j] += weight
        W[j, i] += weight

    if spec.geo:
        adjacency: dict[int, set[int]] = collections.defaultdict(set)
        for a, b in city.edges:
            adjacency[index[a]].add(index[b])
            adjacency[index[b]].add(index[a])
        for a, b in [*city.edges, *anchor_pairs]:
            put(index[a], index[b], 1.0)
        counts["borders"] = len(city.edges)
        counts["anchor_borders"] = len(anchor_pairs)
        if spec.geo_max_hop >= 2:
            hops: collections.Counter = collections.Counter()
            for i in range(n_neigh):
                for node, distance in _hop_distances(adjacency, i).items():
                    if node < n_neigh and 2 <= distance <= spec.geo_max_hop:
                        put(i, node, 0.5 * spec.geo_decay ** (distance - 1))
                        if i < node:
                            hops[distance] += 1
            for hop, count in sorted(hops.items()):
                counts[f"{hop}hop"] = count

    if spec.corr:
        matrix = flu_history.corr(method="pearson", min_periods=12).to_numpy(dtype=np.float64, copy=True)
        np.fill_diagonal(matrix, 0.0)
        n_corr = 0
        for i in range(n_neigh):
            for j in range(i + 1, n_neigh):
                r = matrix[i, j]
                if np.isfinite(r) and r > spec.corr_threshold:
                    put(i, j, spec.corr_coef * (1.0 if spec.corr_binary else float(r)))
                    n_corr += 1
        counts["correlation"] = n_corr

    if spec.demo:
        if static is None:
            raise ValueError("Similarity edges need static area characteristics.")
        from scipy.spatial.distance import pdist, squareform

        distances = squareform(pdist(static.astype(np.float64)))
        bandwidth = np.median(distances[distances > 0])
        similarity = np.exp(-(distances ** 2) / (2 * bandwidth ** 2))
        np.fill_diagonal(similarity, 0.0)
        n_demo = 0
        for i in range(n_neigh):
            for j in range(i + 1, n_neigh):
                if similarity[i, j] > spec.demo_threshold:
                    put(i, j, spec.demo_coef * float(similarity[i, j]))
                    n_demo += 1
        counts["similarity"] = n_demo

    return Graph(node_names=names, n_nodes=n_nodes, n_neigh=n_neigh, W=W, counts=counts)
