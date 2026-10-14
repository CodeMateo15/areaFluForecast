"""Build the area graph from boundary polygons.

Two areas are neighbours when their boundaries share a border. The default rule
needs no GIS library: in a topologically clean boundary file (Census TIGER, most
city open-data portals) two polygons that share a border share the *identical*
vertex coordinates, so two areas are adjacent exactly when they share at least
two boundary vertices. Two rather than one, because a single shared vertex is a
corner touch, not a border. Building neighbour lists from boundary files is
standard in spatial epidemiology (see Morris et al. 2019, Spat Spatiotemporal
Epidemiol 31:100301).

For files that are not topologically clean (areas digitised separately, slivers
between them), `method="touches"` uses shapely with a snapping tolerance. Check
the result: the edges can change with the tolerance.

In the research behind this package, a graph derived this way from boundary
files forecast as well as a hand-drawn map, and the hand-drawn maps were only
~85% right.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd

from .errors import DataError

PRECISION = 6            # decimal degrees, ~0.1 m
MIN_SHARED_VERTICES = 2


def read_geojson(path: str | Path) -> list[dict]:
    """The features of a GeoJSON FeatureCollection (lon/lat, EPSG:4326)."""
    payload = json.loads(Path(path).read_text())
    features = payload.get("features") if isinstance(payload, dict) else None
    if not features:
        raise DataError(f"{path} has no GeoJSON features.")
    return features


def _id_getter(id_of: str | Callable[[dict], object]) -> Callable[[dict], str]:
    if callable(id_of):
        return lambda feature: str(id_of(feature))

    def get(feature: dict) -> str:
        properties = feature.get("properties") or {}
        if id_of not in properties:
            raise DataError(f"Feature has no property {id_of!r}. Available: {sorted(properties)}")
        return str(properties[id_of])
    return get


def _rings(geometry: dict) -> list[list]:
    if geometry is None:
        return []
    kind = geometry["type"]
    if kind == "Polygon":
        return list(geometry["coordinates"])
    if kind == "MultiPolygon":
        return [ring for polygon in geometry["coordinates"] for ring in polygon]
    if kind == "GeometryCollection":
        return [ring for part in geometry["geometries"] for ring in _rings(part)]
    raise DataError(f"Unsupported geometry type {kind!r}; expected polygons.")


def _vertices(geometry: dict, precision: int) -> set[tuple[float, float]]:
    return {(round(x, precision), round(y, precision))
            for ring in _rings(geometry) for x, y, *_ in ring}


def adjacency_from_polygons(features: Iterable[dict], id_of: str | Callable[[dict], object], *,
                            method: str = "vertices", precision: int = PRECISION,
                            min_shared: int = MIN_SHARED_VERTICES,
                            tolerance: float = 1e-5) -> list[tuple[str, str]]:
    """Sorted list of (a, b) neighbour pairs, a < b.

    Several features may carry the same id (ZIP codes rolled up to an area, or
    tracts to a neighbourhood); they are merged into one node, and borders inside
    one node are ignored.

    method="vertices" (default) is the shared-vertex rule above. method="touches"
    needs shapely: two areas are neighbours when their polygons, each buffered by
    `tolerance` degrees, overlap along a line rather than at a point.
    """
    get_id = _id_getter(id_of)
    features = list(features)
    if method == "vertices":
        owners: dict[tuple[float, float], set[str]] = defaultdict(set)
        for feature in features:
            node = get_id(feature)
            for vertex in _vertices(feature["geometry"], precision):
                owners[vertex].add(node)
        shared: dict[tuple[str, str], int] = defaultdict(int)
        for nodes in owners.values():
            ordered = sorted(nodes)
            for i, a in enumerate(ordered):
                for b in ordered[i + 1:]:
                    shared[(a, b)] += 1
        return sorted(pair for pair, count in shared.items() if count >= min_shared)
    if method == "touches":
        return _touches_adjacency(features, get_id, tolerance)
    raise ValueError(f"Unknown method {method!r}; expected 'vertices' or 'touches'.")


def _touches_adjacency(features, get_id, tolerance) -> list[tuple[str, str]]:
    try:
        from shapely.geometry import shape
        from shapely.ops import unary_union
    except ImportError as exc:  # pragma: no cover
        raise ImportError("method='touches' needs shapely: pip install 'areaFluForecast[geo]'") from exc
    parts: dict[str, list] = defaultdict(list)
    for feature in features:
        parts[get_id(feature)].append(shape(feature["geometry"]))
    merged = {node: unary_union(geoms).buffer(tolerance) for node, geoms in parts.items()}
    nodes = sorted(merged)
    pairs = []
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            overlap = merged[a].intersection(merged[b])
            # A corner touch buffered by `tolerance` is a blob ~tolerance wide; a
            # shared border is a strip much longer than it is wide.
            if not overlap.is_empty and overlap.length > 8 * tolerance:
                pairs.append((a, b))
    return pairs


def centroids_from_polygons(features: Iterable[dict],
                            id_of: str | Callable[[dict], object]) -> dict[str, tuple[float, float]]:
    """(lat, lon) per area: the area-weighted centroid of its outer rings.

    Used to fetch weather at each area. Planar arithmetic on lon/lat is fine at
    city scale.
    """
    get_id = _id_getter(id_of)
    sums: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for feature in features:
        node = get_id(feature)
        geometry = feature["geometry"]
        polygons = ([geometry["coordinates"]] if geometry["type"] == "Polygon"
                    else geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [])
        for polygon in polygons:
            ring = polygon[0]
            area = cx = cy = 0.0
            for (x0, y0, *_), (x1, y1, *_) in zip(ring, ring[1:] + ring[:1]):
                cross = x0 * y1 - x1 * y0
                area += cross
                cx += (x0 + x1) * cross
                cy += (y0 + y1) * cross
            if abs(area) < 1e-15:
                continue
            area *= 0.5
            weight = abs(area)
            sums[node][0] += weight * cx / (6.0 * area)
            sums[node][1] += weight * cy / (6.0 * area)
            sums[node][2] += weight
    return {node: (lat / w, lon / w) for node, (lon, lat, w) in sums.items() if w > 0}


def read_edges_csv(path: str | Path) -> list[tuple[str, str]]:
    """Edges from a CSV whose first two columns are the two area ids."""
    frame = pd.read_csv(path, dtype=str)
    if frame.shape[1] < 2:
        raise DataError(f"{path} needs at least two columns (the two areas of each edge).")
    a, b = frame.columns[:2]
    return sorted({tuple(sorted((str(x).strip(), str(y).strip())))
                   for x, y in zip(frame[a], frame[b]) if str(x).strip() != str(y).strip()})


def isolated(nodes: Iterable[str], edges: Iterable[tuple[str, str]]) -> list[str]:
    """Nodes that appear in no edge."""
    touched = {n for pair in edges for n in pair}
    return [n for n in nodes if n not in touched]
