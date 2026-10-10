import json

import pytest

from areafluforecast.adjacency import (adjacency_from_polygons, centroids_from_polygons,
                                       read_edges_csv, read_geojson)


def square(name, x, y, size=1.0):
    ring = [[x, y], [x + size, y], [x + size, y + size], [x, y + size], [x, y]]
    return {"type": "Feature", "properties": {"name": name},
            "geometry": {"type": "Polygon", "coordinates": [ring]}}


def grid(n=3):
    """n x n unit squares named r{row}c{col}."""
    return [square(f"r{r}c{c}", c, r) for r in range(n) for c in range(n)]


def test_rook_adjacency_on_a_grid():
    edges = adjacency_from_polygons(grid(), "name")
    expected = set()
    for r in range(3):
        for c in range(3):
            if c < 2:
                expected.add(tuple(sorted((f"r{r}c{c}", f"r{r}c{c + 1}"))))
            if r < 2:
                expected.add(tuple(sorted((f"r{r}c{c}", f"r{r + 1}c{c}"))))
    # Diagonal neighbours share one vertex (a corner touch) and are not borders.
    assert set(edges) == expected
    assert len(edges) == 12


def test_features_with_the_same_id_merge_and_internal_borders_vanish():
    features = grid()
    for f in features:
        if f["properties"]["name"] in ("r0c0", "r0c1"):
            f["properties"]["name"] = "west"
    edges = adjacency_from_polygons(features, "name")
    assert ("r0c1", "west") not in edges
    assert ("r0c2", "west") in edges and ("r1c0", "west") in edges


def test_multipolygon_and_callable_id():
    a = square("a", 0, 0)
    b = square("b", 1, 0)
    island = square("a", 5, 5)["geometry"]["coordinates"]
    a["geometry"] = {"type": "MultiPolygon", "coordinates": [a["geometry"]["coordinates"], island]}
    edges = adjacency_from_polygons([a, b], lambda f: f["properties"]["name"].upper())
    assert edges == [("A", "B")]


def test_centroids(tmp_path):
    cents = centroids_from_polygons([square("a", 0, 0, 2.0)], "name")
    assert cents["a"] == pytest.approx((1.0, 1.0))
    path = tmp_path / "g.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": grid(2)}))
    assert len(read_geojson(path)) == 4


def test_touches_method_tolerates_a_sliver():
    pytest.importorskip("shapely")
    a = square("a", 0, 0)
    b = square("b", 1.000002, 0)          # 0.2 m gap: vertices do not coincide
    assert adjacency_from_polygons([a, b], "name") == []
    assert adjacency_from_polygons([a, b], "name", method="touches", tolerance=1e-5) == [("a", "b")]


def test_edges_csv(tmp_path):
    path = tmp_path / "e.csv"
    path.write_text("from,to\nb,a\na,b\nc,c\na,c\n")
    assert read_edges_csv(path) == [("a", "b"), ("a", "c")]
