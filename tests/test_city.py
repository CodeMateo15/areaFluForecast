
import pandas as pd
import pytest

from areafluforecast import City, DataError, DataWarning, rates_from_counts, read_rates_csv


def test_basic_city(rates, line_edges):
    city = City(name="t", rates=rates, edges=line_edges)
    assert city.areas == ["a", "b", "c", "d"]
    assert city.season_first_month == 10
    assert city.season_window(2022) == (pd.Timestamp("2022-06-01"), pd.Timestamp("2023-05-31"))
    assert city.complete_seasons()                 # 5 years of data
    assert "4 areas" in city.summary()


def test_southern_hemisphere_season(rates, line_edges):
    city = City(name="t", rates=rates, edges=line_edges, flu_months=(4, 5, 6, 7, 8, 9))
    assert city.season_first_month == 4
    assert city.season_window(2022)[0] == pd.Timestamp("2022-12-01")


def test_missing_whole_weeks_become_missing_rows(rates, line_edges):
    city = City(name="t", rates=rates.drop(rates.index[[10, 11]]), edges=line_edges)
    assert len(city.rates) == len(rates)
    assert city.rates.iloc[10].isna().all()
    assert any("absent" in n for n in city.notes)


@pytest.mark.parametrize("change, message", [
    (lambda r: r.assign(a=-r["a"]), "negative"),
    (lambda r: r.iloc[:50], "two years"),
    (lambda r: r[["a"]], "two areas"),
])
def test_bad_rates_are_explained(rates, line_edges, change, message):
    with pytest.raises(DataError, match=message):
        City(name="t", rates=change(rates), edges=[e for e in line_edges if "a" in e] if
             message == "two areas" else line_edges)


def test_mixed_weekdays_rejected(rates, line_edges):
    shifted = rates.copy()
    shifted.index = shifted.index[:-1].append(pd.DatetimeIndex([shifted.index[-1] + pd.Timedelta(days=1)]))
    with pytest.raises(DataError, match="weekdays"):
        City(name="t", rates=shifted, edges=line_edges)


def test_unknown_edge_names_and_anchors(rates, line_edges):
    with pytest.raises(DataError, match="not columns"):
        City(name="t", rates=rates, edges=line_edges + [("a", "zz")])
    city = City(name="t", rates=rates, edges=line_edges, anchors=["zz"], anchor_edges=[("a", "zz")])
    assert city.anchors == ["zz"]
    with pytest.raises(DataError, match="border no area"):
        City(name="t", rates=rates, edges=line_edges, anchors=["zz"])


def test_isolated_area_warns(rates):
    with pytest.warns(DataWarning, match="border no other area"):
        City(name="t", rates=rates, edges=[("a", "b"), ("b", "c")])


def test_long_and_wide_csv(tmp_path, rates, line_edges):
    wide = tmp_path / "wide.csv"
    rates.rename_axis("week").to_csv(wide)
    long = tmp_path / "long.csv"
    rates.rename_axis("week").reset_index().melt("week", var_name="area", value_name="rate") \
        .to_csv(long, index=False)
    edges = tmp_path / "edges.csv"
    pd.DataFrame(line_edges, columns=["a", "b"]).to_csv(edges, index=False)
    from_wide = City.from_files(wide, edges=edges)
    from_long = City.from_files(long, edges=edges)
    pd.testing.assert_frame_equal(from_wide.rates, from_long.rates, check_names=False)
    duplicated = tmp_path / "dup.csv"
    frame = pd.read_csv(long)
    pd.concat([frame, frame.iloc[:1]]).to_csv(duplicated, index=False)
    with pytest.raises(DataError, match="duplicate"):
        read_rates_csv(duplicated)


def test_from_polygons_with_anchor(rates):
    def square(name, x):
        ring = [[x, 0], [x + 1, 0], [x + 1, 1], [x, 1], [x, 0]]
        return {"properties": {"id": name}, "geometry": {"type": "Polygon", "coordinates": [ring]}}
    features = [square(n, i) for i, n in enumerate(["town", "a", "b", "c", "d"])]
    city = City.from_frames("t", rates, polygons=features, id_col="id", anchors=["town"])
    assert city.edges == [("a", "b"), ("b", "c"), ("c", "d")]
    assert city.anchor_edges == [("a", "town")]
    assert set(city.centroids) == {"town", "a", "b", "c", "d"}


def test_rates_from_counts():
    counts = pd.DataFrame({"a": [10, 20], "b": [5, 0]})
    out = rates_from_counts(counts, {"a": 50_000, "b": 10_000})
    assert out["a"].tolist() == [20.0, 40.0] and out["b"].tolist() == [50.0, 0.0]
    with pytest.raises(DataError):
        rates_from_counts(counts, {"a": 1})


def test_chicago_sample_loads():
    from areafluforecast import load_sample
    city = load_sample("chicago")
    assert city.n_areas == 58 and city.anchors == ["60827"] and city.has_weather
    assert city.rates.max().max() <= 10 and 2025 in city.complete_seasons()
