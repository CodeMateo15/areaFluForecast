import pandas as pd

from areafluforecast import weather


def test_nearby_centroids_share_one_request(monkeypatch):
    calls = []

    def fake_daily(lat, lon, start, end, cache_dir=None, delay=1.0):
        calls.append((lat, lon))
        days = pd.date_range(start, end, freq="D")
        return pd.DataFrame({name: float(lat) for name in weather.DAILY_VARIABLES}, index=days)

    monkeypatch.setattr(weather, "fetch_daily", fake_daily)
    weeks = pd.date_range("2024-01-07", periods=10, freq="7D")
    centroids = {"a": (41.901, -87.70), "b": (41.904, -87.71), "c": (41.70, -87.60)}
    out = weather.fetch_weather(centroids, weeks, verbose=False)
    assert len(calls) == 2                                  # a and b share a 0.1-degree cell
    assert list(out["temp_mean_c"].columns) == ["a", "b", "c"]
    assert out["temp_mean_c"]["a"].equals(out["temp_mean_c"]["b"])
    weather.fetch_weather(centroids, weeks, verbose=False, grid=0)
    assert len(calls) == 5                                  # grid=0: one request per centroid
