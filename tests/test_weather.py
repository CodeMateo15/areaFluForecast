import pandas as pd
import pytest

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


@pytest.mark.network
def test_open_meteo_archive_live():
    """The archive still answers, with the variables the package asks for."""
    daily = weather.fetch_daily(42.36, -71.06, "2024-01-01", "2024-01-14", delay=0)
    weeks = weather.to_weeks(daily, pd.date_range("2024-01-07", periods=1, freq="7D"))
    assert list(weeks.columns) == weather.WEATHER_COLS
    assert weeks.notna().all().all()
