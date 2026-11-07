import json
from datetime import UTC, datetime

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from futurespace import catalog
from futurespace.api.app import create_app
from futurespace.aurora import astro
from futurespace.aurora.sites import BY_ID, SITES
from futurespace.aurora.visibility import compute, kp_needed, mag_lat, ovation_nearby
from futurespace.partitions import json_key, latest_key, parquet_key
from futurespace.pipeline.connectors import aurora
from futurespace.pipeline.http import make_client
from futurespace.pipeline.writer import write_dataset

NOW = datetime(2026, 11, 7, 21, 10, tzinfo=UTC)


def test_sun_and_moon_match_astropy():
    # Valores de referência calculados com o astropy 8.0 (sem refração).
    times = pd.to_datetime(["2024-05-11T00:00Z", "2026-11-07T21:00Z"])
    lat, lon = np.array([69.65, -23.55]), np.array([18.96, -46.63])
    assert astro.sun_altitude(times, lat, lon) == pytest.approx([-1.270, 4.343], abs=0.02)
    assert astro.moon_altitude(times, lat, lon) == pytest.approx([8.131, -12.265], abs=0.4)


def test_moon_phases():
    full, new = astro.moon_illumination(pd.to_datetime(["2024-04-23T23:49Z", "2024-04-08T18:21Z"]))
    assert full > 0.99 and new < 0.01


def test_polar_night_in_tromso():
    noon = pd.date_range("2024-12-21T00:00Z", periods=24, freq="h")
    assert astro.sun_altitude(noon, 69.65, 18.96).max() < 0


def test_kp_needed_follows_the_site_aurora_rule():
    assert kp_needed(70) == 0
    assert kp_needed(54) == pytest.approx(8)
    assert kp_needed(50) == pytest.approx(8 + 4 / 9.5)
    assert kp_needed(40) == float("inf")
    assert mag_lat(80.8, -72.7) == pytest.approx(90)


def test_ovation_nearby_uses_a_500_km_radius():
    tromso = BY_ID["tromso"]
    grid = pd.DataFrame({"lon": [19, 19, 100], "lat": [72, 50, 70], "aurora": [30, 90, 80]})
    # (72°N, 19°E) fica a ~260 km; (50°N) e (70°N, 100°E) ficam longe.
    assert ovation_nearby(grid, tromso.lat, tromso.lon) == 30


def _inputs():
    ovation = pd.DataFrame(
        {
            "time": pd.Timestamp("2026-11-07T21:03Z"),
            "forecast_time": pd.Timestamp("2026-11-07T21:42Z"),
            "lon": [19],
            "lat": [70],
            "aurora": [40],
        }
    )
    weather = pd.DataFrame(
        [
            {"time": pd.Timestamp("2026-11-07T21:00Z") + pd.Timedelta(hours=h), "site": s.id, "cloud_cover": 25.0}
            for s in SITES
            for h in range(25)
        ]
    )
    predictions = pd.DataFrame(
        {
            "time": pd.Timestamp("2026-11-07T21:00Z"),
            "target": "kp",
            "horizon_hours": [3, 6, 12, 24],
            "target_time": pd.to_datetime(
                ["2026-11-08T00:00Z", "2026-11-08T03:00Z", "2026-11-08T09:00Z", "2026-11-08T21:00Z"]
            ),
            "value": [6.0, 5.0, 3.0, 2.0],
        }
    )
    kp = pd.DataFrame({"time": [pd.Timestamp("2026-11-07T18:00Z")], "kp": [4.33]})
    return ovation, weather, predictions, kp


def test_visibility_combines_aurora_clouds_darkness_and_moon():
    rows = compute(*_inputs(), NOW)
    assert len(rows) == len(SITES) * 25
    tromso = rows[rows["site"] == "tromso"].set_index("lead_hours")

    # Antes do primeiro bloco previsto vale o último Kp medido; depois, a previsão do bloco.
    assert tromso.loc[0, ["kp_expected", "kp_source"]].tolist() == [4.33, "observed"]
    assert tromso.loc[3, ["kp_expected", "kp_source"]].tolist() == [6.0, "model"]
    assert tromso.loc[7, "kp_expected"] == 5.0  # 04h fica no bloco das 03h
    assert tromso.loc[10, "kp_expected"] == 4.0  # 07h: bloco das 06h, entre 5 (03h) e 3 (09h)
    assert tromso.loc[24, "kp_expected"] == 2.0

    # OVATION só nas horas 0 e 1, numa coluna própria.
    assert tromso.loc[0, "aurora_score_ovation"] == 0.4
    assert np.isnan(tromso.loc[2, "aurora_score_ovation"])
    assert (tromso["aurora_score"] == tromso["aurora_score_kp"]).all()

    sky = (1 - 0.25) * tromso["darkness"] * np.where(tromso["moon_altitude"] > 0, 1 - 0.3 * tromso["moon_illumination"], 1)
    assert tromso["visibility_score"].to_numpy() == pytest.approx((tromso["aurora_score"] * sky).to_numpy(), abs=1e-3)

    # São Paulo: a aurora nunca chega ao horizonte, mesmo com Kp 9.
    assert (rows.loc[rows["site"] == "sao_paulo", "visibility_score"] == 0).all()


def test_visibility_without_predictions_or_ovation():
    _, weather, _, kp = _inputs()
    rows = compute(pd.DataFrame(), weather, pd.DataFrame(), kp, NOW)
    assert (rows["kp_source"] == "observed").all()
    assert rows["aurora_score_ovation"].isna().all()
    assert rows["visibility_score"].notna().all()


def test_connectors_parse_ovation_and_open_meteo():
    ovation = {
        "Observation Time": "2026-11-07T21:03:00Z",
        "Forecast Time": "2026-11-07T21:42:00Z",
        "coordinates": [[0, 65, 0], [19, 70, 40]],
    }
    hourly = {"time": [1794517200, 1794520800], "cloud_cover": [100, 50], "cloud_cover_low": [90, 40],
              "cloud_cover_mid": [10, 5], "cloud_cover_high": [0, 0], "visibility": [2000.0, 24000.0]}
    meteo = [{"hourly": hourly} for _ in SITES]

    def handler(request: httpx.Request) -> httpx.Response:
        body = ovation if request.url.host == "services.swpc.noaa.gov" else meteo
        return httpx.Response(200, content=json.dumps(body))

    with make_client(transport=httpx.MockTransport(handler)) as c:
        grid = aurora.aurora_ovation(c, NOW, NOW)
        weather = aurora.aurora_weather(c, NOW, NOW)
    assert grid[["lon", "lat", "aurora"]].to_dict("records") == [{"lon": 19, "lat": 70, "aurora": 40}]
    assert len(weather) == 2 * len(SITES)
    assert weather.iloc[0][["site", "cloud_cover"]].tolist() == [SITES[0].id, 100]
    assert (weather["issued"] == pd.Timestamp("2026-11-07T21:00Z")).all()


def test_parquet_only_datasets_skip_json_and_the_api_points_to_parquet(store):
    ds = catalog.get("aurora-ovation")
    grid = pd.DataFrame({"time": [NOW], "forecast_time": [NOW], "lon": [19], "lat": [70], "aurora": [40]})
    write_dataset(store, ds, grid, now=NOW)
    assert store.read(parquet_key(ds.id, "2026-11-07")) is not None
    assert store.read(json_key(ds.id, "2026-11-07")) is None
    assert store.read(latest_key(ds.id)) is None

    rows = compute(*_inputs(), NOW)
    write_dataset(store, catalog.get("aurora-visibility"), rows, now=NOW)
    assert store.read(json_key("aurora-visibility", "2026-11-07")) is None
    assert store.read(latest_key("aurora-visibility")) is not None

    client = TestClient(create_app(store))
    for path in ("/v1/datasets/aurora-ovation/data", "/v1/datasets/aurora-ovation/latest"):
        resp = client.get(path)
        assert resp.status_code == 404 and "Parquet" in resp.json()["detail"]
    latest = client.get("/v1/datasets/aurora-visibility/latest").json()["data"]
    assert len(latest) == len(SITES) * 25
