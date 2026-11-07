"""Conectores da previsão de aurora: mapa OVATION da NOAA e nuvens do Open-Meteo."""

from __future__ import annotations

from datetime import datetime

import httpx
import pandas as pd

from futurespace.aurora.sites import SITES
from futurespace.pipeline.http import get_json

OVATION_URL = "https://services.swpc.noaa.gov/json/ovation_aurora_latest.json"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_VARIABLES = ["cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "visibility"]
FORECAST_HOURS = 25


def aurora_ovation(client: httpx.Client, start: datetime, end: datetime) -> pd.DataFrame:
    """Mapa global de probabilidade de aurora (grade de 1°), só os pontos diferentes de zero.

    A NOAA não mantém arquivo público deste produto: o que não for coletado se perde.
    """
    raw = get_json(client, OVATION_URL)
    grid = pd.DataFrame(raw["coordinates"], columns=["lon", "lat", "aurora"])
    grid = grid[grid["aurora"] > 0]
    grid.insert(0, "forecast_time", pd.Timestamp(raw["Forecast Time"]))
    grid.insert(0, "time", pd.Timestamp(raw["Observation Time"]))
    return grid


def aurora_weather(client: httpx.Client, start: datetime, end: datetime) -> pd.DataFrame:
    """Nuvens e visibilidade previstas para as próximas 25 h em cada cidade (uma chamada só)."""
    params = {
        "latitude": ",".join(str(s.lat) for s in SITES),
        "longitude": ",".join(str(s.lon) for s in SITES),
        "hourly": ",".join(WEATHER_VARIABLES),
        "forecast_hours": FORECAST_HOURS,
        "timezone": "GMT",
        "timeformat": "unixtime",
    }
    raw = get_json(client, OPEN_METEO_URL, params)
    issued = pd.Timestamp(end).floor("h")
    frames = []
    # A resposta vem na mesma ordem das coordenadas pedidas.
    for site, loc in zip(SITES, raw if isinstance(raw, list) else [raw], strict=True):
        hourly = pd.DataFrame(loc["hourly"])
        hourly["time"] = pd.to_datetime(hourly["time"], unit="s", utc=True)
        hourly.insert(1, "site", site.id)
        hourly.insert(2, "issued", issued)
        frames.append(hourly)
    return pd.concat(frames, ignore_index=True)
