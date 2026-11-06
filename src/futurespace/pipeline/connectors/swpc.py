"""Conectores dos produtos JSON públicos da NOAA SWPC (services.swpc.noaa.gov)."""

from __future__ import annotations

import re
from datetime import datetime

import httpx
import pandas as pd

from futurespace.pipeline.http import get_json

BASE = "https://services.swpc.noaa.gov"


def _frame(client: httpx.Client, path: str) -> pd.DataFrame:
    return pd.DataFrame(get_json(client, f"{BASE}/{path}"))


def _select(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Seleciona e renomeia colunas {coluna_na_fonte: coluna_no_catálogo}."""
    present = {src: dst for src, dst in mapping.items() if src in df.columns}
    return df[list(present)].rename(columns=present)


def solar_wind_plasma(client: httpx.Client, start: datetime, end: datetime) -> pd.DataFrame:
    return _select(
        _frame(client, "json/rtsw/rtsw_wind_1m.json"),
        {
            "time_tag": "time",
            "source": "source",
            "active": "active",
            "proton_speed": "speed",
            "proton_density": "density",
            "proton_temperature": "temperature",
            "proton_vx_gsm": "vx_gsm",
            "proton_vy_gsm": "vy_gsm",
            "proton_vz_gsm": "vz_gsm",
        },
    )


def solar_wind_mag(client: httpx.Client, start: datetime, end: datetime) -> pd.DataFrame:
    return _select(
        _frame(client, "json/rtsw/rtsw_mag_1m.json"),
        {
            "time_tag": "time",
            "source": "source",
            "active": "active",
            "bt": "bt",
            "bx_gsm": "bx_gsm",
            "by_gsm": "by_gsm",
            "bz_gsm": "bz_gsm",
            "theta_gsm": "theta_gsm",
            "phi_gsm": "phi_gsm",
            "overall_quality": "quality",
        },
    )


def _pivot_by_energy(df: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    """Transforma linhas (tempo, satélite, energia, fluxo) em uma coluna por energia."""
    df = df[df["energy"].isin(columns)].assign(band=lambda d: d["energy"].map(columns))
    df = df.drop_duplicates(["time_tag", "satellite", "band"], keep="last")
    wide = df.pivot(index=["time_tag", "satellite"], columns="band", values="flux").reset_index()
    wide.columns.name = None
    return wide.rename(columns={"time_tag": "time"})


def goes_xray(client: httpx.Client, start: datetime, end: datetime) -> pd.DataFrame:
    df = _frame(client, "json/goes/primary/xrays-7-day.json")
    return _pivot_by_energy(df, {"0.05-0.4nm": "flux_short", "0.1-0.8nm": "flux_long"})


def _proton_column(energy: str) -> str | None:
    m = re.fullmatch(r">=(\d+) MeV", energy)
    return f"flux_ge_{m.group(1)}mev" if m else None


def goes_protons(client: httpx.Client, start: datetime, end: datetime) -> pd.DataFrame:
    df = _frame(client, "json/goes/primary/integral-protons-7-day.json")
    columns = {e: c for e in df["energy"].unique() if (c := _proton_column(e))}
    return _pivot_by_energy(df, columns)


def goes_flares(client: httpx.Client, start: datetime, end: datetime) -> pd.DataFrame:
    return _select(
        _frame(client, "json/goes/primary/xray-flares-7-day.json"),
        {
            "begin_time": "time",
            "satellite": "satellite",
            "max_time": "max_time",
            "end_time": "end_time",
            "begin_class": "begin_class",
            "max_class": "max_class",
            "end_class": "end_class",
            "max_xrlong": "max_flux",
            "current_int_xrlong": "integrated_flux",
        },
    )


def solar_regions(client: httpx.Client, start: datetime, end: datetime) -> pd.DataFrame:
    return _select(
        _frame(client, "json/solar_regions.json"),
        {
            "observed_date": "time",
            "region": "region",
            "location": "location",
            "latitude": "latitude",
            "longitude": "longitude",
            "area": "area",
            "spot_class": "spot_class",
            "mag_class": "mag_class",
            "number_spots": "number_spots",
            "c_flare_probability": "c_flare_probability",
            "m_flare_probability": "m_flare_probability",
            "x_flare_probability": "x_flare_probability",
            "c_xray_events": "c_xray_events",
            "m_xray_events": "m_xray_events",
            "x_xray_events": "x_xray_events",
        },
    )


def kp(client: httpx.Client, start: datetime, end: datetime) -> pd.DataFrame:
    return _select(
        _frame(client, "products/noaa-planetary-k-index.json"),
        {"time_tag": "time", "Kp": "kp", "a_running": "a_running", "station_count": "station_count"},
    )


def kp_forecast(client: httpx.Client, start: datetime, end: datetime) -> pd.DataFrame:
    """Guarda as previsões de blocos que ainda não começaram, carimbadas com a hora da coleta.

    A NOAA marca os blocos restantes do dia corrente como "estimated" e só os dias seguintes
    como "predicted". Para um bloco que ainda não começou, os dois são previsão de verdade.
    O bloco em andamento fica de fora: ali o "estimated" já é medição parcial.
    """
    df = _frame(client, "products/noaa-planetary-k-index-forecast.json")
    fetched = pd.Timestamp(end)
    block_start = pd.to_datetime(df["time_tag"], utc=True)
    future = df["observed"].isin(["predicted", "estimated"]) & (block_start > fetched)
    out = _select(df[future], {"time_tag": "target_time", "kp": "kp", "noaa_scale": "noaa_scale"})
    out.insert(0, "time", fetched.floor("h"))
    return out


def dst(client: httpx.Client, start: datetime, end: datetime) -> pd.DataFrame:
    return _select(_frame(client, "products/kyoto-dst.json"), {"time_tag": "time", "dst": "dst"})
