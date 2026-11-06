"""OMNI2 horário da NASA via servidor HAPI do CDAWeb.

HAPI (Heliophysics API) é um padrão usado por vários provedores de heliofísica; este
módulo pode virar a base de conectores para outros datasets do CDAWeb.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime

import httpx
import pandas as pd

from futurespace.partitions import format_time
from futurespace.pipeline.chunks import year_chunks
from futurespace.pipeline.http import get, get_json

log = logging.getLogger(__name__)

HAPI = "https://cdaweb.gsfc.nasa.gov/hapi"
DATASET = "OMNI2_H0_MRG1HR"

PARAMETERS = {
    "ABS_B1800": "b_abs",
    "BX_GSE1800": "bx_gse",
    "BY_GSM1800": "by_gsm",
    "BZ_GSM1800": "bz_gsm",
    "T1800": "temperature",
    "N1800": "density",
    "V1800": "speed",
    "Pressure1800": "pressure",
    "E1800": "e_field",
    "Beta1800": "beta",
    "R1800": "sunspot_number",
    "F10_INDEX1800": "f107",
    "KP1800": "kp",
    "DST1800": "dst",
    "AE1800": "ae",
    "AP_INDEX1800": "ap",
    "AL_INDEX1800": "al",
    "AU_INDEX1800": "au",
}

HAPI_NO_DATA = 1201


def _fetch_chunk(client: httpx.Client, names: list[str], start: datetime, end: datetime) -> str:
    resp = get(
        client,
        f"{HAPI}/data",
        params={
            "id": DATASET,
            "time.min": format_time(start),
            "time.max": format_time(end),
            "parameters": ",".join(names),
        },
    )
    text = resp.text
    if text.lstrip().startswith("{"):
        status = resp.json().get("status", {})
        if status.get("code") == HAPI_NO_DATA:
            return ""
        raise RuntimeError(f"erro HAPI: {status}")
    return text


def omni_hourly(client: httpx.Client, start: datetime, end: datetime) -> pd.DataFrame:
    info = get_json(client, f"{HAPI}/info", params={"id": DATASET})
    order = [p["name"] for p in info["parameters"]]
    fills = {p["name"]: p.get("fill") for p in info["parameters"]}
    # O HAPI exige os parâmetros na mesma ordem em que aparecem no /info.
    names = sorted(PARAMETERS, key=order.index)

    frames = []
    for chunk_start, chunk_end in year_chunks(start, end):
        text = _fetch_chunk(client, names, chunk_start, chunk_end)
        if not text.strip():
            continue
        frames.append(pd.read_csv(io.StringIO(text), header=None, names=["time", *names]))
        log.info("omni-hourly: %s -> %s", chunk_start.date(), chunk_end.date())
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    for name in names:
        if fills.get(name) is not None:
            df[name] = df[name].mask(df[name] == float(fills[name]))
    df["KP1800"] = df["KP1800"] / 10  # OMNI guarda Kp*10 (ex.: 33 = 3+)
    # O HAPI carimba o centro da hora (HH:30); padronizamos para o início da hora.
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.floor("h")
    return df.rename(columns=PARAMETERS)
