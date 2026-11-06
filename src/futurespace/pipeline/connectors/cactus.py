"""Catálogo de CMEs do CACTus (SIDC, Observatório Real da Bélgica).

O CACTus detecta CMEs automaticamente nas imagens do coronógrafo LASCO/SOHO. Usamos
três produtos do mesmo algoritmo, o que mantém treino e operação consistentes:

- catálogo científico combinado (1997-05 a 2015-10)
- quicklook mensal (2015-11 em diante)
- tempo real (última semana)
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

import httpx
import pandas as pd

from futurespace.pipeline.http import get_text_optional

BASE = "https://www.sidc.be/cactus"
SCIENCE_URL = f"{BASE}/catalog/LASCO/2_5_0/cme_lz.txt"
MONTHLY_URL = BASE + "/catalog/LASCO/2_5_0/qkl/{year:04d}/{month:02d}/cmecat.txt"
REALTIME_URL = f"{BASE}/out/cmecat.txt"
QUICKLOOK_START = datetime(2015, 11, 1, tzinfo=UTC)
REALTIME_WINDOW = timedelta(days=10)

_ROW = re.compile(
    r"^\s*\d+\|(?P<t0>\d{4}/\d\d/\d\d \d\d:\d\d)\|\s*(?P<dt0>\d+)\s*\|\s*(?P<pa>\d+)\|\s*(?P<da>\d+)\|"
    r"\s*(?P<v>\d+)\|\s*(?P<dv>\d+)\|\s*(?P<minv>\d+)\|\s*(?P<maxv>\d+)\|\s*(?P<halo>I*V?)\s*$"
)
_HALO = {"": 0, "II": 2, "III": 3, "IV": 4}


def parse_catalog(text: str, catalog: str) -> pd.DataFrame:
    """Extrai as CMEs de um arquivo cmecat do CACTus, ignorando a seção de "flows"."""
    rows = []
    for line in text.splitlines():
        if line.startswith("# Flow"):
            break  # flows são detecções suspeitas; não são CMEs
        m = _ROW.match(line)
        if m:
            rows.append(
                {
                    "time": m["t0"].replace("/", "-"),
                    "duration": int(m["dt0"]),
                    "pa": int(m["pa"]),
                    "width": int(m["da"]),
                    "speed": int(m["v"]),
                    "speed_sigma": int(m["dv"]),
                    "speed_min": int(m["minv"]),
                    "speed_max": int(m["maxv"]),
                    "halo": _HALO.get(m["halo"], 0),
                    "catalog": catalog,
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def prefer_curated(df: pd.DataFrame) -> pd.DataFrame:
    """Descarta detecções em tempo real de períodos que o quicklook/científico já cobre.

    O mesmo evento aparece nos dois arquivos com valores um pouco diferentes; manter os
    dois contaria a CME duas vezes.
    """
    curated = df[df["catalog"] != "realtime"]
    if curated.empty:
        return df
    return df[(df["catalog"] != "realtime") | (df["time"] > curated["time"].max())]


def _months(start: datetime, end: datetime):
    cur = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while cur <= end:
        yield cur
        cur = (cur + timedelta(days=32)).replace(day=1)


def cme_cactus(client: httpx.Client, start: datetime, end: datetime) -> pd.DataFrame:
    frames = []
    if start < QUICKLOOK_START:
        text = get_text_optional(client, SCIENCE_URL)
        if text:
            frames.append(parse_catalog(text, "science"))
    if end >= QUICKLOOK_START:
        for month in _months(max(start, QUICKLOOK_START), end):
            text = get_text_optional(client, MONTHLY_URL.format(year=month.year, month=month.month))
            if text:
                frames.append(parse_catalog(text, "quicklook"))
    if end >= datetime.now(UTC) - REALTIME_WINDOW:
        text = get_text_optional(client, REALTIME_URL)
        if text:
            frames.append(parse_catalog(text, "realtime"))

    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # O científico só vale até o início do quicklook, para não contar a mesma CME duas vezes.
    df = df[(df["catalog"] != "science") | (df["time"] < QUICKLOOK_START)]
    df = prefer_curated(df)
    return df[(df["time"] >= start) & (df["time"] <= end)]
