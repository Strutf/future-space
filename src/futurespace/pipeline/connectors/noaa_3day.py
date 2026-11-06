"""Arquivo histórico do "3-Day Forecast" da NOAA SWPC, guardado pela NCEI (desde 2022).

A NOAA publica a previsão duas vezes por dia (00:30 e 12:30 UTC), com o Kp esperado em
cada bloco de 3 h dos três dias seguintes. Guardamos só os blocos que ainda não tinham
começado na hora da publicação: os demais já eram observação, não previsão.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import httpx
import pandas as pd

from futurespace.pipeline.http import get_text_optional

ARCHIVE = "https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/daily_reports/3day_forecast"
MONTHS = {m: i for i, m in enumerate(
    ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"), start=1)}

_ISSUED = re.compile(r"Issued:\s*(\d{4}) (\w{3}) (\d{1,2}) (\d{2})(\d{2}) UTC")
_COLUMNS = re.compile(r"^\s+(\w{3}) (\d{1,2})\s+(\w{3}) (\d{1,2})\s+(\w{3}) (\d{1,2})\s*$")
_ROW = re.compile(r"^(\d\d)-\d\dUT\s+(.*)$")
_VALUE = re.compile(r"(\d+(?:\.\d+)?)(?:\s*\((G\d)\))?")


def parse_forecast(text: str) -> pd.DataFrame:
    """Extrai a tabela "NOAA Kp index breakdown" de uma edição do 3-Day Forecast."""
    issued_m = _ISSUED.search(text)
    if not issued_m or "Kp index breakdown" not in text:
        return pd.DataFrame()
    y, mon, d, hh, mm = issued_m.groups()
    issued = datetime(int(y), MONTHS[mon], int(d), int(hh), int(mm), tzinfo=UTC)

    lines = text[text.index("Kp index breakdown"):].splitlines()
    dates, rows = None, []
    for line in lines[1:]:
        if dates is None:
            m = _COLUMNS.match(line)
            if m:
                parts = m.groups()
                dates = []
                for mon_s, day_s in zip(parts[0::2], parts[1::2], strict=True):
                    month = MONTHS[mon_s]
                    # Virada de ano: dezembro publicado, janeiro previsto.
                    year = issued.year + (1 if month < issued.month else 0)
                    dates.append(datetime(year, month, int(day_s), tzinfo=UTC))
            continue
        m = _ROW.match(line)
        if m:
            values = _VALUE.findall(m.group(2))
            for col, (kp, scale) in enumerate(values[:3]):
                rows.append({
                    "target_time": dates[col] + timedelta(hours=int(m.group(1))),
                    "kp": float(kp),
                    "noaa_scale": scale or None,
                })
        elif rows and not line.strip():
            break  # fim da tabela
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.insert(0, "time", issued)
    return df[df["target_time"] > issued].reset_index(drop=True)


def _month_files(client: httpx.Client, month: datetime) -> list[str]:
    listing = get_text_optional(client, f"{ARCHIVE}/{month:%Y}/{month:%m}/") or ""
    return [f"{ARCHIVE}/{month:%Y}/{month:%m}/{name}"
            for name in sorted(set(re.findall(r"\d{12}three_day_forecast\.txt", listing)))]


def kp_forecast_3day(client: httpx.Client, start: datetime, end: datetime) -> pd.DataFrame:
    months, cur = [], start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while cur <= end:
        months.append(cur)
        cur = (cur + timedelta(days=32)).replace(day=1)
    urls = [u for month in months for u in _month_files(client, month)]
    # Arquivo nomeado AAAAMMDDHHMM: filtra pelo período antes de baixar.
    lo, hi = f"{start:%Y%m%d%H%M}", f"{end:%Y%m%d%H%M}"
    urls = [u for u in urls if lo <= u.rsplit("/", 1)[1][:12] <= hi]
    with ThreadPoolExecutor(max_workers=8) as pool:
        texts = list(pool.map(lambda u: get_text_optional(client, u), urls))
    frames = [parse_forecast(t) for t in texts if t]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
