"""Índice Kp oficial do GFZ Potsdam (https://kp.gfz.de), série desde 1932."""

from __future__ import annotations

from datetime import datetime

import httpx
import pandas as pd

from futurespace.partitions import format_time
from futurespace.pipeline.chunks import year_chunks
from futurespace.pipeline.http import get_json

URL = "https://kp.gfz.de/app/json/"


def kp_gfz(client: httpx.Client, start: datetime, end: datetime) -> pd.DataFrame:
    frames = []
    for chunk_start, chunk_end in year_chunks(start, end, years=10):
        data = get_json(
            client,
            URL,
            params={"start": format_time(chunk_start), "end": format_time(chunk_end), "index": "Kp"},
        )
        if data.get("datetime"):
            frames.append(
                pd.DataFrame({"time": data["datetime"], "kp": data["Kp"], "status": data["status"]})
            )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
