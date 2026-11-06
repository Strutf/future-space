"""Um conector por dataset do catálogo.

Todo conector tem a assinatura ``(client, start, end) -> DataFrame`` e devolve colunas
com os nomes de ``Dataset.fields``. Fontes de tempo real ignoram start/end e devolvem
tudo o que o endpoint oferece; a deduplicação é feita pelo writer.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

import httpx
import pandas as pd

from futurespace.pipeline.connectors import gfz, omni, swpc

Connector = Callable[[httpx.Client, datetime, datetime], pd.DataFrame]

CONNECTORS: dict[str, Connector] = {
    "solar-wind-plasma": swpc.solar_wind_plasma,
    "solar-wind-mag": swpc.solar_wind_mag,
    "goes-xray": swpc.goes_xray,
    "goes-protons": swpc.goes_protons,
    "goes-flares": swpc.goes_flares,
    "solar-regions": swpc.solar_regions,
    "kp": swpc.kp,
    "kp-forecast": swpc.kp_forecast,
    "dst": swpc.dst,
    "kp-gfz": gfz.kp_gfz,
    "omni-hourly": omni.omni_hourly,
}
