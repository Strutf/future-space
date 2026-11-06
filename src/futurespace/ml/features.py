"""Montagem da tabela horária e das features dos modelos de previsão.

Regra principal: a linha do instante ``t`` só usa dados até o fim da hora ``t``, e só
usa variáveis que também existem em tempo real (vento solar no L1, Dst, Kp, horário).
Assim o modelo treinado no OMNI histórico pode rodar com os dados coletados ao vivo.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd

from futurespace.partitions import manifest_key, parquet_key
from futurespace.storage import Storage

FEATURE_VERSION = 1

# Colunas da tabela horária que alimenta as features.
SOLAR_WIND = ["b_abs", "by_gsm", "bz_gsm", "speed", "density", "temperature", "pressure"]
BASE_COLUMNS = [*SOLAR_WIND, "dst", "kp"]

# Kp >= 5- (4.67) corresponde a tempestade G1 na escala da NOAA.
KP_STORM = 4.67
DST_STORM = -50.0


def load_dataset(storage: Storage, dataset_id: str) -> pd.DataFrame:
    """Lê todas as partições Parquet de um dataset."""
    raw = storage.read(manifest_key(dataset_id))
    if not raw:
        raise RuntimeError(f"sem dados de '{dataset_id}'; rode o backfill antes")
    frames = [
        pd.read_parquet(io.BytesIO(storage.read(parquet_key(dataset_id, lbl))))
        for lbl in json.loads(raw)["partitions"]
    ]
    return pd.concat(frames, ignore_index=True)


def hourly_from_history(omni: pd.DataFrame, kp_gfz: pd.DataFrame) -> pd.DataFrame:
    """Tabela horária (índice = início da hora, UTC) a partir do OMNI + Kp oficial do GFZ."""
    hourly = omni.set_index("time")[[*SOLAR_WIND, "dst"]].sort_index()
    hourly = hourly[~hourly.index.duplicated(keep="last")]
    hourly = hourly.reindex(pd.date_range(hourly.index.min(), hourly.index.max(), freq="h"))
    kp = kp_gfz.set_index("time")["kp"].sort_index()
    hourly["kp"] = kp.reindex(hourly.index, method="ffill", limit=2)
    return hourly


def _kp_known(kp_hourly: pd.Series) -> pd.Series:
    """Último Kp de um bloco de 3 h já encerrado ao fim da hora t.

    O bloco que começa em T só termina em T+3h; ao fim da hora t (t+1h) o último bloco
    encerrado começa em floor_3h(t+1h) - 3h.
    """
    idx = kp_hourly.index
    block_start = (idx + pd.Timedelta(hours=1)).floor("3h") - pd.Timedelta(hours=3)
    return pd.Series(kp_hourly.reindex(block_start).to_numpy(), index=idx)


def build_features(hourly: pd.DataFrame) -> pd.DataFrame:
    h = hourly
    f = pd.DataFrame(index=h.index)
    for col in SOLAR_WIND:
        f[col] = h[col]

    # Acoplamento vento solar-magnetosfera.
    bt_perp = np.hypot(h["by_gsm"], h["bz_gsm"])
    clock = np.arctan2(h["by_gsm"], h["bz_gsm"])
    f["newell"] = h["speed"] ** (4 / 3) * bt_perp ** (2 / 3) * np.abs(np.sin(clock / 2)) ** (8 / 3)
    f["vbs"] = h["speed"] * (-h["bz_gsm"]).clip(lower=0) / 1000  # mV/m, só Bz para o sul
    f["clock_sin"] = np.sin(clock)
    f["clock_cos"] = np.cos(clock)

    for col in ("bz_gsm", "speed", "density", "newell", "vbs"):
        src = f[col]
        for lag in (1, 2, 3, 6):
            f[f"{col}_lag{lag}"] = src.shift(lag)
        for win in (3, 6, 12, 24):
            f[f"{col}_mean{win}"] = src.rolling(win, min_periods=max(1, win // 2)).mean()
    for win in (3, 6, 12):
        f[f"bz_gsm_min{win}"] = f["bz_gsm"].rolling(win, min_periods=1).min()
    f["pressure_mean6"] = f["pressure"].rolling(6, min_periods=3).mean()

    f["dst"] = h["dst"]
    for lag in (1, 2, 3, 6):
        f[f"dst_lag{lag}"] = h["dst"].shift(lag)
    f["dst_diff1"] = h["dst"].diff(1)
    f["dst_diff3"] = h["dst"].diff(3)
    f["dst_min24"] = h["dst"].rolling(24, min_periods=12).min()

    kp_known = _kp_known(h["kp"])
    f["kp_known"] = kp_known
    f["kp_known_prev"] = kp_known.shift(3)
    f["kp_known_max24"] = kp_known.rolling(24, min_periods=12).max()

    # Sazonalidade (efeito Russell-McPherron) e hora do dia.
    doy = f.index.day_of_year.to_numpy()
    hour = f.index.hour.to_numpy()
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    f["ut_sin"] = np.sin(2 * np.pi * hour / 24)
    f["ut_cos"] = np.cos(2 * np.pi * hour / 24)
    return f


def build_targets(hourly: pd.DataFrame, horizons: dict[str, list[int]]) -> pd.DataFrame:
    """Alvo `<índice>_h<h>` = valor observado na hora t+h."""
    t = pd.DataFrame(index=hourly.index)
    for target, hs in horizons.items():
        for h in hs:
            t[f"{target}_h{h}"] = hourly[target].shift(-h)
    return t


def persistence(features: pd.DataFrame, target: str) -> pd.Series:
    """Baseline "vai continuar igual": último valor conhecido do índice."""
    return features["dst"] if target == "dst" else features["kp_known"]
