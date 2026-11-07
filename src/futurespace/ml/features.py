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

from futurespace.model_index import FEATURE_GROUPS  # noqa: F401 (reexportado para o treino)
from futurespace.partitions import manifest_key, parquet_key
from futurespace.pipeline.connectors.cactus import prefer_curated
from futurespace.storage import Storage

FEATURE_VERSION = 3

# Rotação solar vista da Terra (~27,3 dias): feixes rápidos de vento solar tendem a voltar.
ROTATION_HOURS = 27 * 24

# Atraso entre o início da CME e ela aparecer no catálogo em tempo real do CACTus.
CME_LATENCY = pd.Timedelta(hours=6)
CME_WIDE = 90  # graus: CMEs largas (halo parcial ou total) podem atingir a Terra
AU_KM = 1.496e8

# Colunas da tabela horária que alimenta as features.
SOLAR_WIND = ["b_abs", "by_gsm", "bz_gsm", "speed", "density", "temperature", "pressure"]
BASE_COLUMNS = [*SOLAR_WIND, "dst", "kp"]

# Kp >= 5- (4.67) corresponde a tempestade G1 na escala da NOAA.
KP_STORM = 4.67
DST_STORM = -50.0


_SOLAR_WIND_PREFIXES = {"b", "by", "bz", "speed", "density", "temperature", "pressure", "newell", "vbs", "clock"}


def feature_group(name: str) -> str:
    """Grupo da variável (chaves de FEATURE_GROUPS), usado na importância por permutação."""
    if name.startswith("rec27"):
        return "recurrence"
    if name.startswith("cme_"):
        return "cmes"
    if name.startswith(("dst", "kp_known")):
        return "earth_state"
    if name.startswith(("doy", "ut_")):
        return "calendar"
    if name.split("_")[0] in _SOLAR_WIND_PREFIXES:
        return "solar_wind"
    raise ValueError(f"variável sem grupo: {name}")


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


def cme_features(cmes: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Variáveis sobre CMEs recentes, conhecidas só a partir de t0 + CME_LATENCY.

    Antes do início do catálogo os valores ficam vazios (desconhecido), não zero.
    """
    f = pd.DataFrame(index=index)
    if "catalog" in cmes.columns:
        # O armazenamento acumula detecções em tempo real já substituídas pelo quicklook.
        cmes = prefer_curated(cmes)
    wide = cmes[cmes["width"] >= CME_WIDE]
    known = (wide["time"] + CME_LATENCY).dt.ceil("h")

    speed = wide.groupby(known)["speed"].max().reindex(index)
    count = wide.groupby(known).size().reindex(index, fill_value=0)
    for win in (24, 48, 72, 96):
        f[f"cme_wide_speed_max{win}"] = speed.rolling(win, min_periods=1).max().fillna(0)
    f["cme_wide_count72"] = count.rolling(72, min_periods=1).sum()

    # Chegada estimada (balística, velocidade constante) da CME larga mais próxima.
    transit = pd.to_timedelta(AU_KM / wide["speed"].clip(lower=200) / 3600, unit="h")
    arrival = (wide["time"] + transit).to_numpy()
    rows = []
    for k, a, v, w in zip(known.to_numpy(), arrival, wide["speed"], wide["width"], strict=True):
        hours = pd.date_range(k, a + np.timedelta64(12, "h"), freq="h")[:150]
        eta = (a - hours.to_numpy()) / np.timedelta64(1, "h")
        rows.append(pd.DataFrame({"hour": hours, "eta": eta, "speed": v, "width": w}))
    if rows:
        eta = pd.concat(rows, ignore_index=True)
        eta = eta.loc[eta.groupby("hour")["eta"].idxmin()].set_index("hour").reindex(index)
        f["cme_eta_hours"] = eta["eta"]
        f["cme_eta_speed"] = eta["speed"]
        f["cme_eta_width"] = eta["width"]

    if not cmes.empty:
        f.loc[f.index < cmes["time"].min()] = np.nan
    return f


def build_features(hourly: pd.DataFrame, cmes: pd.DataFrame | None = None) -> pd.DataFrame:
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

    # Recorrência: como estava uma rotação solar atrás, nas 24 h equivalentes às próximas 24 h.
    # shift(R - 24).rolling(24) em t cobre [t - R + 1, t - R + 24].
    ahead = ROTATION_HOURS - 24
    f["rec27_speed_now"] = h["speed"].shift(ROTATION_HOURS)
    f["rec27_speed_mean24"] = h["speed"].shift(ahead).rolling(24, min_periods=12).mean()
    f["rec27_kp_max24"] = h["kp"].shift(ahead).rolling(24, min_periods=12).max()
    f["rec27_dst_min24"] = h["dst"].shift(ahead).rolling(24, min_periods=12).min()

    # Sazonalidade (efeito Russell-McPherron) e hora do dia.
    doy = f.index.day_of_year.to_numpy()
    hour = f.index.hour.to_numpy()
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    f["ut_sin"] = np.sin(2 * np.pi * hour / 24)
    f["ut_cos"] = np.cos(2 * np.pi * hour / 24)
    if cmes is not None:
        f = f.join(cme_features(cmes, f.index))
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
