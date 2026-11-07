"""Previsão de visibilidade da aurora por cidade, de 0 a 24 h à frente (fórmula v0).

O resultado é uma pontuação de 0 a 1 (mais alto = condições melhores), não uma probabilidade:
os pesos ainda não foram comparados com auroras observadas. Quando forem calibrados, a
probabilidade entra numa coluna própria.

visibility_score = aurora_score × (1 − nuvens) × escuridão × fator da Lua

- aurora_score: logística da diferença entre o Kp esperado e o Kp necessário para a cidade,
  com a mesma regra do mapa do site (linha do horizonte ≈ 66° − 1,5° × Kp de latitude
  magnética). Usada em todas as antecedências, para a série ser coerente ao longo das 24 h.
- OVATION (só horas 0 e 1): a maior probabilidade num raio de 500 km (a aurora, a ~110 km de
  altura, aparece no horizonte a essa distância), com a sua própria visibilidade em
  `visibility_score_ovation`. Fica separado porque está noutra escala (raramente passa de
  ~50%) e depende da hora local magnética, que a regra do Kp ignora; o histórico desse mapa
  serve para calibrar a próxima versão da fórmula.
- Kp esperado: previsão dos modelos do projeto para o bloco de 3 h (os modelos cobrem os
  blocos de +3, +6, +12 e +24 h; os blocos entre eles são interpolados); antes do primeiro
  bloco previsto, o último Kp medido.
- escuridão: 0 com o Sol acima de −6°, 1 abaixo de −15°, linear entre os dois.
- Lua: acima do horizonte, reduz a pontuação em até 30% (Lua cheia).

Os pesos ainda não foram calibrados com observações: cada componente fica numa coluna para
a fórmula poder ser ajustada depois (por exemplo, com câmeras all-sky).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from futurespace.aurora import astro
from futurespace.aurora.sites import SITES
from futurespace.ml.serve import read_recent
from futurespace.storage import Storage

log = logging.getLogger(__name__)

METHOD = "v0"
LEAD_HOURS = range(25)
OVATION_LEADS = 1  # o OVATION vale por ~30 a 90 min: usado só nas horas 0 e 1
OVATION_MAX_AGE = timedelta(hours=2)
PREDICTIONS_MAX_AGE = timedelta(hours=3)
VIEW_RADIUS_KM = 500
KP_SCALE = 0.5  # largura da logística, em unidades de Kp
MOON_PENALTY = 0.3
EARTH_RADIUS_KM = 6371.0
# Polo norte geomagnético (dipolo do IGRF, ~2025), o mesmo do mapa do site.
POLE_LAT, POLE_LON = 80.8, -72.7


def mag_lat(lat, lon) -> np.ndarray:
    rad = np.pi / 180
    lat, lon = np.asarray(lat) * rad, np.asarray(lon) * rad
    s = np.sin(lat) * np.sin(POLE_LAT * rad) + np.cos(lat) * np.cos(POLE_LAT * rad) * np.cos(lon - POLE_LON * rad)
    return np.arcsin(np.clip(s, -1, 1)) / rad


def kp_needed(abs_mag_lat: float) -> float:
    """Menor Kp em que a aurora chega ao horizonte da latitude magnética dada (inf se nunca).

    Inversa da linha do horizonte do mapa do site: 66 − 1,5·Kp, que desce mais rápido acima
    de Kp 8 (66 − 1,5·Kp − 8·(Kp − 8)).
    """
    kp = (66 - abs_mag_lat) / 1.5
    if kp > 8:
        kp = 8 + (54 - abs_mag_lat) / 9.5
    return float("inf") if kp > 9 else max(kp, 0.0)


def ovation_nearby(grid: pd.DataFrame, lat: float, lon: float) -> float:
    """Maior probabilidade do OVATION num raio de VIEW_RADIUS_KM da cidade."""
    if grid.empty:
        return 0.0
    rad = np.pi / 180
    glat, glon = grid["lat"].to_numpy() * rad, grid["lon"].to_numpy() * rad
    a = np.sin((glat - lat * rad) / 2) ** 2 + np.cos(lat * rad) * np.cos(glat) * np.sin((glon - lon * rad) / 2) ** 2
    dist = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))
    near = grid["aurora"].to_numpy()[dist <= VIEW_RADIUS_KM]
    return float(near.max()) if near.size else 0.0


def _kp_timeline(predictions: pd.DataFrame, kp_observed: pd.DataFrame, issue: pd.Timestamp):
    """(blocos previstos {início do bloco: Kp}, último Kp medido)."""
    observed = float(kp_observed.sort_values("time")["kp"].iloc[-1]) if not kp_observed.empty else np.nan
    blocks = pd.Series(dtype=float)
    if not predictions.empty:
        kp = predictions[predictions["target"] == "kp"]
        if not kp.empty and kp["time"].max() >= issue - PREDICTIONS_MAX_AGE:
            latest = kp[kp["time"] == kp["time"].max()].sort_values("horizon_hours")
            latest = latest.drop_duplicates("target_time")  # o menor horizonte vence
            blocks = latest.set_index("target_time")["value"].sort_index()
    return blocks, observed


def _expected_kp(target: pd.Timestamp, blocks: pd.Series, observed: float) -> tuple[float, str]:
    if blocks.empty or target < blocks.index.min():
        return observed, "observed"
    block = target.floor("3h")
    x = (blocks.index - blocks.index[0]).total_seconds().to_numpy()
    # Interpolação linear entre os blocos previstos; depois do último, o último valor.
    return float(np.interp((block - blocks.index[0]).total_seconds(), x, blocks.to_numpy())), "model"


def compute(
    ovation: pd.DataFrame,
    weather: pd.DataFrame,
    predictions: pd.DataFrame,
    kp_observed: pd.DataFrame,
    now: datetime,
) -> pd.DataFrame:
    issue = pd.Timestamp(now).floor("h")
    targets = pd.DatetimeIndex([issue + pd.Timedelta(hours=h) for h in LEAD_HOURS])

    grid = pd.DataFrame()
    if not ovation.empty:
        latest = ovation[ovation["time"] == ovation["time"].max()]
        if abs(latest["forecast_time"].iloc[0] - issue) <= OVATION_MAX_AGE:
            grid = latest
        else:
            log.warning("mapa OVATION desatualizado (%s); usando só o Kp", latest["forecast_time"].iloc[0])

    clouds = {}
    if not weather.empty:
        clouds = weather.set_index(["time", "site"])["cloud_cover"].to_dict()

    blocks, observed = _kp_timeline(predictions, kp_observed, issue)
    expected = [_expected_kp(t, blocks, observed) for t in targets]
    kp_values = np.array([k for k, _ in expected], dtype=float)
    kp_sources = [s for _, s in expected]
    illumination = astro.moon_illumination(targets)

    rows = []
    for site in SITES:
        m = float(mag_lat(site.lat, site.lon))
        need = kp_needed(abs(m))
        nearby = ovation_nearby(grid, site.lat, site.lon) if not grid.empty else np.nan
        sun = astro.sun_altitude(targets, site.lat, site.lon)
        moon = astro.moon_altitude(targets, site.lat, site.lon)
        with np.errstate(over="ignore"):
            chance_kp = 1 / (1 + np.exp(-(kp_values - need) / KP_SCALE)) if np.isfinite(need) else np.zeros(len(targets))
        darkness = np.clip((-6 - sun) / 9, 0, 1)
        moon_factor = np.where(moon > 0, 1 - MOON_PENALTY * illumination, 1.0)
        for i, (target, lead) in enumerate(zip(targets, LEAD_HOURS, strict=True)):
            chance_ovation = nearby / 100 if lead <= OVATION_LEADS and not np.isnan(nearby) else np.nan
            chance = chance_kp[i]
            cloud = clouds.get((target, site.id), np.nan)
            sky = (1 - cloud / 100) * darkness[i] * moon_factor[i]
            rows.append(
                {
                    "time": issue,
                    "site": site.id,
                    "target_time": target,
                    "lead_hours": lead,
                    "latitude": site.lat,
                    "longitude": site.lon,
                    "mag_lat": round(m, 2),
                    "kp_expected": round(float(kp_values[i]), 2),
                    "kp_source": kp_sources[i],
                    "kp_needed": round(need, 2) if np.isfinite(need) else np.nan,
                    "ovation_nearby": nearby if lead <= OVATION_LEADS else np.nan,
                    "aurora_score_ovation": chance_ovation,
                    "aurora_score_kp": round(float(chance_kp[i]), 4),
                    "aurora_score": round(float(chance), 4),
                    "cloud_cover": cloud,
                    "sun_altitude": round(float(sun[i]), 2),
                    "moon_altitude": round(float(moon[i]), 2),
                    "moon_illumination": round(float(illumination[i]), 3),
                    "darkness": round(float(darkness[i]), 3),
                    "visibility_score": round(float(chance * sky), 4),
                    "visibility_score_ovation": round(float(chance_ovation * sky), 4),
                    "method": METHOD,
                }
            )
    return pd.DataFrame(rows)


def forecast(store: Storage, now: datetime | None = None) -> pd.DataFrame:
    now = now or datetime.now(UTC)
    return compute(
        read_recent(store, "aurora-ovation", now - timedelta(hours=3), now),
        read_recent(store, "aurora-weather", now - timedelta(hours=1), now + timedelta(hours=26)),
        read_recent(store, "predictions", now - timedelta(hours=6), now),
        read_recent(store, "kp", now - timedelta(days=2), now),
        now,
    )
