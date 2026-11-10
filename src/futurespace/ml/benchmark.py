"""Benchmark histórico contra a previsão oficial da NOAA SWPC (3-Day Forecast, desde 2022).

Para cada edição do 3-Day Forecast (00:30 e 12:30 UTC), o modelo prevê com os dados até a
última hora completa antes da edição, e as duas previsões são comparadas com o Kp definitivo
do GFZ nos mesmos blocos de 3 h. O modelo deve ser o treinado só até 2018
(`futurespace train --no-refit`), que nunca viu o período comparado.

Usado por `futurespace benchmark`, que imprime as tabelas em Markdown.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from futurespace.ml import metrics
from futurespace.ml.features import KP_STORM, build_features, hourly_from_history, load_dataset
from futurespace.storage import Storage

# Antecedência da NOAA (início do bloco − emissão), em horas, para a tabela da NOAA sozinha.
LEAD_BINS = ((0, 12), (12, 24), (24, 48), (48, 72))


def noaa_vs_observed(noaa: pd.DataFrame, kp: pd.Series) -> pd.DataFrame:
    """Previsões da NOAA com o Kp observado do bloco e a antecedência em horas."""
    out = noaa.assign(observed=noaa["target_time"].map(kp))
    out["lead_hours"] = (out["target_time"] - out["time"]).dt.total_seconds() / 3600
    return out.dropna(subset=["observed", "kp"])


def model_forecasts(bundle: dict, features: pd.DataFrame, issues: pd.Series) -> pd.DataFrame:
    """Previsões de Kp do modelo emitidas na última hora completa antes de cada edição da NOAA."""
    x_all = features.reindex(columns=bundle["features"])
    hours = pd.DatetimeIndex(sorted(set(issues.dt.floor("h") - pd.Timedelta(hours=1))))
    x = x_all.loc[x_all.index.intersection(hours)]
    x = x[x[["speed", "bz_gsm"]].notna().all(axis=1)]  # sem vento solar não há previsão
    rows = []
    for name, model in bundle["models"].items():
        target, horizon = name.split("_h")
        if target != "kp" or x.empty:
            continue
        h = int(horizon)
        rows.append(
            pd.DataFrame(
                {
                    "issue_hour": x.index,
                    "target_time": (x.index + pd.Timedelta(hours=h)).floor("3h"),
                    "horizon_hours": h,
                    "model": model["regressor"].predict(x),
                    "alert": model["classifier"].predict_proba(x)[:, 1] >= model["threshold"],
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def compare(noaa: pd.DataFrame, model: pd.DataFrame) -> pd.DataFrame:
    """Mesmos blocos e mesma emissão: modelo × NOAA × observado."""
    noaa = noaa.assign(issue_hour=noaa["time"].dt.floor("h") - pd.Timedelta(hours=1))
    return noaa.merge(model, on=["issue_hour", "target_time"])


def _scores(observed: np.ndarray, value: np.ndarray, storm_forecast: np.ndarray) -> dict:
    reg = metrics.regression(observed, value)
    ev = metrics.events(observed >= KP_STORM, storm_forecast)
    return {"mae": reg["mae"], "rmse": reg["rmse"], "pod": ev["pod"], "far": ev["far"], "hss": ev["hss"]}


def _fmt(x: float) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:.2f}".replace(".", ",")


def markdown(noaa: pd.DataFrame, both: pd.DataFrame) -> str:
    lines = [
        (
            f"Edições da NOAA com Kp observado: {noaa['time'].nunique()} "
            f"({noaa['time'].min():%m/%Y} a {noaa['time'].max():%m/%Y}), {len(noaa)} blocos previstos."
        ),
        "",
        "**NOAA SWPC × Kp observado (GFZ), por antecedência.**",
        "",
        "| Antecedência | n | MAE | RMSE | POD | FAR | HSS |",
        "|---|---|---|---|---|---|---|",
    ]
    for lo, hi in LEAD_BINS:
        g = noaa[(noaa["lead_hours"] > lo) & (noaa["lead_hours"] <= hi)]
        if g.empty:
            continue
        s = _scores(g["observed"].to_numpy(), g["kp"].to_numpy(), g["kp"].to_numpy() >= KP_STORM)
        lines.append(f"| {lo}–{hi} h | {len(g)} | " + " | ".join(_fmt(s[k]) for k in ("mae", "rmse", "pod", "far", "hss")) + " |")

    if both.empty:
        return "\n".join(lines)
    lines += [
        "",
        (
            f"**Modelo × NOAA nos mesmos blocos e com a mesma emissão** "
            f"({both['issue_hour'].min():%m/%Y} a {both['issue_hour'].max():%m/%Y}). "
            "Tempestade = Kp ≥ 5− (G1); alerta do modelo pelo limiar do classificador; "
            "alerta da NOAA quando o Kp previsto é ≥ 5−."
        ),
        "",
        (
            "| Horizonte do modelo | n | Tempestades | MAE modelo | MAE NOAA | RMSE modelo | RMSE NOAA "
            "| POD modelo | POD NOAA | FAR modelo | FAR NOAA | HSS modelo | HSS NOAA |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for h, g in both.groupby("horizon_hours"):
        if len(g) < 30:  # combinações raras (ex.: +3 h cai no bloco em andamento da NOAA)
            continue
        obs = g["observed"].to_numpy()
        m = _scores(obs, g["model"].to_numpy(), g["alert"].to_numpy())
        n = _scores(obs, g["kp"].to_numpy(), g["kp"].to_numpy() >= KP_STORM)
        cells = [_fmt(score[k]) for k in ("mae", "rmse", "pod", "far", "hss") for score in (m, n)]
        lines.append(f"| +{h} h | {len(g)} | {int((obs >= KP_STORM).sum())} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run(store: Storage, bundle: dict) -> str:
    kp_gfz = load_dataset(store, "kp-gfz")
    kp = kp_gfz.set_index("time")["kp"].sort_index()
    noaa = noaa_vs_observed(load_dataset(store, "kp-forecast-3day"), kp)
    hourly = hourly_from_history(load_dataset(store, "omni-hourly"), kp_gfz)
    features = build_features(hourly, load_dataset(store, "cme-cactus"))
    both = compare(noaa, model_forecasts(bundle, features, noaa["time"]))
    return markdown(noaa, both)
