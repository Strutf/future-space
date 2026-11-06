"""Verificação das previsões: compara o que foi previsto com o que aconteceu.

Para cada previsão cujo alvo já passou, registra o valor observado, o erro e duas
referências calculadas no mesmo instante da emissão:
- persistência: o último valor conhecido do índice ("vai continuar igual")
- NOAA (só Kp): a previsão oficial mais recente emitida até aquele instante

As linhas vão para o dataset `verification`, e um resumo com as métricas dos últimos
7 e 30 dias vai para json/verification/summary.json (servido pela API).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from futurespace.ml.features import DST_STORM, KP_STORM
from futurespace.ml.serve import read_recent
from futurespace.partitions import VERIFICATION_SUMMARY_KEY, format_time
from futurespace.storage import Storage

WINDOWS_DAYS = (7, 30)


def _observed_kp(store: Storage, start: datetime, end: datetime) -> pd.Series:
    """Kp por início de bloco de 3 h: GFZ quando existir, senão a estimativa da NOAA."""
    gfz = read_recent(store, "kp-gfz", start, end)
    noaa = read_recent(store, "kp", start, end)
    series = [df.set_index("time")["kp"] for df in (gfz, noaa) if not df.empty]
    if not series:
        return pd.Series(dtype=float)
    kp = series[0] if len(series) == 1 else series[0].combine_first(series[1])
    return kp[~kp.index.duplicated(keep="first")].sort_index()


def _is_storm(target: pd.Series, values: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(target == "dst", values <= DST_STORM, values >= KP_STORM), index=values.index
    ).where(values.notna())


def verify(store: Storage, now: datetime | None = None, days: int = 30) -> pd.DataFrame:
    now = now or datetime.now(UTC)
    start = now - timedelta(days=days + 2)
    preds = read_recent(store, "predictions", start, now)
    if preds.empty:
        return pd.DataFrame()

    dst = read_recent(store, "dst", start - timedelta(days=1), now)
    dst = dst.set_index("time")["dst"].sort_index() if not dst.empty else pd.Series(dtype=float)
    kp = _observed_kp(store, start - timedelta(days=1), now)

    p = preds.copy()
    issue_hour = p["time"] - pd.Timedelta(hours=1)
    is_dst = p["target"] == "dst"

    # Observado: Dst da hora-alvo; Kp do bloco-alvo, só depois que o bloco terminou.
    observed = pd.Series(np.nan, index=p.index)
    observed[is_dst] = dst.reindex(p.loc[is_dst, "target_time"]).to_numpy()
    kp_done = ~is_dst & (p["target_time"] + pd.Timedelta(hours=3) <= now)
    observed[kp_done] = kp.reindex(p.loc[kp_done, "target_time"]).to_numpy()
    p["observed"] = observed

    # Persistência no momento da emissão.
    persistence = pd.Series(np.nan, index=p.index)
    persistence[is_dst] = dst.reindex(issue_hour[is_dst]).to_numpy()
    last_block = p["time"].dt.floor("3h") - pd.Timedelta(hours=3)
    persistence[~is_dst] = kp.reindex(last_block[~is_dst]).to_numpy()
    p["persistence"] = persistence

    # Previsão da NOAA mais recente emitida até a nossa, para o mesmo bloco de Kp.
    p["noaa"] = np.nan
    noaa = read_recent(store, "kp-forecast", start - timedelta(days=3), now)
    if not noaa.empty:
        noaa = noaa.sort_values("time")
        kp_rows = p.loc[~is_dst, ["time", "target_time"]].sort_values("time")
        matched = pd.merge_asof(
            kp_rows.reset_index(),
            noaa.rename(columns={"time": "noaa_time", "kp": "noaa"})[["noaa_time", "target_time", "noaa"]],
            left_on="time",
            right_on="noaa_time",
            by="target_time",
            direction="backward",
        ).set_index("index")
        p.loc[matched.index, "noaa"] = matched["noaa"]

    p = p[p["observed"].notna()]
    out = pd.DataFrame(
        {
            "time": p["time"],
            "target": p["target"],
            "horizon_hours": p["horizon_hours"],
            "target_time": p["target_time"],
            "predicted": p["value"],
            "observed": p["observed"],
            "error": p["value"] - p["observed"],
            "persistence": p["persistence"],
            "noaa": p["noaa"],
            "storm_probability": p["storm_probability"],
            "storm_alert": p["storm_alert"],
            "storm_observed": _is_storm(p["target"], p["observed"]),
            "model_version": p["model_version"],
        }
    )
    return out.reset_index(drop=True)


def _mae(err: pd.Series) -> float | None:
    err = err.dropna()
    return round(float(err.abs().mean()), 3) if len(err) else None


def summarize(rows: pd.DataFrame, now: datetime) -> dict:
    """Métricas por índice/horizonte nas janelas de 7 e 30 dias."""
    windows = {}
    for days in WINDOWS_DAYS:
        recent = rows[rows["target_time"] >= now - timedelta(days=days)]
        groups = []
        for (target, horizon), g in recent.groupby(["target", "horizon_hours"]):
            storm = g["storm_observed"].astype(bool)
            alert = g["storm_alert"].astype(bool)
            groups.append(
                {
                    "target": target,
                    "horizon_hours": int(horizon),
                    "n": len(g),
                    "mae_model": _mae(g["error"]),
                    "mae_persistence": _mae(g["persistence"] - g["observed"]),
                    "mae_noaa": _mae(g["noaa"] - g["observed"]) if target == "kp" else None,
                    "storms_observed": int(storm.sum()),
                    "storm_hits": int((storm & alert).sum()),
                    "storm_misses": int((storm & ~alert).sum()),
                    "false_alarms": int((~storm & alert).sum()),
                }
            )
        windows[f"{days}d"] = groups
    return {"updated_at": format_time(now), "windows": windows}


def write_summary(store: Storage, summary: dict) -> None:
    store.write(VERIFICATION_SUMMARY_KEY, json.dumps(summary, ensure_ascii=False).encode())
