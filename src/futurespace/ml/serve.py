"""Previsões ao vivo com os dados coletados em tempo real.

A tabela horária é montada no mesmo formato da usada no treino (OMNI), a partir de:
- vento solar no L1 da sonda operacional (`active`), média de cada hora
- Dst de Kyoto e Kp da NOAA em tempo real
- CMEs do catálogo em tempo real do CACTus

Os modelos ficam no armazenamento em models/<versão>/bundle.joblib, e
models/current.json aponta a versão em uso. models/index.json lista todos os treinos
(publicados ou recusados pela trava), com métricas e importância de cada grupo de variáveis.
"""

from __future__ import annotations

import io
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from futurespace.ml.features import BASE_COLUMNS, build_features
from futurespace.model_index import MODELS_POINTER, index_entry, record
from futurespace.partitions import labels_between, manifest_key, parquet_key, parse_time
from futurespace.storage import Storage

log = logging.getLogger(__name__)

HISTORY = timedelta(days=29)  # recorrência de 27 dias + janelas de 24 h
CME_HISTORY = timedelta(days=8)  # CMEs lançadas até ~7 dias atrás ainda podem estar a caminho
CME_MAX_STALENESS = timedelta(days=3)
MIN_MINUTES = 30  # minutos de dados exigidos para considerar uma hora válida
DST_FILL_HOURS = 3  # o Dst de Kyoto costuma chegar com 1-2 h de atraso


def publish_bundle(store: Storage, bundle_path: Path, summary: dict) -> None:
    version = summary["model_version"]
    base = f"models/{version}"
    store.write(f"{base}/bundle.joblib", bundle_path.read_bytes())
    store.write(f"{base}/report.json", json.dumps(summary, indent=2, ensure_ascii=False).encode())
    pointer = {"model_version": version, "bundle": f"{base}/bundle.joblib", "report": f"{base}/report.json"}
    record(store, index_entry(summary, "published", f"{base}/report.json"), current=version)
    store.write(MODELS_POINTER, json.dumps(pointer).encode())


def record_rejected(store: Storage, summary: dict, reason: str) -> str:
    """Guarda o relatório de um treino que a trava não publicou (sem os modelos, que são grandes).

    Fica numa pasta própria para nunca sobrescrever o relatório de uma versão publicada no mesmo dia.
    """
    stamp = (summary.get("trained_at") or "").replace("-", "").replace(":", "")
    key = f"models/rejected/{summary['model_version']}-{stamp}/report.json"
    store.write(key, json.dumps(summary, indent=2, ensure_ascii=False).encode())
    record(store, index_entry(summary, "rejected", key, reason))
    return key


class NoModelPublished(RuntimeError):
    pass


def current_report(store: Storage) -> dict | None:
    """Relatório de avaliação da versão publicada, ou None se ainda não houver modelo."""
    raw = store.read(MODELS_POINTER)
    if not raw:
        return None
    report = store.read(json.loads(raw)["report"])
    return json.loads(report) if report else None


def load_bundle(store: Storage) -> dict:
    raw = store.read(MODELS_POINTER)
    if not raw:
        raise NoModelPublished("nenhum modelo publicado; rode `futurespace train --publish`")
    # Arquivos joblib executam código ao carregar: só carregamos do nosso próprio armazenamento.
    return joblib.load(io.BytesIO(store.read(json.loads(raw)["bundle"])))


def read_recent(store: Storage, dataset_id: str, start: datetime, end: datetime) -> pd.DataFrame:
    from futurespace import catalog

    raw = store.read(manifest_key(dataset_id))
    if not raw:
        return pd.DataFrame()
    available = set(json.loads(raw)["partitions"])
    labels = [lbl for lbl in labels_between(start, end, catalog.get(dataset_id).partition) if lbl in available]
    frames = [pd.read_parquet(io.BytesIO(store.read(parquet_key(dataset_id, lbl)))) for lbl in labels]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df[(df["time"] >= start) & (df["time"] <= end)]


def _hourly_mean(df: pd.DataFrame, columns: dict[str, str], index: pd.DatetimeIndex) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(index=index, columns=list(columns.values()), dtype=float)
    df = df[df["active"].fillna(False).astype(bool)]
    grouped = df.groupby(df["time"].dt.floor("h"))
    means = grouped[list(columns)].mean().rename(columns=columns)
    enough = grouped.size() >= MIN_MINUTES
    return means[enough].reindex(index)


def hourly_from_realtime(
    plasma: pd.DataFrame, mag: pd.DataFrame, dst: pd.DataFrame, kp: pd.DataFrame, index: pd.DatetimeIndex
) -> pd.DataFrame:
    hourly = _hourly_mean(plasma, {"speed": "speed", "density": "density", "temperature": "temperature"}, index)
    hourly = hourly.join(_hourly_mean(mag, {"bt": "b_abs", "by_gsm": "by_gsm", "bz_gsm": "bz_gsm"}, index))
    # Mesma fórmula do OMNI: pressão dinâmica (nPa) = 2e-6 * n * v^2.
    hourly["pressure"] = 2e-6 * hourly["density"] * hourly["speed"] ** 2
    dst_s = dst.set_index("time")["dst"] if not dst.empty else pd.Series(dtype=float)
    hourly["dst"] = dst_s.reindex(index).ffill(limit=DST_FILL_HOURS)
    kp_s = kp.set_index("time")["kp"].sort_index() if not kp.empty else pd.Series(dtype=float)
    hourly["kp"] = kp_s.reindex(index, method="ffill", limit=2) if not kp_s.empty else np.nan
    return hourly[BASE_COLUMNS].astype(float)


def _history(store: Storage, start: datetime, end: datetime, index: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=index, columns=BASE_COLUMNS)
    omni = read_recent(store, "omni-hourly", start, end)
    if not omni.empty:
        o = omni.set_index("time").sort_index()
        o = o[~o.index.duplicated(keep="last")]
        for col in BASE_COLUMNS:
            if col != "kp":
                out[col] = o[col].reindex(index)
    kp = read_recent(store, "kp-gfz", start - timedelta(hours=3), end)
    if not kp.empty:
        out["kp"] = kp.set_index("time")["kp"].sort_index().reindex(index, method="ffill", limit=2)
    return out


def predict(store: Storage, now: datetime | None = None) -> pd.DataFrame:
    now = now or datetime.now(UTC)
    bundle = load_bundle(store)
    issue = pd.Timestamp(now).floor("h") - pd.Timedelta(hours=1)  # última hora completa
    start = (issue - HISTORY).to_pydatetime()
    end = (issue + pd.Timedelta(minutes=59)).to_pydatetime()
    index = pd.date_range(issue - HISTORY, issue, freq="h")

    realtime = hourly_from_realtime(
        read_recent(store, "solar-wind-plasma", start, end),
        read_recent(store, "solar-wind-mag", start, end),
        read_recent(store, "dst", start, end),
        read_recent(store, "kp", start - timedelta(hours=3), end),
        index,
    )
    # O tempo real tem prioridade; o histórico (OMNI, Kp do GFZ) cobre as horas mais antigas,
    # necessárias para as variáveis de recorrência de 27 dias.
    hourly = realtime.combine_first(_history(store, start, end, index))
    current = realtime.loc[issue]
    if pd.isna(current["speed"]) or pd.isna(current["bz_gsm"]):
        raise RuntimeError(f"sem vento solar suficiente na hora {issue:%Y-%m-%d %H}h UTC; previsão não emitida")

    cmes = None
    if any(c.startswith("cme_") for c in bundle["features"]):
        raw = store.read(manifest_key("cme-cactus"))
        cme_end = parse_time(json.loads(raw)["end"]) if raw else None
        if cme_end and cme_end >= now - CME_MAX_STALENESS:
            cmes = read_recent(store, "cme-cactus", (issue - CME_HISTORY).to_pydatetime(), end)
        else:
            log.warning("catálogo de CMEs desatualizado; variáveis de CME ficam vazias")

    features = build_features(hourly, cmes).reindex(columns=bundle["features"])
    x = features.loc[[issue]]
    rows = []
    for name, m in bundle["models"].items():
        target, horizon = name.split("_h")
        h = int(horizon)
        target_time = issue + pd.Timedelta(hours=h)
        if target == "kp":
            target_time = target_time.floor("3h")
        prob = float(m["classifier"].predict_proba(x)[0, 1])
        rows.append(
            {
                "time": issue + pd.Timedelta(hours=1),
                "target": target,
                "horizon_hours": h,
                "target_time": target_time,
                "value": round(float(m["regressor"].predict(x)[0]), 2),
                "storm_probability": round(prob, 4),
                "storm_alert": prob >= m["threshold"],
                "model_version": bundle["model_version"],
            }
        )
    return pd.DataFrame(rows)
