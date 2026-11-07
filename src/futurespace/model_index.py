"""Histórico dos modelos: um índice com todos os treinos, publicados ou recusados.

Fica em models/index.json, ao lado de models/<versão>/report.json (ficha completa de cada treino).
Só usa json, para a API poder ler sem carregar as bibliotecas de machine learning.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from futurespace.storage import Storage

MODELS_POINTER = "models/current.json"
MODELS_INDEX = "models/index.json"

# Grupos de variáveis usados para medir o que a IA mais valoriza (importância por permutação).
FEATURE_GROUPS = {
    "solar_wind": "Vento solar no L1",
    "earth_state": "Estado atual da Terra (Kp/Dst)",
    "recurrence": "Recorrência de 27 dias",
    "cmes": "CMEs vistas no Sol",
    "calendar": "Calendário",
}


def _model_entry(m: dict) -> dict:
    storm = m["storm"]
    importance = m.get("importance") or {}
    return {
        "target": m["target"],
        "horizon_hours": m["horizon_hours"],
        "rmse": m["regression"]["model"]["rmse"],
        "rmse_persistence": m["regression"]["persistence"]["rmse"],
        "bss": round(1 - storm["brier"] / storm["brier_climatology"], 4),
        "hss": storm["classifier"].get("hss"),
        "importance": (importance.get("regression") or {}).get("share"),
        "importance_storm": (importance.get("storm") or {}).get("share"),
    }


def index_entry(summary: dict, status: str, report_key: str, reason: str | None = None) -> dict:
    repro = summary.get("reproducibility") or {}
    return {
        # Pasta do relatório: a versão (publicados) ou versão + horário do treino (recusados).
        "id": report_key.split("/")[-2],
        "model_version": summary["model_version"],
        "status": status,
        "reason": reason,
        "trained_at": summary.get("trained_at"),
        "recorded_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "feature_version": summary.get("feature_version"),
        "features": len(summary.get("features", [])),
        "uses_cmes": summary.get("uses_cmes"),
        "data_end": summary.get("data_end"),
        # Versões treinadas antes de existir a ficha de reprodutibilidade ficam com None.
        "code_commit": (repro.get("code") or {}).get("commit"),
        "data_revision": (repro.get("data") or {}).get("revision"),
        "report": report_key,
        "models": [_model_entry(m) for m in summary.get("models", [])],
    }


def read_index(store: Storage) -> dict:
    raw = store.read(MODELS_INDEX)
    if raw:
        return json.loads(raw)
    index = {"current": None, "versions": []}
    # Primeiro uso: inclui a versão publicada antes de o índice existir.
    pointer = store.read(MODELS_POINTER)
    if pointer:
        p = json.loads(pointer)
        report = store.read(p["report"])
        index["current"] = p["model_version"]
        if report:
            index["versions"].append(index_entry(json.loads(report), "published", p["report"]))
    return index


def record(store: Storage, entry: dict, current: str | None = None) -> None:
    index = read_index(store)
    versions = [v for v in index["versions"] if v["report"] != entry["report"]] + [entry]
    index["versions"] = sorted(versions, key=lambda v: v.get("trained_at") or "", reverse=True)
    if current:
        index["current"] = current
    store.write(MODELS_INDEX, json.dumps(index, indent=2, ensure_ascii=False).encode())
