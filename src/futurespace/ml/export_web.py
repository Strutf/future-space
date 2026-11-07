"""Arquivos estáticos da parte interativa do site, que rodam no navegador sem servidor.

- sim-model.json: modelo enxuto do simulador "Lance sua tempestade". Recebe só o vento
  solar (velocidade, densidade e Bz, médias de 6 h, como numa tempestade sustentada) e
  devolve Kp e Dst. As árvores são exportadas para JSON e avaliadas em JavaScript.
- replays.json: tempestades famosas hora a hora (dados reais do OMNI e do GFZ) com o que
  a IA teria previsto 3 h antes, usando um modelo treinado só até 2018.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from futurespace.ml.features import build_features

# O simulador recebe velocidade, densidade e Bz, mas o modelo usa VBs (velocidade × Bz para o
# sul), que é zero com o "ímã" para o norte. Com o Bz bruto, o modelo aprendia que vento rápido
# com campo forte dá tempestade mesmo com Bz para o norte (no histórico os dois andam juntos),
# o que contradiz a física que o simulador quer ensinar.
SIM_INPUTS = ["speed", "density", "bz_gsm"]
SIM_FEATURES = ["speed", "density", "vbs"]
SIM_WINDOW_HOURS = 6


def sim_features(speed, density, bz):
    """Mesma conta do JavaScript: VBs em mV/m."""
    return [speed, density, speed * max(-bz, 0) / 1000]

# Janelas (UTC) das tempestades da "Viagem no tempo".
EVENTS = {
    "may2024": ("2024-05-10T06:00Z", "2024-05-12T12:00Z"),
    "stpatrick2015": ("2015-03-16T18:00Z", "2015-03-18T18:00Z"),
    "halloween2003": ("2003-10-29T00:00Z", "2003-10-31T12:00Z"),
    "quebec1989": ("1989-03-12T18:00Z", "1989-03-15T00:00Z"),
}
TRAINING_END = pd.Timestamp("2019-01-01", tz="UTC")  # o modelo da viagem viu dados até 2018


def export_trees(model: HistGradientBoostingRegressor) -> dict:
    """Árvores do HistGradientBoosting num formato compacto.

    Cada nó: [variável, limiar, filho esquerdo, filho direito, valor, faltante vai à esquerda].
    Folhas têm variável -1. Predição = baseline + soma do valor da folha de cada árvore.
    Usa atributos internos do scikit-learn: a versão é fixada em requirements-pipeline.txt.
    """
    trees = []
    for (predictor,) in model._predictors:
        nodes = []
        for n in predictor.nodes:
            if n["is_leaf"]:
                nodes.append([-1, 0, 0, 0, round(float(n["value"]), 6), 0])
            else:
                nodes.append([
                    int(n["feature_idx"]), float(n["num_threshold"]), int(n["left"]), int(n["right"]),
                    0, int(n["missing_go_to_left"]),
                ])
        trees.append(nodes)
    return {"baseline": float(np.ravel(model._baseline_prediction)[0]), "trees": trees}


def predict_exported(exported: dict, x: list[float]) -> float:
    """Mesma lógica do JavaScript do site; usada nos testes para garantir que batem."""
    total = exported["baseline"]
    for nodes in exported["trees"]:
        i = 0
        while nodes[i][0] != -1:
            feature, threshold, left, right, _, missing_left = nodes[i]
            value = x[feature]
            go_left = missing_left if value is None or np.isnan(value) else value <= threshold
            i = left if go_left else right
        total += nodes[i][4]
    return total


def train_simulator(hourly: pd.DataFrame) -> dict:
    hourly = hourly.assign(vbs=hourly["speed"] * (-hourly["bz_gsm"]).clip(lower=0) / 1000)
    conditions = hourly[SIM_FEATURES].rolling(SIM_WINDOW_HOURS, min_periods=4).mean()
    data = conditions.assign(kp=hourly["kp"], dst=hourly["dst"]).loc["1995-01-01":].dropna()
    models = {}
    # Monotonia física: mais velocidade, densidade ou VBs nunca diminuem a tempestade.
    for target, direction in (("kp", 1), ("dst", -1)):
        model = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=50, random_state=0,
            monotonic_cst=[direction] * len(SIM_FEATURES),
        ).fit(data[SIM_FEATURES], data[target])
        models[target] = export_trees(model)
    return {
        "inputs": SIM_INPUTS,
        "features": SIM_FEATURES,
        # Maiores médias de 6 h já vistas: além disso, as árvores não conseguem extrapolar.
        "limits": {
            "speed_max": round(float(data["speed"].max()), 0),
            "vbs_max": round(float(data["vbs"].max()), 1),
        },
        "window_hours": SIM_WINDOW_HOURS,
        "trained_rows": len(data),
        "trained_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "models": models,
    }


def _clean(v, digits: int = 1):
    return None if v is None or pd.isna(v) else round(float(v), digits)


def build_replays(hourly: pd.DataFrame, cmes: pd.DataFrame, bundle: dict) -> dict:
    features = build_features(hourly, cmes).reindex(columns=bundle["features"])
    kp_model = bundle["models"]["kp_h3"]
    dst_model = bundle["models"]["dst_h3"]
    events = {}
    for key, (start, end) in EVENTS.items():
        hours = pd.date_range(start, end, freq="h")
        # Previsão para a hora t, emitida em t - 3 h (o modelo de 3 h à frente).
        x = features.reindex(hours - pd.Timedelta(hours=3))
        valid = x[["speed", "bz_gsm"]].notna().all(axis=1).to_numpy()
        ia_kp = np.full(len(hours), np.nan)
        ia_dst = np.full(len(hours), np.nan)
        ia_prob = np.full(len(hours), np.nan)
        if valid.any():
            xv = x[valid]
            ia_kp[valid] = np.clip(kp_model["regressor"].predict(xv), 0, 9)
            ia_dst[valid] = dst_model["regressor"].predict(xv)
            ia_prob[valid] = kp_model["classifier"].predict_proba(xv)[:, 1]
        obs = hourly.reindex(hours)
        rows = [
            {
                "time": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "speed": _clean(obs.at[t, "speed"], 0),
                "density": _clean(obs.at[t, "density"]),
                "bz": _clean(obs.at[t, "bz_gsm"]),
                "kp": _clean(obs.at[t, "kp"], 2),
                "dst": _clean(obs.at[t, "dst"], 0),
                "ia_kp": _clean(ia_kp[i], 2),
                "ia_dst": _clean(ia_dst[i], 0),
                "ia_prob": _clean(ia_prob[i], 3),
            }
            for i, t in enumerate(hours)
        ]
        events[key] = {
            "start": start,
            "end": end,
            "in_training": bool(hours[0] < TRAINING_END),
            "solar_wind_coverage": round(float(valid.mean()), 2),
            "hours": rows,
        }
    return {"model_version": bundle["model_version"], "events": events}
