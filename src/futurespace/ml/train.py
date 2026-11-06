"""Treino e avaliação dos modelos de previsão de Dst e Kp.

Divisão temporal (nunca aleatória, para não vazar informação do futuro):
treino 1995-2014, validação 2015-2018 (escolha de hiperparâmetros), teste 2019 em diante.
O modelo final é re-treinado em treino+validação e avaliado uma única vez no teste.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from futurespace.ml import metrics
from futurespace.ml.features import (
    DST_STORM,
    FEATURE_VERSION,
    KP_STORM,
    build_features,
    build_targets,
    persistence,
)

log = logging.getLogger(__name__)

HORIZONS = {"dst": [1, 3, 6], "kp": [3, 6]}
TRAIN = ("1995-01-01", "2015-01-01")
VALID = ("2015-01-01", "2019-01-01")
TEST_START = "2019-01-01"
MAX_ITER_GRID = (200, 500, 1000)


@dataclass
class Trained:
    target: str
    horizon: int
    model: HistGradientBoostingRegressor
    report: dict


def _model(max_iter: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=max_iter,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=0,
    )


def _is_storm(target: str, values: np.ndarray) -> np.ndarray:
    return values <= DST_STORM if target == "dst" else values >= KP_STORM


def _scores(target: str, y: np.ndarray, pred: np.ndarray) -> dict:
    return {**metrics.regression(y, pred), "storm": metrics.events(_is_storm(target, y), _is_storm(target, pred))}


def _slice(df: pd.DataFrame, start: str, end: str | None = None) -> pd.DataFrame:
    """Linhas em [start, end) — `end` exclusivo para os períodos não se sobreporem."""
    lo = pd.Timestamp(start, tz="UTC")
    if end is None:
        return df.loc[lo:]
    return df.loc[lo : pd.Timestamp(end, tz="UTC") - pd.Timedelta(hours=1)]


def train_one(features: pd.DataFrame, y: pd.Series, target: str, horizon: int) -> Trained:
    data = features.assign(_y=y).dropna(subset=["_y"])
    train, valid = _slice(data, *TRAIN), _slice(data, *VALID)
    test = _slice(data, TEST_START)
    cols = list(features.columns)

    # Escolhe o número de árvores pela validação.
    best_iter, best_rmse = MAX_ITER_GRID[0], float("inf")
    for n in MAX_ITER_GRID:
        pred = _model(n).fit(train[cols], train["_y"]).predict(valid[cols])
        rmse = metrics.regression(valid["_y"].to_numpy(), pred)["rmse"]
        log.info("%s h%d: max_iter=%d rmse_valid=%.3f", target, horizon, n, rmse)
        if rmse < best_rmse:
            best_iter, best_rmse = n, rmse

    final = pd.concat([train, valid])
    model = _model(best_iter).fit(final[cols], final["_y"])

    # Avalia só onde o baseline também existe, para a comparação ser justa.
    base = persistence(test[cols], target)
    test = test[base.notna()]
    y_test = test["_y"].to_numpy()
    report = {
        "target": target,
        "horizon_hours": horizon,
        "max_iter": best_iter,
        "train_rows": len(final),
        "test_rows": len(test),
        "test_period": [str(test.index.min()), str(test.index.max())],
        "model": _scores(target, y_test, model.predict(test[cols])),
        "persistence": _scores(target, y_test, base[base.notna()].to_numpy()),
    }
    return Trained(target, horizon, model, report)


def train_all(hourly: pd.DataFrame) -> tuple[list[Trained], dict]:
    features = build_features(hourly)
    targets = build_targets(hourly, HORIZONS)
    trained = [
        train_one(features, targets[f"{target}_h{h}"], target, h)
        for target, hs in HORIZONS.items()
        for h in hs
    ]
    summary = {
        "feature_version": FEATURE_VERSION,
        "features": list(features.columns),
        "trained_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "splits": {"train": TRAIN, "valid": VALID, "test_start": TEST_START},
        "data_end": str(hourly.index.max()),
        "models": [t.report for t in trained],
    }
    return trained, summary
