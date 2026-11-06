"""Treino e avaliação dos modelos de previsão de Dst e Kp.

Para cada índice e horizonte são treinados dois modelos:
- regressão: o valor previsto (ex.: Dst daqui a 6 h)
- classificação: a probabilidade de tempestade (Kp >= 5 ou Dst <= -50), com um limiar
  de alerta escolhido na validação para maximizar o TSS

Divisão temporal (nunca aleatória, para não vazar informação do futuro):
treino 1995-2014, validação 2015-2018 (hiperparâmetros e limiar), teste 2019 em diante.
O modelo final é re-treinado em treino+validação e avaliado uma única vez no teste.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

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

HORIZONS = {"dst": [1, 3, 6, 12, 24], "kp": [3, 6, 12, 24]}
TRAIN = ("1995-01-01", "2015-01-01")
VALID = ("2015-01-01", "2019-01-01")
TEST_START = "2019-01-01"
MAX_ITER_GRID = (200, 500, 1000)
CLASSIFIER_ITER = 400
THRESHOLDS = np.round(np.arange(0.02, 0.91, 0.01), 2)

_TREE_PARAMS = {
    "learning_rate": 0.05,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 40,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": 0,
}


@dataclass
class Trained:
    target: str
    horizon: int
    regressor: HistGradientBoostingRegressor
    classifier: HistGradientBoostingClassifier
    threshold: float
    report: dict

    @property
    def name(self) -> str:
        return f"{self.target}_h{self.horizon}"


def is_storm(target: str, values: np.ndarray) -> np.ndarray:
    return values <= DST_STORM if target == "dst" else values >= KP_STORM


def _slice(df: pd.DataFrame, start: str, end: str | None = None) -> pd.DataFrame:
    """Linhas em [start, end) — `end` exclusivo para os períodos não se sobreporem."""
    lo = pd.Timestamp(start, tz="UTC")
    if end is None:
        return df.loc[lo:]
    return df.loc[lo : pd.Timestamp(end, tz="UTC") - pd.Timedelta(hours=1)]


def _best_threshold(storm: np.ndarray, prob: np.ndarray) -> float:
    """Limiar de alerta que maximiza o HSS na validação.

    O HSS pune alarmes falsos; o TSS quase não pune quando o evento é raro, e maximizá-lo
    levava a limiares baixos com mais de 80% de alertas falsos.
    """
    hss = [metrics.events(storm, prob >= t)["hss"] for t in THRESHOLDS]
    return float(THRESHOLDS[int(np.nanargmax(hss))])


def train_one(
    features: pd.DataFrame, y: pd.Series, target: str, horizon: int, refit_all: bool = True
) -> Trained:
    data = features.assign(_y=y).dropna(subset=["_y"])
    cols = list(features.columns)
    train, valid, test = _slice(data, *TRAIN), _slice(data, *VALID), _slice(data, TEST_START)
    storm_train = is_storm(target, train["_y"].to_numpy())
    storm_valid = is_storm(target, valid["_y"].to_numpy())

    best_iter, best_rmse = MAX_ITER_GRID[0], float("inf")
    for n in MAX_ITER_GRID:
        reg = HistGradientBoostingRegressor(max_iter=n, **_TREE_PARAMS).fit(train[cols], train["_y"])
        rmse = metrics.regression(valid["_y"].to_numpy(), reg.predict(valid[cols]))["rmse"]
        log.info("%s h%d: max_iter=%d rmse_valid=%.3f", target, horizon, n, rmse)
        if rmse < best_rmse:
            best_iter, best_rmse = n, rmse

    clf = HistGradientBoostingClassifier(max_iter=CLASSIFIER_ITER, **_TREE_PARAMS)
    clf.fit(train[cols], storm_train)
    threshold = _best_threshold(storm_valid, clf.predict_proba(valid[cols])[:, 1])

    final = pd.concat([train, valid])
    storm_final = is_storm(target, final["_y"].to_numpy())
    regressor = HistGradientBoostingRegressor(max_iter=best_iter, **_TREE_PARAMS).fit(final[cols], final["_y"])
    classifier = HistGradientBoostingClassifier(max_iter=CLASSIFIER_ITER, **_TREE_PARAMS)
    classifier.fit(final[cols], storm_final)

    # Avalia só onde o baseline também existe, para a comparação ser justa.
    base = persistence(test[cols], target)
    test = test[base.notna()]
    base = base[base.notna()].to_numpy()
    y_test = test["_y"].to_numpy()
    storm_test = is_storm(target, y_test)
    pred = regressor.predict(test[cols])
    prob = classifier.predict_proba(test[cols])[:, 1]
    climatology = float(storm_final.mean())

    report = {
        "target": target,
        "horizon_hours": horizon,
        "max_iter": best_iter,
        "storm_threshold": threshold,
        "train_rows": len(final),
        "test_rows": len(test),
        "test_period": [str(test.index.min()), str(test.index.max())],
        "regression": {
            "model": metrics.regression(y_test, pred),
            "persistence": metrics.regression(y_test, base),
        },
        "storm": {
            "classifier": metrics.events(storm_test, prob >= threshold),
            "regression": metrics.events(storm_test, is_storm(target, pred)),
            "persistence": metrics.events(storm_test, is_storm(target, base)),
            "brier": metrics.brier(storm_test, prob),
            "brier_climatology": metrics.brier(storm_test, np.full(len(prob), climatology)),
        },
    }

    if refit_all:
        # O modelo publicado aprende com tudo, inclusive o período de teste (ciclo solar atual).
        # As métricas acima continuam sendo as da avaliação honesta, sem esse período.
        everything = pd.concat([final, _slice(data, TEST_START)])
        storm_all = is_storm(target, everything["_y"].to_numpy())
        regressor = HistGradientBoostingRegressor(max_iter=best_iter, **_TREE_PARAMS)
        regressor.fit(everything[cols], everything["_y"])
        classifier = HistGradientBoostingClassifier(max_iter=CLASSIFIER_ITER, **_TREE_PARAMS)
        classifier.fit(everything[cols], storm_all)
        report["published_train_rows"] = len(everything)
    return Trained(target, horizon, regressor, classifier, threshold, report)


def bss(report: dict) -> float:
    """Brier skill score do classificador em relação à frequência histórica."""
    storm = report["storm"]
    return 1 - storm["brier"] / storm["brier_climatology"]


def compare(new: dict, old: dict, rmse_tolerance: float = 1.02, bss_tolerance: float = 0.02) -> tuple[bool, str]:
    """Trava de publicação: o modelo novo não pode ser pior que o atual na avaliação.

    Os dois relatórios usam a mesma divisão temporal, então diferenças vêm de mudanças
    de código, variáveis ou dados, e não do período avaliado.
    """
    previous = {(m["target"], m["horizon_hours"]): m for m in old.get("models", [])}
    ratios, deltas = [], []
    for m in new["models"]:
        o = previous.get((m["target"], m["horizon_hours"]))
        if o:
            ratios.append(m["regression"]["model"]["rmse"] / o["regression"]["model"]["rmse"])
            deltas.append(bss(m) - bss(o))
    if not ratios:
        return True, "nenhum modelo comparável na versão atual"
    ratio, delta = float(np.mean(ratios)), float(np.mean(deltas))
    ok = ratio <= rmse_tolerance and delta >= -bss_tolerance
    return ok, f"RMSE médio {ratio:.3f}x o atual, BSS médio {delta:+.3f}"


def train_all(
    hourly: pd.DataFrame, cmes: pd.DataFrame | None = None, refit_all: bool = True
) -> tuple[list[Trained], dict]:
    features = build_features(hourly, cmes)
    targets = build_targets(hourly, HORIZONS)
    trained = [
        train_one(features, targets[f"{target}_h{h}"], target, h, refit_all)
        for target, hs in HORIZONS.items()
        for h in hs
    ]
    summary = {
        "model_version": f"v{FEATURE_VERSION}-{datetime.now(UTC):%Y%m%d}",
        "feature_version": FEATURE_VERSION,
        "uses_cmes": cmes is not None,
        "features": list(features.columns),
        "trained_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "splits": {"train": TRAIN, "valid": VALID, "test_start": TEST_START},
        "published_model_trained_on": "todos os dados" if refit_all else "treino + validação",
        "threshold_criterion": "hss",
        "data_end": str(hourly.index.max()),
        "models": [t.report for t in trained],
    }
    return trained, summary
