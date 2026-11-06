"""Métricas de regressão e de detecção de tempestade (eventos raros)."""

from __future__ import annotations

import numpy as np


def regression(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - y
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "r": float(np.corrcoef(y, pred)[0, 1]),
    }


def brier(observed: np.ndarray, probability: np.ndarray) -> float:
    """Erro quadrático médio da probabilidade (0 é perfeito)."""
    return float(np.mean((probability - observed.astype(float)) ** 2))


def events(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    """Tabela de contingência para eventos binários (ex.: Kp >= 5).

    TSS (true skill statistic) = POD - POFD: não é inflado pela raridade dos eventos,
    ao contrário da acurácia. 1 é perfeito, 0 é sem habilidade.
    """
    o = observed.astype(bool)
    p = predicted.astype(bool)
    hits = int(np.sum(o & p))
    misses = int(np.sum(o & ~p))
    false_alarms = int(np.sum(~o & p))
    correct_neg = int(np.sum(~o & ~p))
    pod = hits / (hits + misses) if hits + misses else float("nan")
    pofd = false_alarms / (false_alarms + correct_neg) if false_alarms + correct_neg else float("nan")
    far = false_alarms / (hits + false_alarms) if hits + false_alarms else float("nan")
    n = hits + misses + false_alarms + correct_neg
    expected = ((hits + misses) * (hits + false_alarms) + (correct_neg + misses) * (correct_neg + false_alarms)) / n
    hss = (hits + correct_neg - expected) / (n - expected) if n != expected else float("nan")
    return {
        "events": hits + misses,
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "pod": pod,
        "far": far,
        "tss": pod - pofd,
        "hss": hss,
    }
