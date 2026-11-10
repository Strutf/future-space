"""Métricas das previsões ao vivo a partir do dataset `verification`, em tabelas Markdown.

Para cada índice e horizonte:
- erro do valor (MAE e RMSE) do modelo e da persistência; para o Kp, também da previsão da
  NOAA SWPC, calculado só nos blocos em que as três existem (mesmo período, mesmos blocos);
- tempestade (Kp >= 5-, ou seja, 4,67, G1 da NOAA): acurácia, precisão e revocação do alerta
  do modelo, da previsão da NOAA (Kp previsto >= 4,67) e da persistência.

Usado por `futurespace evaluate`, que imprime as tabelas para colar no README.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from futurespace.ml.features import KP_STORM


def _mae(err: pd.Series) -> float:
    return float(err.abs().mean())


def _rmse(err: pd.Series) -> float:
    return float(np.sqrt((err**2).mean()))


def _events(observed: pd.Series, predicted: pd.Series) -> dict:
    observed, predicted = observed.astype(bool), predicted.astype(bool)
    hits = int((observed & predicted).sum())
    false_alarms = int((~observed & predicted).sum())
    misses = int((observed & ~predicted).sum())
    n = len(observed)
    return {
        "observed": int(observed.sum()),
        "hits": hits,
        "false_alarms": false_alarms,
        "misses": misses,
        "accuracy": (n - false_alarms - misses) / n if n else np.nan,
        "precision": hits / (hits + false_alarms) if hits + false_alarms else np.nan,
        "recall": hits / (hits + misses) if hits + misses else np.nan,
    }


def value_errors(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (target, horizon), g in rows.groupby(["target", "horizon_hours"]):
        if target == "kp":
            g = g[g["noaa"].notna() & g["persistence"].notna()]
        else:
            g = g[g["persistence"].notna()]
        if g.empty:
            continue
        row = {
            "target": target,
            "horizon_hours": int(horizon),
            "n": len(g),
            "mae_model": _mae(g["predicted"] - g["observed"]),
            "rmse_model": _rmse(g["predicted"] - g["observed"]),
            "mae_persistence": _mae(g["persistence"] - g["observed"]),
            "rmse_persistence": _rmse(g["persistence"] - g["observed"]),
            "mae_noaa": _mae(g["noaa"] - g["observed"]) if target == "kp" else np.nan,
            "rmse_noaa": _rmse(g["noaa"] - g["observed"]) if target == "kp" else np.nan,
        }
        out.append(row)
    return pd.DataFrame(out)


def storm_scores(rows: pd.DataFrame) -> pd.DataFrame:
    kp = rows[(rows["target"] == "kp") & rows["noaa"].notna() & rows["persistence"].notna()]
    out = []
    for horizon, g in kp.groupby("horizon_hours"):
        observed = g["observed"] >= KP_STORM
        for source, predicted in (
            ("modelo (alerta)", g["storm_alert"].astype(bool)),
            ("NOAA SWPC", g["noaa"] >= KP_STORM),
            ("persistência", g["persistence"] >= KP_STORM),
        ):
            out.append({"horizon_hours": int(horizon), "source": source, "n": len(g), **_events(observed, predicted)})
    return pd.DataFrame(out)


def _fmt(x, digits: int = 2) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}".replace(".", ",")


def _pct(x) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.0f}%"


def markdown(rows: pd.DataFrame) -> str:
    if rows.empty:
        return "Ainda não há previsões verificadas."
    start, end = rows["target_time"].min(), rows["target_time"].max()
    versions = ", ".join(sorted(rows["model_version"].dropna().unique()))
    lines = [
        (
            f"Período verificado: {start:%d/%m/%Y %H:%M} a {end:%d/%m/%Y %H:%M} UTC · "
            f"{len(rows)} previsões verificadas · modelos: {versions}."
        ),
        "",
        (
            "**Erro do valor previsto.** Para o Kp, só os blocos com previsão do modelo, da NOAA e "
            "persistência; a NOAA não prevê o Dst."
        ),
        "",
        "| Índice | Horizonte | n | MAE modelo | RMSE modelo | MAE NOAA | RMSE NOAA | MAE persistência | RMSE persistência |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in value_errors(rows).itertuples():
        lines.append(
            f"| {r.target.capitalize()} | +{r.horizon_hours} h | {r.n} | {_fmt(r.mae_model)} | {_fmt(r.rmse_model)} | "
            f"{_fmt(r.mae_noaa)} | {_fmt(r.rmse_noaa)} | {_fmt(r.mae_persistence)} | {_fmt(r.rmse_persistence)} |"
        )
    storms = storm_scores(rows)
    if not storms.empty:
        lines += [
            "",
            "**Tempestade geomagnética (Kp ≥ 5−, G1).** Mesmos blocos da tabela anterior.",
            "",
            "| Horizonte | Previsão | n | Tempestades observadas | Acertos | Alarmes falsos | Perdidas | Acurácia | Precisão | Revocação |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in storms.itertuples():
            lines.append(
                f"| +{r.horizon_hours} h | {r.source} | {r.n} | {r.observed} | {r.hits} | {r.false_alarms} | "
                f"{r.misses} | {_pct(r.accuracy)} | {_pct(r.precision)} | {_pct(r.recall)} |"
            )
    return "\n".join(lines)
