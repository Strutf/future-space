"""Ajusta e avalia a probabilidade de aurora (e de céu limpo) com a base do build_dataset.

Aurora: P(aurora na câmera | Kp, latitude magnética, hora local magnética), em momentos
escuros e de céu limpo. Validação por noite (5 dobras). Compara com a pontuação v0 do
backend (logística de Kp − Kp necessário) e com a climatologia.

Nuvens: P(céu limpo | cobertura de nuvens ERA5), comparada com a regra v0 (1 − nuvens/100).

Saída: <pasta>/report.json (métricas, diagramas de confiabilidade, coeficientes).
Uso: python fit_probability.py <pasta_base>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from futurespace.aurora.visibility import KP_SCALE, kp_needed


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return [
        {"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int((idx == b).sum()),
         "forecast": float(p[idx == b].mean()), "observed": float(y[idx == b].mean())}
        for b in range(bins) if (idx == b).any()
    ]


def scores(y: np.ndarray, p: np.ndarray, climatology: float) -> dict:
    brier = float(np.mean((p - y) ** 2))
    ref = float(np.mean((climatology - y) ** 2))
    return {"brier": brier, "bss": 1 - brier / ref, "reliability": reliability(y, p)}


def aurora_features(df: pd.DataFrame) -> pd.DataFrame:
    angle = 2 * np.pi * df["mlt"] / 24
    m = df["mag_lat"].abs()
    return pd.DataFrame({
        "kp": df["kp"], "mlat": m, "kp_mlat": df["kp"] * m,
        "mlt_sin": np.sin(angle), "mlt_cos": np.cos(angle), "kp_mlt_cos": df["kp"] * np.cos(angle),
    })


def cross_validate(model_factory, x: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(x, y, groups):
        model = model_factory().fit(x.iloc[tr], y[tr])
        p[te] = model.predict_proba(x.iloc[te])[:, 1]
    return p


def fit_aurora(df: pd.DataFrame) -> dict:
    df = df[df["label"].isin(["aurora", "clear"]) & df["kp"].notna()].reset_index(drop=True)
    y = (df["label"] == "aurora").to_numpy(dtype=float)
    groups = df["night"].to_numpy()
    x = aurora_features(df)
    climatology = float(y.mean())

    logistic = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    boosting = lambda: HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
        monotonic_cst=[1, 0, 0, 0, 0, 0], random_state=0,
    )
    need = np.array([kp_needed(m) for m in df["mag_lat"].abs()])
    with np.errstate(over="ignore", invalid="ignore"):
        v0 = np.where(np.isfinite(need), 1 / (1 + np.exp(-(df["kp"] - need) / KP_SCALE)), 0.0)

    final = logistic().fit(x, y)
    scaler, lr = final.named_steps["standardscaler"], final.named_steps["logisticregression"]
    return {
        "samples": len(df),
        "nights": int(pd.Series(groups).nunique()),
        "sites": df["site"].value_counts().to_dict(),
        "aurora_fraction": climatology,
        "excluded_other": None,
        "v0_score": scores(y, v0, climatology),
        "logistic_cv": scores(y, cross_validate(logistic, x, y, groups), climatology),
        "boosting_cv": scores(y, cross_validate(boosting, x, y, groups), climatology),
        "logistic_model": {
            "features": list(x.columns),
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "coef": lr.coef_[0].tolist(),
            "intercept": float(lr.intercept_[0]),
        },
        "by_kp": df.assign(y=y).groupby(df["kp"].round()).agg(n=("y", "size"), aurora=("y", "mean")).reset_index().to_dict("records"),
    }


def fit_clouds(df: pd.DataFrame) -> dict:
    df = df[df["cloud_cover"].notna()].reset_index(drop=True)
    y = (df["label"] == "not_cloudy").to_numpy(dtype=float)
    groups = df["night"].to_numpy()
    x = df[["cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high"]]
    climatology = float(y.mean())
    model = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    by_cover = df.assign(y=y).groupby((df["cloud_cover"] // 10) * 10).agg(n=("y", "size"), clear=("y", "mean"))
    return {
        "samples": len(df),
        "clear_fraction": climatology,
        "v0_rule": scores(y, np.clip(1 - df["cloud_cover"].to_numpy() / 100, 0, 1), climatology),
        "logistic_cv": scores(y, cross_validate(model, x, y, groups), climatology),
        "by_cloud_cover": by_cover.reset_index().to_dict("records"),
    }


def main(folder: str) -> None:
    base = Path(folder)
    aurora = pd.read_parquet(base / "aurora_samples.parquet")
    report = {"aurora": fit_aurora(aurora), "clouds": fit_clouds(pd.read_parquet(base / "cloud_samples.parquet"))}
    report["aurora"]["excluded_other"] = int((aurora["label"] == "other").sum())
    (base / "report.json").write_text(json.dumps(report, indent=2, default=float))
    a, c = report["aurora"], report["clouds"]
    print(f"aurora: {a['samples']} amostras, {a['nights']} noites, {a['aurora_fraction']:.1%} com aurora")
    for k in ("v0_score", "logistic_cv", "boosting_cv"):
        print(f"  {k:<12} Brier {a[k]['brier']:.4f}  BSS {a[k]['bss']:+.3f}")
    print(f"nuvens: {c['samples']} horários, {c['clear_fraction']:.1%} limpos")
    for k in ("v0_rule", "logistic_cv"):
        print(f"  {k:<12} Brier {c[k]['brier']:.4f}  BSS {c[k]['bss']:+.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
