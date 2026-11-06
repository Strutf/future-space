from datetime import UTC, datetime

import pandas as pd

from futurespace import catalog
from futurespace.ml.train import compare
from futurespace.ml.verify import summarize, verify
from futurespace.pipeline.writer import write_dataset

NOW = datetime(2026, 11, 7, 12, tzinfo=UTC)


def _write(store, dataset_id, rows):
    write_dataset(store, catalog.get(dataset_id), pd.DataFrame(rows), now=NOW)


def test_verify_matches_observations_and_baselines(store):
    issued = "2026-11-06T21:00Z"  # dados até o fim da hora 20h
    _write(
        store,
        "predictions",
        [
            {"time": issued, "target": "dst", "horizon_hours": 1, "target_time": "2026-11-06T21:00Z",
             "value": -5.0, "storm_probability": 0.01, "storm_alert": False, "model_version": "v"},
            {"time": issued, "target": "kp", "horizon_hours": 3, "target_time": "2026-11-06T21:00Z",
             "value": 2.5, "storm_probability": 0.6, "storm_alert": True, "model_version": "v"},
        ],
    )
    hours = pd.date_range("2026-11-06", periods=24, freq="h", tz="UTC")
    _write(store, "dst", {"time": hours, "dst": [-float(h.hour) for h in hours]})
    _write(store, "kp", {"time": ["2026-11-06T18:00Z", "2026-11-06T21:00Z"], "kp": [3.0, 4.0]})
    _write(store, "kp-gfz", {"time": ["2026-11-06T21:00Z"], "kp": [4.333], "status": ["now"]})
    _write(
        store,
        "kp-forecast",
        {
            "time": ["2026-11-06T19:00Z", "2026-11-06T22:00Z"],  # a segunda é posterior à nossa
            "target_time": ["2026-11-06T21:00Z", "2026-11-06T21:00Z"],
            "kp": [5.0, 4.0],
        },
    )

    rows = verify(store, now=NOW).set_index("target")
    dst, kp = rows.loc["dst"], rows.loc["kp"]
    assert (dst["observed"], dst["persistence"], dst["error"]) == (-21.0, -20.0, 16.0)
    assert kp["observed"] == 4.333  # GFZ tem prioridade sobre a estimativa da NOAA
    assert kp["persistence"] == 3.0  # último bloco encerrado na emissão: 18h-21h
    assert kp["noaa"] == 5.0  # só previsões da NOAA emitidas até a nossa
    assert not kp["storm_observed"] and kp["storm_alert"]

    summary = summarize(rows.reset_index(), NOW)
    kp_summary = next(g for g in summary["windows"]["7d"] if g["target"] == "kp")
    assert kp_summary["false_alarms"] == 1
    assert kp_summary["mae_noaa"] == round(abs(5.0 - 4.333), 3)


def test_verify_skips_targets_not_yet_observed(store):
    _write(
        store,
        "predictions",
        [{"time": "2026-11-07T11:00Z", "target": "kp", "horizon_hours": 3,
          "target_time": "2026-11-07T12:00Z", "value": 2.0, "storm_probability": 0.0,
          "storm_alert": False, "model_version": "v"}],
    )
    _write(store, "kp", {"time": ["2026-11-07T12:00Z"], "kp": [2.0]})
    assert verify(store, now=NOW).empty  # bloco 12h-15h ainda não terminou


def _report(rmse: float, brier: float) -> dict:
    model = {
        "target": "kp",
        "horizon_hours": 3,
        "regression": {"model": {"rmse": rmse}},
        "storm": {"brier": brier, "brier_climatology": 0.1},
    }
    return {"models": [model]}


def test_publication_gate():
    assert compare(_report(0.75, 0.05), _report(0.76, 0.05))[0]
    assert not compare(_report(0.90, 0.05), _report(0.75, 0.05))[0]  # RMSE 20% pior
    assert not compare(_report(0.75, 0.08), _report(0.75, 0.05))[0]  # probabilidade pior
    assert compare(_report(0.75, 0.05), {"models": []})[0]
