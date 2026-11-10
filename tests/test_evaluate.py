import numpy as np
import pandas as pd
import pytest

from futurespace.ml.evaluate import markdown, storm_scores, value_errors


def _rows():
    times = pd.date_range("2026-11-07", periods=4, freq="3h", tz="UTC")
    kp = pd.DataFrame(
        {
            "time": times - pd.Timedelta(hours=3),
            "target": "kp",
            "horizon_hours": 3,
            "target_time": times,
            "predicted": [2.0, 3.0, 5.0, 4.0],
            "observed": [2.0, 5.33, 5.0, 3.0],
            "persistence": [2.0, 2.0, 5.33, 5.0],
            "noaa": [3.0, 3.67, 4.67, np.nan],  # último bloco sem NOAA: fica de fora do Kp
            "storm_alert": [False, False, True, False],
            "model_version": "v3",
        }
    )
    dst = pd.DataFrame(
        {
            "time": times,
            "target": "dst",
            "horizon_hours": 1,
            "target_time": times + pd.Timedelta(hours=1),
            "predicted": [-10.0, -20.0, -30.0, -40.0],
            "observed": [-12.0, -20.0, -36.0, -40.0],
            "persistence": [-8.0, -12.0, -20.0, -36.0],
            "noaa": np.nan,
            "storm_alert": False,
            "model_version": "v3",
        }
    )
    return pd.concat([kp, dst], ignore_index=True)


def test_value_errors_use_the_same_blocks_for_model_and_noaa():
    v = value_errors(_rows()).set_index("target")
    assert v.loc["kp", "n"] == 3
    assert v.loc["kp", "mae_model"] == pytest.approx((0 + 2.33 + 0) / 3)
    assert v.loc["kp", "rmse_noaa"] == pytest.approx(np.sqrt((1 + 1.66**2 + 0.33**2) / 3))
    assert v.loc["dst", "mae_model"] == pytest.approx((2 + 0 + 6 + 0) / 4)
    assert np.isnan(v.loc["dst", "mae_noaa"])


def test_storm_scores_for_model_noaa_and_persistence():
    s = storm_scores(_rows()).set_index("source")
    # Observado: tempestade nos blocos 2 (5,33) e 3 (5,0).
    assert s.loc["modelo (alerta)", ["hits", "misses", "false_alarms"]].tolist() == [1, 1, 0]
    assert s.loc["modelo (alerta)", "precision"] == 1 and s.loc["modelo (alerta)", "recall"] == 0.5
    assert s.loc["NOAA SWPC", ["hits", "misses"]].tolist() == [1, 1]
    assert s.loc["persistência", "accuracy"] == pytest.approx(2 / 3)


def test_markdown_reports_period_and_counts():
    text = markdown(_rows())
    assert "8 previsões verificadas" in text
    assert "| Kp | +3 h | 3 |" in text
    assert markdown(pd.DataFrame()) == "Ainda não há previsões verificadas."
