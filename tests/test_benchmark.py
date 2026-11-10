import numpy as np
import pandas as pd

from futurespace.ml.benchmark import compare, markdown, model_forecasts, noaa_vs_observed


class _Constant:
    """Modelo falso: o regressor devolve sempre o mesmo valor e o classificador, a mesma probabilidade."""

    def __init__(self, value):
        self.value = value

    def predict(self, x):
        return np.full(len(x), self.value)

    def predict_proba(self, x):
        return np.column_stack([np.full(len(x), 1 - self.value), np.full(len(x), self.value)])


def _features():
    idx = pd.date_range("2023-01-01", periods=72, freq="h", tz="UTC")
    f = pd.DataFrame({"speed": 400.0, "bz_gsm": -2.0, "kp_known": 2.0}, index=idx)
    f.loc[idx[11], "speed"] = np.nan  # sem vento solar nessa hora: sem previsão
    return f


def _bundle():
    return {
        "features": ["speed", "bz_gsm", "kp_known"],
        "models": {
            "kp_h6": {"regressor": _Constant(3.0), "classifier": _Constant(0.3), "threshold": 0.2},
            "dst_h1": {"regressor": _Constant(-20.0), "classifier": _Constant(0.1), "threshold": 0.5},
        },
    }


def test_model_issues_one_hour_before_each_noaa_edition_and_only_kp():
    issues = pd.Series(pd.to_datetime(["2023-01-01T12:30Z", "2023-01-02T00:30Z"]))
    out = model_forecasts(_bundle(), _features(), issues)
    # 12:30 → hora 11 (sem vento solar, descartada); 00:30 → hora 23 do dia 1.
    assert out["issue_hour"].tolist() == [pd.Timestamp("2023-01-01T23:00Z")]
    assert out["horizon_hours"].tolist() == [6]
    assert out["target_time"].tolist() == [pd.Timestamp("2023-01-02T03:00Z")]  # 23h + 6 h = 05h → bloco das 03h
    assert out["alert"].tolist() == [True]


def test_compare_matches_same_block_and_same_issue():
    kp = pd.Series([5.0, 2.0], index=pd.to_datetime(["2023-01-02T03:00Z", "2023-01-02T06:00Z"]))
    noaa = pd.DataFrame(
        {
            "time": pd.to_datetime(["2023-01-02T00:30Z"] * 2),
            "target_time": pd.to_datetime(["2023-01-02T03:00Z", "2023-01-02T06:00Z"]),
            "kp": [4.67, 3.0],
        }
    )
    noaa = noaa_vs_observed(noaa, kp)
    assert noaa["lead_hours"].tolist() == [2.5, 5.5]
    both = compare(noaa, model_forecasts(_bundle(), _features(), noaa["time"]))
    assert len(both) == 1
    row = both.iloc[0]
    assert (row["observed"], row["kp"], row["model"]) == (5.0, 4.67, 3.0)
    assert "**NOAA SWPC × Kp observado" in markdown(noaa, both)
