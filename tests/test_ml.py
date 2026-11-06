import numpy as np
import pandas as pd

from futurespace.ml import metrics
from futurespace.ml.features import BASE_COLUMNS, _kp_known, build_features, build_targets


def synthetic_hourly(hours: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-05-01", periods=hours, freq="h", tz="UTC")
    return pd.DataFrame(rng.normal(size=(hours, len(BASE_COLUMNS))), index=idx, columns=BASE_COLUMNS)


def test_features_never_use_future_values():
    hourly = synthetic_hourly()
    cut = 60
    changed = hourly.copy()
    changed.iloc[cut + 1 :] = 999.0  # muda tudo depois do instante `cut`

    before = build_features(hourly).iloc[: cut + 1]
    after = build_features(changed).iloc[: cut + 1]
    pd.testing.assert_frame_equal(before, after)


def test_kp_known_only_uses_completed_blocks():
    idx = pd.date_range("2024-05-01T00:00", periods=9, freq="h", tz="UTC")
    kp = pd.Series([1, 1, 1, 2, 2, 2, 3, 3, 3], index=idx, dtype=float)
    known = _kp_known(kp)
    assert np.isnan(known.iloc[1])  # 01h: bloco 00-03h ainda não terminou
    assert known.iloc[2] == 1  # fim da hora 02h = 03h: bloco 00-03h encerrado
    assert known.iloc[4] == 1
    assert known.iloc[5] == 2  # fim da hora 05h = 06h: bloco 03-06h encerrado


def test_targets_are_future_values():
    hourly = synthetic_hourly(10)
    t = build_targets(hourly, {"dst": [1, 3]})
    assert t["dst_h1"].iloc[0] == hourly["dst"].iloc[1]
    assert t["dst_h3"].iloc[0] == hourly["dst"].iloc[3]
    assert t["dst_h3"].iloc[-3:].isna().all()


def test_event_metrics():
    observed = np.array([1, 1, 0, 0, 0, 0], dtype=bool)
    predicted = np.array([1, 0, 1, 0, 0, 0], dtype=bool)
    m = metrics.events(observed, predicted)
    assert (m["hits"], m["misses"], m["false_alarms"]) == (1, 1, 1)
    assert m["pod"] == 0.5
    assert m["tss"] == 0.5 - 0.25


def test_recurrence_features_look_one_rotation_back():
    from futurespace.ml.features import ROTATION_HOURS

    hourly = synthetic_hourly(ROTATION_HOURS + 100)
    f = build_features(hourly)
    t = ROTATION_HOURS + 50
    assert f["rec27_speed_now"].iloc[t] == hourly["speed"].iloc[t - ROTATION_HOURS]
    window = hourly["dst"].iloc[t - ROTATION_HOURS + 1 : t - ROTATION_HOURS + 25]
    assert f["rec27_dst_min24"].iloc[t] == window.min()
