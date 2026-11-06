from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from futurespace import catalog
from futurespace.ml.features import (
    AU_KM,
    BASE_COLUMNS,
    ROTATION_HOURS,
    build_features,
    cme_features,
)
from futurespace.ml.serve import predict, publish_bundle
from futurespace.pipeline.connectors.cactus import parse_catalog
from futurespace.pipeline.writer import write_dataset

CACTUS_SAMPLE = """\
:Issued: Fri Nov  6 19:56:35 2026
# CME |          t0    | dt0| pa | da |  v  |  dv | minv| maxv| halo?
  0002|2024/05/09 09:24| 02 | 254| 166| 1250| 0300| 0400| 1900|  II
  0001|2024/05/08 21:49| 01 | 245| 360| 0563| 0100| 0300| 0800|  IV
  0000|2024/05/08 20:00| 01 | 100| 010| 0300| 0050| 0200| 0400|

# Flow|          t0    | dt0| pa | da |  v  |  dv | minv| maxv| halo?
  0001|2024/05/09 06:36| 02 | 031| 008| 0238| 0116| 0107| 0411|
"""


def test_parse_cactus_ignores_flows():
    df = parse_catalog(CACTUS_SAMPLE, "quicklook")
    assert len(df) == 3
    assert df["halo"].tolist() == [2, 4, 0]
    assert str(df["time"].iloc[0]) == "2024-05-09 09:24:00+00:00"
    assert df["width"].iloc[1] == 360


def test_cme_features_respect_detection_latency():
    cmes = pd.DataFrame(
        {"time": [pd.Timestamp("2024-05-01T00:00Z")], "width": [120], "speed": [1000]}
    )
    index = pd.date_range("2024-04-30T20:00", "2024-05-03T00:00", freq="h", tz="UTC")
    f = cme_features(cmes, index)
    assert f.loc["2024-04-30T23:00Z"].isna().all()  # antes do início do catálogo
    assert f.loc["2024-05-01T05:00Z", "cme_wide_speed_max24"] == 0  # ainda não detectada
    assert f.loc["2024-05-01T06:00Z", "cme_wide_speed_max24"] == 1000  # t0 + 6 h
    transit = AU_KM / 1000 / 3600
    assert f.loc["2024-05-01T06:00Z", "cme_eta_hours"] == pytest.approx(transit - 6)


def _populate_realtime(store, now: datetime):
    minutes = pd.date_range(now - pd.Timedelta(hours=50), now, freq="1min", tz="UTC")
    plasma = pd.DataFrame(
        {"time": minutes, "source": "SOLAR1", "active": True, "speed": 450.0, "density": 5.0, "temperature": 1e5}
    )
    mag = pd.DataFrame(
        {"time": minutes, "source": "SOLAR1", "active": True, "bt": 6.0, "by_gsm": 2.0, "bz_gsm": -4.0}
    )
    hours = pd.date_range(now - pd.Timedelta(hours=50), now, freq="h", tz="UTC").floor("h")
    write_dataset(store, catalog.get("solar-wind-plasma"), plasma, now=now)
    write_dataset(store, catalog.get("solar-wind-mag"), mag, now=now)
    write_dataset(store, catalog.get("dst"), pd.DataFrame({"time": hours, "dst": -20.0}), now=now)
    blocks = pd.date_range(hours[0].floor("3h"), now, freq="3h", tz="UTC")
    write_dataset(store, catalog.get("kp"), pd.DataFrame({"time": blocks, "kp": 3.0}), now=now)
    cmes = pd.DataFrame(
        {"time": [now - pd.Timedelta(days=1)], "pa": [90], "width": [200], "speed": [900], "halo": [3]}
    )
    write_dataset(store, catalog.get("cme-cactus"), cmes, now=now)


def _tiny_bundle(tmp_path, store):
    # Mais de uma rotação solar, para as variáveis de recorrência não ficarem vazias.
    hours = ROTATION_HOURS + 300
    idx = pd.date_range("2024-01-01", periods=hours, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    hourly = pd.DataFrame(rng.normal(size=(hours, len(BASE_COLUMNS))), index=idx, columns=BASE_COLUMNS)
    cmes = pd.DataFrame({"time": [idx[10]], "width": [150], "speed": [800]})
    x = build_features(hourly, cmes)
    y = rng.normal(size=len(x))
    models = {
        name: {
            "regressor": HistGradientBoostingRegressor(max_iter=5).fit(x, y),
            "classifier": HistGradientBoostingClassifier(max_iter=5).fit(x, y > 0),
            "threshold": 0.5,
        }
        for name in ("dst_h1", "kp_h3")
    }
    import joblib

    path = tmp_path / "bundle.joblib"
    joblib.dump({"model_version": "vteste", "features": list(x.columns), "models": models}, path)
    publish_bundle(store, path, {"model_version": "vteste"})


def test_predict_end_to_end(tmp_path, store):
    now = datetime(2026, 11, 6, 20, 7, tzinfo=UTC)
    _populate_realtime(store, now)
    _tiny_bundle(tmp_path, store)

    rows = predict(store, now=now)
    assert set(zip(rows["target"], rows["horizon_hours"], strict=True)) == {("dst", 1), ("kp", 3)}
    assert (rows["time"] == pd.Timestamp("2026-11-06T20:00Z")).all()  # fim da hora 19h
    kp = rows[rows["target"] == "kp"].iloc[0]
    assert kp["target_time"] == pd.Timestamp("2026-11-06T21:00Z")  # bloco que contém 22h
    assert 0 <= kp["storm_probability"] <= 1
    assert rows["model_version"].eq("vteste").all()


def test_predict_refuses_without_solar_wind(tmp_path, store):
    _tiny_bundle(tmp_path, store)
    with pytest.raises(RuntimeError, match="sem vento solar"):
        predict(store, now=datetime(2026, 11, 6, 20, 7, tzinfo=UTC))


def test_realtime_cmes_superseded_by_quicklook_are_dropped():
    from futurespace.pipeline.connectors.cactus import prefer_curated

    df = pd.DataFrame(
        {
            "time": pd.to_datetime(["2026-11-01T19:00Z", "2026-11-01T19:00Z", "2026-11-05T19:12Z"]),
            "catalog": ["quicklook", "realtime", "realtime"],
        }
    )
    kept = prefer_curated(df)
    assert kept["catalog"].tolist() == ["quicklook", "realtime"]
    assert kept["time"].iloc[-1] == pd.Timestamp("2026-11-05T19:12Z")
