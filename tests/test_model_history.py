import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from futurespace.api.app import create_app
from futurespace.ml.features import BASE_COLUMNS, build_features, feature_group
from futurespace.ml.serve import publish_bundle, record_rejected
from futurespace.ml.train import code_version, group_importance
from futurespace.model_index import MODELS_POINTER, read_index


def _model_report(target="kp", horizon=3, rmse=0.8, share=None):
    return {
        "target": target,
        "horizon_hours": horizon,
        "regression": {"model": {"rmse": rmse}, "persistence": {"rmse": 1.0}},
        "storm": {"brier": 0.02, "brier_climatology": 0.04, "classifier": {"hss": 0.5}},
        **({"importance": {"regression": {"share": share}, "storm": {"share": share}}} if share else {}),
    }


def _summary(version, trained_at, rmse=0.8, commit="abc123"):
    return {
        "model_version": version,
        "trained_at": trained_at,
        "features": ["speed", "kp_known"],
        "reproducibility": {"code": {"commit": commit}, "data": {"revision": "rev1"}},
        "models": [_model_report(rmse=rmse, share={"solar_wind": 0.7, "earth_state": 0.3})],
    }


@pytest.fixture
def bundle(tmp_path):
    path = tmp_path / "bundle.joblib"
    path.write_bytes(b"modelo")
    return path


def test_every_feature_has_a_group():
    idx = pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC")
    hourly = pd.DataFrame(np.ones((50, len(BASE_COLUMNS))), index=idx, columns=BASE_COLUMNS)
    cmes = pd.DataFrame({"time": [idx[2]], "width": [120], "speed": [800]})
    groups = {feature_group(c) for c in build_features(hourly, cmes).columns}
    assert groups == {"solar_wind", "earth_state", "recurrence", "cmes", "calendar"}


def test_group_importance_finds_the_variable_that_matters():
    rng = np.random.default_rng(0)
    n = 3000
    x = pd.DataFrame({"speed": rng.normal(size=n), "kp_known": rng.normal(size=n), "doy_sin": rng.normal(size=n)})
    y = 3 * x["speed"].to_numpy() + 0.1 * rng.normal(size=n)
    storm = x["speed"].to_numpy() > 1
    reg = HistGradientBoostingRegressor(max_iter=50).fit(x, y)
    clf = HistGradientBoostingClassifier(max_iter=50).fit(x, storm)

    out = group_importance(reg, clf, x, y, storm)
    assert out["regression"]["share"]["solar_wind"] > 0.9
    assert out["storm"]["share"]["solar_wind"] > 0.9
    assert sum(out["regression"]["share"].values()) == pytest.approx(1, abs=1e-3)
    assert out["regression"]["increase"]["solar_wind"] > 1  # RMSE sobe muito sem a velocidade


def test_code_version_uses_github_sha(monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "deadbeef")
    assert code_version()["commit"] == "deadbeef"
    assert code_version()["uncommitted_changes"] is False


def test_index_includes_version_published_before_it_existed(store):
    # Versão antiga: só current.json e report.json, sem a ficha de reprodutibilidade.
    store.write("models/v3-20261106/report.json", json.dumps({"model_version": "v3-20261106",
        "trained_at": "2026-11-06T21:00:00Z", "features": ["a"], "models": [_model_report()]}).encode())
    store.write(MODELS_POINTER, json.dumps({"model_version": "v3-20261106",
        "report": "models/v3-20261106/report.json"}).encode())

    index = read_index(store)
    assert index["current"] == "v3-20261106"
    [old] = index["versions"]
    assert old["status"] == "published" and old["code_commit"] is None
    assert old["models"][0]["bss"] == 0.5 and old["models"][0]["importance"] is None


def test_publish_and_reject_keep_full_history(store, bundle):
    publish_bundle(store, bundle, _summary("v3-20261201", "2026-12-01T05:00:00Z"))
    # Re-treino no mesmo dia recusado pela trava: não pode apagar o relatório publicado.
    key = record_rejected(store, _summary("v3-20261201", "2026-12-01T09:30:00Z", rmse=0.95), "RMSE pior")

    published = json.loads(store.read("models/v3-20261201/report.json"))
    assert published["models"][0]["regression"]["model"]["rmse"] == 0.8
    assert key == "models/rejected/v3-20261201-20261201T093000Z/report.json"

    index = read_index(store)
    assert index["current"] == "v3-20261201"
    assert [(v["id"], v["status"]) for v in index["versions"]] == [
        ("v3-20261201-20261201T093000Z", "rejected"),
        ("v3-20261201", "published"),
    ]
    assert index["versions"][0]["reason"] == "RMSE pior"
    assert index["versions"][1]["data_revision"] == "rev1"

    # Publicar de novo a mesma versão substitui a entrada em vez de duplicar.
    publish_bundle(store, bundle, _summary("v3-20261201", "2026-12-01T05:00:00Z", commit="def456"))
    published = [v for v in read_index(store)["versions"] if v["status"] == "published"]
    assert [v["code_commit"] for v in published] == ["def456"]


def test_models_endpoints_without_models(populated):
    assert TestClient(create_app(populated)).get("/v1/models").status_code == 404


def test_models_endpoints(populated, bundle):
    publish_bundle(populated, bundle, _summary("v3-20261201", "2026-12-01T05:00:00Z"))
    client = TestClient(create_app(populated))
    body = client.get("/v1/models").json()
    assert body["current"] == "v3-20261201"
    assert body["versions"][0]["models"][0]["importance"]["solar_wind"] == 0.7
    assert body["groups"]["solar_wind"]
    assert body["links"]["v3-20261201"] == "/v1/models/v3-20261201"

    report = client.get("/v1/models/v3-20261201").json()
    assert report["status"] == "published"
    assert report["reproducibility"]["code"]["commit"] == "abc123"
    assert client.get("/v1/models/v9-inexistente").status_code == 404
    assert client.get("/v1/models/..%2Fcurrent.json").status_code == 404
