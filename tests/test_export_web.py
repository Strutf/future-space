import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from futurespace.ml.export_web import export_trees, predict_exported, sim_features

STATIC = Path(__file__).resolve().parent.parent / "src" / "futurespace" / "api" / "web" / "static"


def test_exported_trees_reproduce_sklearn_predictions():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 3))
    x[rng.random(x.shape) < 0.05] = np.nan  # valores faltantes também seguem o caminho certo
    y = 2 * np.nan_to_num(x[:, 0]) - np.nan_to_num(x[:, 2]) ** 2
    model = HistGradientBoostingRegressor(max_iter=50, max_leaf_nodes=15, random_state=0).fit(x, y)
    exported = json.loads(json.dumps(export_trees(model)))  # passa por JSON, como no site
    sample = x[:300]
    ours = [predict_exported(exported, list(row)) for row in sample]
    np.testing.assert_allclose(ours, model.predict(sample), atol=1e-4)


@pytest.mark.parametrize("target", ["kp", "dst"])
def test_published_simulator_model_is_physically_monotonic(target):
    sim = json.loads((STATIC / "sim-model.json").read_text())
    model = sim["models"][target]
    sign = 1 if target == "kp" else -1

    def f(v, n, bz):
        return sign * predict_exported(model, sim_features(v, n, bz))

    for n in (5, 20):
        for bz in (-30, -10, 0, 10):
            values = [f(v, n, bz) for v in range(300, 2600, 100)]
            assert values == sorted(values), (n, bz)  # mais velocidade nunca acalma a tempestade
    assert f(800, 15, -20) > f(800, 15, 20)  # ímã para o sul é pior que para o norte


def test_replays_cover_the_famous_storms():
    replays = json.loads((STATIC / "replays.json").read_text())["events"]
    assert set(replays) == {"may2024", "stpatrick2015", "halloween2003", "quebec1989"}
    may = replays["may2024"]
    assert not may["in_training"]  # a IA da viagem nunca viu maio de 2024
    assert min(h["dst"] for h in may["hours"] if h["dst"] is not None) <= -400
