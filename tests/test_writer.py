import json

import pandas as pd

from futurespace import catalog
from futurespace.pipeline.writer import read_parquet, write_dataset

from .conftest import NOW

KP = catalog.get("kp-gfz")


def test_merge_is_idempotent_and_keeps_latest_value(store):
    first = pd.DataFrame(
        {"time": ["2026-11-01T00:00:00Z", "2026-11-01T03:00:00Z"], "kp": [2.0, 3.0], "status": "pre"}
    )
    write_dataset(store, KP, first, now=NOW)
    revised = pd.DataFrame(
        {"time": ["2026-11-01T03:00:00Z", "2026-11-01T06:00:00Z"], "kp": [3.333, 4.0], "status": "def"}
    )
    write_dataset(store, KP, revised, now=NOW)
    write_dataset(store, KP, revised, now=NOW)

    df = read_parquet(store, "parquet/kp-gfz/2026.parquet")
    assert df["kp"].tolist() == [2.0, 3.333, 4.0]
    assert df["status"].tolist() == ["pre", "def", "def"]


def test_manifest_and_json_partitions(store):
    df = pd.DataFrame({"time": ["2025-12-31T21:00:00Z", "2026-01-01T00:00:00Z"], "kp": [1.0, 2.0]})
    write_dataset(store, KP, df, now=NOW)

    manifest = json.loads(store.read("json/kp-gfz/manifest.json"))
    assert manifest["partitions"] == ["2025", "2026"]
    assert manifest["start"] == "2025-12-31T21:00:00Z"
    assert manifest["end"] == "2026-01-01T00:00:00Z"

    records = json.loads(store.read("json/kp-gfz/2026.json"))
    assert records == [{"time": "2026-01-01T00:00:00Z", "kp": 2.0, "status": None}]


def test_latest_keeps_one_row_per_source(populated):
    latest = json.loads(populated.read("json/solar-wind-plasma/latest.json"))
    assert {r["source"] for r in latest["data"]} == {"SOLAR1", "ACE"}
    assert all(r["time"] == "2026-11-06T23:59:00Z" for r in latest["data"])


def test_coerce_drops_rows_without_time(store):
    df = pd.DataFrame({"time": ["2026-11-01T00:00:00Z", None, "lixo"], "kp": [1, 2, 3]})
    result = write_dataset(store, KP, df, now=NOW)
    assert result.rows == 1
