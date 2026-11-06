import pytest
from fastapi.testclient import TestClient

from futurespace.api.app import create_app


@pytest.fixture
def client(populated):
    return TestClient(create_app(populated))


def test_list_datasets_includes_coverage(client):
    body = client.get("/v1/datasets").json()
    kp = next(d for d in body["datasets"] if d["id"] == "kp")
    assert kp["coverage"]["end"] == "2026-11-06T00:00:00Z"


def test_data_range_filters_and_selects_fields(client):
    resp = client.get(
        "/v1/datasets/kp/data",
        params={"start": "2026-11-02T00:00Z", "end": "2026-11-02T09:00Z", "fields": "kp"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["fields"] == ["time", "kp"]
    assert [r["time"] for r in body["data"]] == [
        "2026-11-02T00:00:00Z",
        "2026-11-02T03:00:00Z",
        "2026-11-02T06:00:00Z",
        "2026-11-02T09:00:00Z",
    ]
    assert "max-age" in resp.headers["cache-control"]


def test_data_spans_daily_partitions(client):
    body = client.get(
        "/v1/datasets/solar-wind-plasma/data",
        params={"start": "2026-11-05T23:59Z", "end": "2026-11-06T00:00Z"},
    ).json()
    assert body["count"] == 4  # 2 minutos x 2 sondas, em duas partições diárias


def test_default_window_ends_at_last_available_data(client):
    body = client.get("/v1/datasets/kp/data").json()
    assert body["end"] == "2026-11-06T00:00:00Z"
    assert body["start"] == "2026-10-30T00:00:00Z"


def test_csv_format(client):
    resp = client.get("/v1/datasets/kp/data", params={"start": "2026-11-01", "end": "2026-11-01T03:00Z"})
    csv = client.get(
        "/v1/datasets/kp/data", params={"start": "2026-11-01", "end": "2026-11-01T03:00Z", "format": "csv"}
    )
    assert csv.headers["content-type"].startswith("text/csv")
    assert csv.text.splitlines()[0] == "time,kp,a_running,station_count"
    assert len(csv.text.splitlines()) == resp.json()["count"] + 1


@pytest.mark.parametrize(
    "dataset, params, message",
    [
        ("solar-wind-plasma", {"start": "2026-01-01", "end": "2026-11-01"}, "intervalo máximo"),
        ("kp", {"start": "ontem"}, "inválido"),
        ("kp", {"start": "2026-11-02", "end": "2026-11-01"}, "anterior"),
        ("kp", {"fields": "kp,xyz"}, "desconhecidos"),
    ],
)
def test_validation_errors(client, dataset, params, message):
    resp = client.get(f"/v1/datasets/{dataset}/data", params=params)
    assert resp.status_code == 400
    assert message in resp.json()["detail"]


def test_unknown_and_empty_datasets(client):
    assert client.get("/v1/datasets/nada/data").status_code == 404
    assert client.get("/v1/datasets/dst/latest").status_code == 404


def test_latest(client):
    body = client.get("/v1/datasets/kp/latest").json()
    assert body["data"][0]["time"] == "2026-11-06T00:00:00Z"
