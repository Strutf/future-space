import httpx
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


@pytest.mark.parametrize("value", ["9999-01-01", "0001-01-01"])
def test_dates_outside_supported_range_are_rejected(client, value):
    resp = client.get("/v1/datasets/kp/data", params={"start": value, "end": value})
    assert resp.status_code == 400
    assert "fora do intervalo" in resp.json()["detail"]


def test_security_headers(client):
    headers = client.get("/v1/datasets/kp/latest").headers
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["x-frame-options"] == "DENY"


def test_storage_outage_returns_503_without_internal_details():
    from futurespace.storage import HttpStorage

    def offline(request):
        raise httpx.ConnectError("sem rede", request=request)

    storage = HttpStorage("https://exemplo.invalid/segredo")
    storage.client = httpx.Client(transport=httpx.MockTransport(offline))
    resp = TestClient(create_app(storage)).get("/v1/datasets/kp/latest")
    assert resp.status_code == 503
    assert "exemplo.invalid" not in resp.text


@pytest.mark.parametrize("path, marker, lang", [
    ("/", "O que nós fazemos", "pt-BR"),
    ("/dashboard", "/v1/compare/kp", "pt-BR"),
    ("/sobre", "Como acontece uma tempestade solar", "pt-BR"),
    ("/interativo", "Em breve", "pt-BR"),
    ("/en", "What we do", "en"),
    ("/en/dashboard", "/v1/compare/kp", "en"),
    ("/en/about", "How a solar storm happens", "en"),
    ("/en/interactive", "Coming soon", "en"),
])
def test_site_pages(client, path, marker, lang):
    page = client.get(path)
    assert page.status_code == 200
    assert marker in page.text
    assert f'<html lang="{lang}"' in page.text
    assert 'href="/static/chalk.css"' in page.text
    assert "{{" not in page.text  # molde e traduções todos preenchidos


@pytest.mark.parametrize("path, other", [("/sobre", "/en/about"), ("/en/about", "/sobre"), ("/", "/en")])
def test_language_switch_points_to_same_page(client, path, other):
    assert f'class="lang-switch" href="{other}"' in client.get(path).text


def test_english_pages_have_no_portuguese_leftovers(client):
    for path in ("/en", "/en/dashboard", "/en/about", "/en/interactive"):
        text = client.get(path).text.lower()
        for word in ("tempestade", "previsões", "ontem", "chegada", "agora", "a gente"):
            assert word not in text, (path, word)


def test_static_files_and_traversal(client):
    css = client.get("/static/chalk.css")
    assert css.status_code == 200 and css.headers["content-type"].startswith("text/css")
    assert client.get("/static/..%2Fbase.html").status_code == 404
    assert client.get("/static/nada.js").status_code == 404


def test_compare_endpoint_without_predictions(client):
    body = client.get("/v1/compare/kp", params={"date": "2026-11-06"}).json()
    assert [d["summary"]["date"] for d in body["days"]] == ["2026-11-05", "2026-11-06"]
    assert all(b["ia"] is None for d in body["days"] for b in d["blocks"])


def test_unknown_query_params_are_rejected(client):
    resp = client.get("/v1/datasets/kp/latest", params={"cachebuster": "123"})
    assert resp.status_code == 400
    assert "cachebuster" in resp.json()["detail"]
    assert client.get("/v1/datasets/kp/data", params={"fields": "kp"}).status_code == 200


def test_rate_limit_per_client(populated):
    limited = TestClient(create_app(populated, rate_limit_per_minute=3))
    for _ in range(3):
        assert limited.get("/health", headers={"x-real-ip": "1.1.1.1"}).status_code == 200
    blocked = limited.get("/health", headers={"x-real-ip": "1.1.1.1", "origin": "https://exemplo.com"})
    assert blocked.status_code == 429
    assert int(blocked.headers["retry-after"]) >= 1
    assert blocked.headers["cache-control"] == "no-store"
    assert blocked.headers["access-control-allow-origin"] == "*"  # CORS também nas respostas 429
    assert limited.get("/health", headers={"x-real-ip": "2.2.2.2"}).status_code == 200


def test_cache_keeps_serving_stale_responses(client):
    cache = client.get("/v1/datasets/kp/latest").headers["cache-control"]
    assert "stale-while-revalidate" in cache and "stale-if-error" in cache


def test_rate_limiter_window_and_memory_bound():
    from futurespace.api.app import RateLimiter

    limiter = RateLimiter(per_minute=1, max_clients=2)
    assert limiter.retry_after("a") is None
    assert limiter.retry_after("a") is not None
    limiter.retry_after("b")
    limiter.retry_after("c")  # passa do teto de clientes: limpa os antigos sem crescer sem fim
    assert len(limiter._hits) <= 2
