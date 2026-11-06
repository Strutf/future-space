"""API pública de consulta. Só lê os arquivos JSON gerados pelo pipeline: não há banco.

Depende apenas de fastapi + httpx para caber em funções serverless gratuitas.
"""

from __future__ import annotations

import csv
import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse

from futurespace import __version__, catalog
from futurespace.api import compare
from futurespace.partitions import (
    CATALOG_KEY,
    VERIFICATION_SUMMARY_KEY,
    format_time,
    json_key,
    labels_between,
    latest_key,
    manifest_key,
    parse_time,
)
from futurespace.storage import Storage, reader_from_env

DESCRIPTION = """
Dados abertos de **clima espacial** reunidos de várias fontes (NOAA, NASA, GFZ, Kyoto),
padronizados num formato comum: toda tabela tem uma coluna `time` em UTC.

- `GET /v1/datasets`: o que existe e a cobertura temporal de cada dataset
- `GET /v1/datasets/{id}/data`: série temporal num intervalo (JSON ou CSV)
- `GET /v1/datasets/{id}/latest`: valores mais recentes
- `GET /v1/datasets/predictions/latest`: previsões de Dst e Kp dos modelos do projeto
- `GET /v1/verification/summary`: quanto as previsões erraram nos últimos 7 e 30 dias

Cite sempre a fonte original indicada em `provider` e `license`.
"""

DASHBOARD_HTML = (Path(__file__).parent / "dashboard.html").read_text(encoding="utf-8")

# Datas aceitas nos filtros: cobre todo o histórico (Kp desde 1932) e evita estouros de calendário.
MIN_TIME = datetime(1900, 1, 1, tzinfo=UTC)
MAX_TIME = datetime(2100, 1, 1, tzinfo=UTC)


class CachedStorage(Storage):
    """Cache em memória com TTL: instâncias serverless "quentes" evitam leituras repetidas."""

    def __init__(self, inner: Storage, ttl: float = 60):
        self.inner = inner
        self.ttl = ttl
        self._cache: dict[str, tuple[float, bytes | None]] = {}
        self._lock = threading.Lock()

    def read(self, key: str) -> bytes | None:
        now = time.monotonic()
        with self._lock:
            hit = self._cache.get(key)
        if hit and hit[0] > now:
            return hit[1]
        data = self.inner.read(key)
        with self._lock:
            self._cache[key] = (now + self.ttl, data)
            if len(self._cache) > 512:
                self._cache.pop(next(iter(self._cache)))
        return data


def _cache_header(seconds: int) -> dict[str, str]:
    return {"Cache-Control": f"public, max-age={seconds}, s-maxage={seconds}"}


def create_app(storage: Storage | None = None) -> FastAPI:
    store = CachedStorage(storage or reader_from_env())
    pool = ThreadPoolExecutor(max_workers=8)

    app = FastAPI(
        title="Future Space API",
        version=__version__,
        description=DESCRIPTION,
        license_info={"name": "MIT (código). Dados: ver licença de cada dataset."},
    )
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])

    @app.middleware("http")
    async def security_headers(request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Frame-Options", "DENY")
        return response

    def read_json(key: str):
        try:
            raw = store.read(key)
        except httpx.HTTPError:
            # Não repassa detalhes internos (URLs, cabeçalhos) para quem chamou.
            raise HTTPException(503, "armazenamento de dados indisponível; tente novamente em instantes") from None
        return json.loads(raw) if raw else None

    def dataset_or_404(dataset_id: str) -> catalog.Dataset:
        try:
            return catalog.get(dataset_id)
        except KeyError:
            raise HTTPException(404, f"dataset '{dataset_id}' não existe; veja /v1/datasets") from None

    def parse_or_400(value: str, name: str) -> datetime:
        try:
            dt = parse_time(value)
        except ValueError:
            raise HTTPException(400, f"'{name}' inválido: use ISO 8601, ex.: 2024-05-10T00:00Z") from None
        if not MIN_TIME <= dt <= MAX_TIME:
            raise HTTPException(400, f"'{name}' fora do intervalo aceito ({MIN_TIME.year} a {MAX_TIME.year})")
        return dt

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/dashboard")

    @app.get("/v1", include_in_schema=False)
    def info():
        return {"name": "Future Space API", "version": __version__, "docs": "/docs", "datasets": "/v1/datasets"}

    @app.get("/health", include_in_schema=False)
    def health():
        return {"status": "ok"}

    @app.get("/v1/datasets", summary="Lista os datasets e sua cobertura")
    def list_datasets(response: Response):
        response.headers.update(_cache_header(300))
        published = read_json(CATALOG_KEY)
        coverage = {d["id"]: d.get("coverage") for d in (published or {}).get("datasets", [])}
        return {
            "datasets": [
                {
                    "id": ds.id,
                    "title": ds.title,
                    "stage": ds.stage,
                    "cadence": ds.cadence,
                    "provider": ds.provider,
                    "coverage": coverage.get(ds.id),
                    "links": {
                        "self": f"/v1/datasets/{ds.id}",
                        "data": f"/v1/datasets/{ds.id}/data",
                        "latest": f"/v1/datasets/{ds.id}/latest",
                    },
                }
                for ds in catalog.DATASETS
            ]
        }

    @app.get("/v1/datasets/{dataset_id}", summary="Metadados, campos e cobertura de um dataset")
    def get_dataset(dataset_id: str, response: Response):
        ds = dataset_or_404(dataset_id)
        response.headers.update(_cache_header(300))
        manifest = read_json(manifest_key(ds.id)) or {}
        return {**ds.to_dict(), "coverage": {k: manifest.get(k) for k in ("start", "end", "updated_at")}}

    @app.get("/v1/datasets/{dataset_id}/latest", summary="Valores mais recentes")
    def get_latest(dataset_id: str, response: Response):
        ds = dataset_or_404(dataset_id)
        latest = read_json(latest_key(ds.id))
        if latest is None:
            raise HTTPException(404, f"ainda não há dados para '{ds.id}'")
        response.headers.update(_cache_header(60))
        return latest

    def load_records(dataset_id: str, start: datetime, end: datetime) -> list[dict]:
        ds = catalog.get(dataset_id)
        manifest = read_json(manifest_key(ds.id))
        if not manifest:
            return []
        available = set(manifest["partitions"])
        labels = [lbl for lbl in labels_between(start, end, ds.partition) if lbl in available]
        lo, hi = format_time(start), format_time(end)
        parts = pool.map(lambda lbl: read_json(json_key(ds.id, lbl)) or [], labels)
        return [r for part in parts for r in part if lo <= r["time"] <= hi]

    @app.get("/dashboard", include_in_schema=False)
    def dashboard():
        return HTMLResponse(DASHBOARD_HTML, headers=_cache_header(300))

    @app.get("/v1/compare/kp", summary="IA x NOAA: previsões de Kp de ontem e de hoje, com o observado")
    def compare_kp(
        response: Response,
        day: str | None = Query(None, alias="date", description="Dia (UTC), ex.: 2026-11-07. Padrão: hoje"),
    ):
        now = datetime.now(UTC)
        target_day = parse_or_400(day, "date").date() if day else now.date()
        start = datetime(target_day.year, target_day.month, target_day.day, tzinfo=UTC) - timedelta(days=1)
        end = start + timedelta(days=2)
        response.headers.update(_cache_header(300))
        return compare.build(
            # Previsões emitidas até 3 dias antes ainda podem valer para os blocos do período.
            load_records("predictions", start - timedelta(days=2), end),
            load_records("kp-forecast", start - timedelta(days=4), end),
            load_records("kp", start, end),
            load_records("kp-gfz", start, end),
            target_day,
            now,
        )

    @app.get("/v1/verification/summary", summary="Desempenho recente das previsões (7 e 30 dias)")
    def verification_summary(response: Response):
        summary = read_json(VERIFICATION_SUMMARY_KEY)
        if summary is None:
            raise HTTPException(404, "ainda não há previsões verificadas")
        response.headers.update(_cache_header(300))
        return summary

    @app.get("/v1/datasets/{dataset_id}/data", summary="Série temporal num intervalo")
    def get_data(
        dataset_id: str,
        start: str | None = Query(None, description="Início (ISO 8601, UTC). Padrão: fim - default_days"),
        end: str | None = Query(None, description="Fim (ISO 8601, UTC). Padrão: último dado disponível"),
        fields: str | None = Query(None, description="Campos separados por vírgula. Padrão: todos"),
        format: Literal["json", "csv"] = "json",
    ):
        ds = dataset_or_404(dataset_id)
        manifest = read_json(manifest_key(ds.id))
        if not manifest:
            raise HTTPException(404, f"ainda não há dados para '{ds.id}'")

        end_dt = parse_or_400(end, "end") if end else parse_time(manifest["end"])
        start_dt = parse_or_400(start, "start") if start else end_dt - timedelta(days=ds.default_days)
        if start_dt > end_dt:
            raise HTTPException(400, "'start' deve ser anterior a 'end'")
        if end_dt - start_dt > timedelta(days=ds.max_query_days):
            raise HTTPException(
                400,
                f"intervalo máximo para '{ds.id}' é de {ds.max_query_days} dias; "
                "para volumes maiores baixe os arquivos Parquet do dataset",
            )

        columns = ds.columns
        if fields:
            wanted = [f.strip() for f in fields.split(",") if f.strip()]
            unknown = sorted(set(wanted) - set(columns))
            if unknown:
                raise HTTPException(400, f"campos desconhecidos: {', '.join(unknown)}")
            columns = [c for c in columns if c in ds.key or c in wanted]

        available = set(manifest["partitions"])
        labels = [lbl for lbl in labels_between(start_dt, end_dt, ds.partition) if lbl in available]
        lo, hi = format_time(start_dt), format_time(end_dt)
        rows = [
            {c: r.get(c) for c in columns}
            for part in pool.map(lambda lbl: read_json(json_key(ds.id, lbl)) or [], labels)
            for r in part
            if lo <= r["time"] <= hi
        ]

        recent = end_dt > datetime.now(UTC) - timedelta(days=3)
        headers = _cache_header(300 if recent else 86400)
        if format == "csv":
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            return Response(buf.getvalue(), media_type="text/csv", headers=headers)
        body = {
            "dataset": ds.id,
            "start": lo,
            "end": hi,
            "count": len(rows),
            "fields": columns,
            "data": rows,
        }
        return Response(json.dumps(body, separators=(",", ":")), media_type="application/json", headers=headers)

    return app


app = create_app()
