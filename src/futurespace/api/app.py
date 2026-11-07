"""API pública de consulta. Só lê os arquivos JSON gerados pelo pipeline: não há banco.

Depende apenas de fastapi + httpx para caber em funções serverless gratuitas.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from futurespace import __version__, catalog, model_index
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
- `GET /v1/models`: histórico dos modelos treinados, com métricas e o peso de cada grupo de variáveis
- `GET /v1/models/{id}`: ficha completa de um treino (código, dados, hiperparâmetros, avaliação)

Cite sempre a fonte original indicada em `provider` e `license`.
"""

WEB = Path(__file__).parent / "web"
STATIC = WEB / "static"
# Versão de cada arquivo estático = início do hash do conteúdo. Vai no endereço (?v=...), então
# qualquer mudança no arquivo muda o endereço e nenhum navegador usa uma cópia antiga.
STATIC_VERSIONS = {
    f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:10] for f in STATIC.iterdir() if f.is_file()
}

# Páginas do site: cada uma é o miolo em web/pages, encaixado no molde web/base.html, com os
# textos vindos de web/locales/<idioma>.json. Cada idioma tem os próprios endereços.
LANGS = ("pt", "en", "es", "fr")
PAGE_URLS = {
    "home": {"pt": "/", "en": "/en", "es": "/es", "fr": "/fr"},
    "dashboard": {"pt": "/dashboard", "en": "/en/dashboard", "es": "/es/dashboard", "fr": "/fr/dashboard"},
    "sobre": {"pt": "/sobre", "en": "/en/about", "es": "/es/acerca", "fr": "/fr/a-propos"},
    "interativo": {"pt": "/interativo", "en": "/en/interactive", "es": "/es/interactivo", "fr": "/fr/interactif"},
    "tempestade": {
        "pt": "/interativo/tempestade", "en": "/en/interactive/storm",
        "es": "/es/interactivo/tormenta", "fr": "/fr/interactif/tempete",
    },
    "viagem": {
        "pt": "/interativo/viagem-no-tempo", "en": "/en/interactive/time-travel",
        "es": "/es/interactivo/viaje-en-el-tiempo", "fr": "/fr/interactif/voyage-dans-le-temps",
    },
    "aurora": {
        "pt": "/interativo/aurora", "en": "/en/interactive/aurora",
        "es": "/es/interactivo/aurora", "fr": "/fr/interactif/aurore",
    },
}
# Páginas que pertencem a uma seção do menu (o menu destaca a seção).
SECTIONS = {"tempestade": "interativo", "viagem": "interativo", "aurora": "interativo"}
SITE_URL = "https://future-space-six.vercel.app"
LOCALES = {lang: json.loads((WEB / "locales" / f"{lang}.json").read_text(encoding="utf-8")) for lang in LANGS}


def _lookup(locale: dict, key: str) -> str:
    value = locale
    for part in key.split("."):
        value = value[part]  # KeyError aqui = chave faltando no arquivo de tradução
    return value


def _language_menu(name: str, lang: str) -> str:
    """Itens do seletor: cada idioma com o próprio nome, apontando para a mesma página."""
    items = []
    for other in LANGS:
        loc = LOCALES[other]
        current = ' aria-current="true"' if other == lang else ""
        items.append(
            f'<a href="{PAGE_URLS[name][other]}" lang="{loc["lang"]}" hreflang="{loc["lang"]}" '
            f'data-lang="{other}" role="listitem"{current}>{loc["nav"]["lang_name"]}</a>'
        )
    return "\n        ".join(items)


def _hreflang_links(name: str) -> str:
    links = [
        f'<link rel="alternate" hreflang="{LOCALES[other]["lang"]}" href="{SITE_URL}{PAGE_URLS[name][other]}">'
        for other in LANGS
    ]
    links.append(f'<link rel="alternate" hreflang="x-default" href="{SITE_URL}{PAGE_URLS[name]["en"]}">')
    return "\n".join(links)


def render_page(name: str, lang: str) -> str:
    html = (WEB / "base.html").read_text(encoding="utf-8").replace(
        "{{content}}", (WEB / "pages" / f"{name}.html").read_text(encoding="utf-8")
    )
    locale = LOCALES[lang]
    # Primeiro os textos (que podem conter {{url:...}}), depois os endereços.
    html = re.sub(r"\{\{t:([\w.]+)\}\}", lambda m: _lookup(locale, m.group(1)), html)
    html = re.sub(r"\{\{url:(\w+)\}\}", lambda m: PAGE_URLS[m.group(1)][lang], html)
    html = re.sub(
        r"\{\{static:([\w.-]+)\}\}", lambda m: f"/static/{m.group(1)}?v={STATIC_VERSIONS[m.group(1)]}", html
    )
    js_texts = json.dumps(locale["js"], ensure_ascii=False).replace("</", "<\\/")
    # Endereços desta página em cada idioma, para o redirecionamento da primeira visita.
    page_urls = json.dumps(PAGE_URLS[name]).replace('"', "&quot;")
    replacements = {
        "{{title}}": locale["titles"][name],
        "{{page}}": name,
        "{{section}}": SECTIONS.get(name, name),
        "{{lang}}": lang,
        "{{page_urls}}": page_urls,
        "{{hreflang_links}}": _hreflang_links(name),
        "{{language_menu}}": _language_menu(name, lang),
        "{{js_i18n}}": js_texts,
    }
    for placeholder, value in replacements.items():
        html = html.replace(placeholder, value)
    return html


RENDERED_PAGES = {
    urls[lang]: render_page(name, lang) for name, urls in PAGE_URLS.items() for lang in LANGS
}

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


class RateLimiter:
    """Janela deslizante de 60 s por cliente, em memória.

    Na Vercel cada instância da função tem a própria memória, então o limite vale por
    instância: barra scripts e abusos simples, não um ataque distribuído (para isso, a
    regra de rate limit do firewall da Vercel).
    """

    WINDOW = 60.0

    def __init__(self, per_minute: int, max_clients: int = 10_000):
        self.per_minute = per_minute
        self.max_clients = max_clients
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def retry_after(self, client: str) -> float | None:
        """Registra a requisição; devolve os segundos de espera se o limite estourou."""
        now = time.monotonic()
        with self._lock:
            hits = self._hits.get(client)
            if hits is None:
                if len(self._hits) >= self.max_clients:
                    self._prune(now)
                hits = self._hits[client] = deque()
            while hits and now - hits[0] >= self.WINDOW:
                hits.popleft()
            if len(hits) >= self.per_minute:
                return self.WINDOW - (now - hits[0])
            hits.append(now)
            return None

    def _prune(self, now: float) -> None:
        self._hits = {k: v for k, v in self._hits.items() if v and now - v[-1] < self.WINDOW}
        if len(self._hits) >= self.max_clients:
            self._hits.clear()  # proteção de memória: em último caso, recomeça a contagem


# Únicos parâmetros de URL aceitos. Qualquer outro é recusado, o que impede furar o cache
# da CDN com URLs sempre diferentes (?x=1, ?x=2...) sem custo para quem ataca.
ALLOWED_QUERY_PARAMS = frozenset({"start", "end", "fields", "format", "date", "v"})


def _client_ip(request: Request) -> str:
    # Na Vercel, x-real-ip e x-forwarded-for são definidos pela borda, não pelo cliente.
    forwarded = request.headers.get("x-forwarded-for", "")
    return request.headers.get("x-real-ip") or forwarded.split(",")[0].strip() or (
        request.client.host if request.client else "desconhecido"
    )


def _cache_header(seconds: int) -> dict[str, str]:
    # stale-while-revalidate: a CDN continua servindo a última resposta boa enquanto
    # atualiza em segundo plano, o que segura picos de acesso sem chamar a função.
    return {
        "Cache-Control": f"public, max-age={seconds}, s-maxage={seconds}, "
        f"stale-while-revalidate={max(seconds, 600)}, stale-if-error=86400"
    }


def create_app(storage: Storage | None = None, rate_limit_per_minute: int | None = None) -> FastAPI:
    store = CachedStorage(storage or reader_from_env())
    pool = ThreadPoolExecutor(max_workers=8)
    if rate_limit_per_minute is None:
        rate_limit_per_minute = int(os.environ.get("FS_RATE_LIMIT_PER_MINUTE", "60"))
    limiter = RateLimiter(rate_limit_per_minute) if rate_limit_per_minute > 0 else None

    app = FastAPI(
        title="Future Space API",
        version=__version__,
        description=DESCRIPTION,
        license_info={"name": "MIT (código). Dados: ver licença de cada dataset."},
    )

    # Registrado antes do CORS e dos cabeçalhos de segurança, que assim também envolvem
    # as respostas 400/429 daqui.
    @app.middleware("http")
    async def guard(request: Request, call_next):
        unknown = sorted(set(request.query_params) - ALLOWED_QUERY_PARAMS)
        if unknown:
            return JSONResponse(
                {"detail": f"parâmetros desconhecidos: {', '.join(unknown)}"},
                status_code=400,
                headers=_cache_header(3600),
            )
        if limiter:
            wait = limiter.retry_after(_client_ip(request))
            if wait is not None:
                return JSONResponse(
                    {"detail": "muitas requisições; aguarde antes de tentar de novo"},
                    status_code=429,
                    headers={"Retry-After": str(max(1, int(wait) + 1)), "Cache-Control": "no-store"},
                )
        return await call_next(request)

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

    def page_route(html: str):
        return lambda: HTMLResponse(html, headers=_cache_header(300))

    for path, html in RENDERED_PAGES.items():
        app.add_api_route(path, page_route(html), methods=["GET"], include_in_schema=False)

    @app.get("/static/{name}", include_in_schema=False)
    def static_file(name: str, v: str | None = None):
        # Só arquivos que existem direto em web/static: nada de subpastas nem "..".
        path = STATIC / name
        if "/" in name or name.startswith(".") or not path.is_file():
            raise HTTPException(404, "arquivo não encontrado")
        media = {".css": "text/css", ".js": "text/javascript", ".svg": "image/svg+xml", ".png": "image/png", ".json": "application/json"}
        # Com a versão certa no endereço, o conteúdo nunca muda: pode ficar em cache para sempre.
        # Sem ela (ou com versão antiga), cache curto, para não prender ninguém num arquivo velho.
        cache = (
            {"Cache-Control": "public, max-age=31536000, immutable"}
            if v and v == STATIC_VERSIONS.get(name)
            else _cache_header(60)
        )
        return Response(
            path.read_bytes(),
            media_type=media.get(path.suffix, "application/octet-stream"),
            headers=cache,
        )

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
            # Coletas de hora em hora + arquivo das edições oficiais (cobre dias antes da coleta).
            load_records("kp-forecast", start - timedelta(days=4), end)
            + load_records("kp-forecast-3day", start - timedelta(days=4), end),
            load_records("kp", start, end),
            load_records("kp-gfz", start, end),
            target_day,
            now,
        )

    @app.get("/v1/compare/dst", summary="Dst hora a hora (ontem, hoje e amanhã): observado e previsão da IA")
    def compare_dst(
        response: Response,
        day: str | None = Query(None, alias="date", description="Dia (UTC), ex.: 2026-11-07. Padrão: hoje"),
    ):
        now = datetime.now(UTC)
        target_day = parse_or_400(day, "date").date() if day else now.date()
        start = datetime(target_day.year, target_day.month, target_day.day, tzinfo=UTC) - timedelta(days=1)
        end = start + timedelta(days=2)
        response.headers.update(_cache_header(300))
        return compare.build_dst(
            load_records("predictions", start - timedelta(days=2), end),
            load_records("dst", start, end),
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

    @app.get("/v1/models", summary="Histórico dos modelos: versões, métricas e importância das variáveis")
    def list_models(response: Response):
        try:
            index = model_index.read_index(store)
        except httpx.HTTPError:
            raise HTTPException(503, "armazenamento de dados indisponível; tente novamente em instantes") from None
        if not index["versions"]:
            raise HTTPException(404, "nenhum modelo treinado ainda")
        response.headers.update(_cache_header(3600))
        return {
            **index,
            "groups": model_index.FEATURE_GROUPS,
            "notes": {
                "metrics": "período de teste (2019 em diante), com o modelo treinado sem esse período",
                "importance": "fatia de cada grupo no aumento do erro (RMSE na regressão, Brier na "
                "chance de tempestade) quando as variáveis do grupo são embaralhadas",
            },
            "links": {v["id"]: f"/v1/models/{v['id']}" for v in index["versions"]},
        }

    @app.get("/v1/models/{model_id}", summary="Ficha completa de um treino")
    def get_model(model_id: str, response: Response):
        try:
            index = model_index.read_index(store)
        except httpx.HTTPError:
            raise HTTPException(503, "armazenamento de dados indisponível; tente novamente em instantes") from None
        # Só abre relatórios listados no índice: o id nunca vira caminho diretamente.
        entry = next((v for v in index["versions"] if v["id"] == model_id), None)
        if entry is None:
            raise HTTPException(404, f"modelo '{model_id}' não existe; veja /v1/models")
        report = read_json(entry["report"])
        if report is None:
            raise HTTPException(404, f"relatório de '{model_id}' não encontrado")
        response.headers.update(_cache_header(3600))
        return {"status": entry["status"], "reason": entry["reason"], **report}

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
