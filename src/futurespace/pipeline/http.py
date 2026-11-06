from __future__ import annotations

import logging
import time

import httpx

from futurespace import __version__

log = logging.getLogger(__name__)

USER_AGENT = f"future-space/{__version__} (dados abertos de clima espacial)"


def make_client(transport: httpx.BaseTransport | None = None) -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=httpx.Timeout(60, connect=15),
        follow_redirects=True,
        transport=transport,
    )


def get(client: httpx.Client, url: str, params: dict | None = None, retries: int = 3) -> httpx.Response:
    """GET com nova tentativa em erros de rede e respostas 5xx/429."""
    for attempt in range(retries + 1):
        try:
            resp = client.get(url, params=params)
            if resp.status_code < 500 and resp.status_code != 429:
                resp.raise_for_status()
                return resp
            error: Exception = httpx.HTTPStatusError(
                f"HTTP {resp.status_code}", request=resp.request, response=resp
            )
        except httpx.TransportError as exc:
            error = exc
        if attempt == retries:
            raise error
        wait = 2**attempt
        log.warning("falha em %s (%s); nova tentativa em %ss", url, error, wait)
        time.sleep(wait)
    raise AssertionError("inalcançável")


def get_json(client: httpx.Client, url: str, params: dict | None = None):
    return get(client, url, params).json()
