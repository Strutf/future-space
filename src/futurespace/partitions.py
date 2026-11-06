"""Layout dos arquivos no armazenamento, compartilhado entre pipeline e API.

    parquet/<dataset>/<partição>.parquet   -> arquivo para treino/análise
    json/<dataset>/<partição>.json         -> lido pela API (e pelo futuro frontend)
    json/<dataset>/manifest.json           -> partições existentes e cobertura temporal
    json/<dataset>/latest.json             -> últimos valores
    json/catalog.json                      -> catálogo completo
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from futurespace.catalog import Partition

TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def label(dt: datetime, partition: Partition) -> str:
    if partition == "day":
        return dt.strftime("%Y-%m-%d")
    if partition == "month":
        return dt.strftime("%Y-%m")
    return dt.strftime("%Y")


def labels_between(start: datetime, end: datetime, partition: Partition) -> list[str]:
    """Rótulos de todas as partições que intersectam [start, end]."""
    out: list[str] = []
    cur = start
    while cur <= end:
        lbl = label(cur, partition)
        if not out or out[-1] != lbl:
            out.append(lbl)
        if partition == "day":
            cur += timedelta(days=1)
        elif partition == "month":
            cur = (cur.replace(day=1) + timedelta(days=32)).replace(day=1)
        else:
            cur = cur.replace(year=cur.year + 1, month=1, day=1)
    last = label(end, partition)
    if out[-1:] != [last]:
        out.append(last)
    return out


def format_time(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime(TIME_FORMAT)


def parse_time(value: str) -> datetime:
    """Aceita '2026-11-06', '2026-11-06T12:00' ou '2026-11-06T12:00:00Z'. Sem fuso = UTC."""
    dt = datetime.fromisoformat(value.strip())
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def parquet_key(dataset_id: str, lbl: str) -> str:
    return f"parquet/{dataset_id}/{lbl}.parquet"


def json_key(dataset_id: str, lbl: str) -> str:
    return f"json/{dataset_id}/{lbl}.json"


def manifest_key(dataset_id: str) -> str:
    return f"json/{dataset_id}/manifest.json"


def latest_key(dataset_id: str) -> str:
    return f"json/{dataset_id}/latest.json"


CATALOG_KEY = "json/catalog.json"
VERIFICATION_SUMMARY_KEY = "json/verification/summary.json"
