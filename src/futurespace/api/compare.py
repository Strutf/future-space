"""Comparação diária IA x NOAA para o Kp, usada pelo dashboard.

Para cada bloco de 3 h de dois dias (o anterior e o escolhido), cada fonte entra com a
última previsão que emitiu **antes de o bloco começar** (ou até agora, para blocos
futuros). Assim ninguém é avaliado com informação que não tinha.

Python puro, sem pandas, para rodar na função serverless da API.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta

from futurespace.partitions import format_time

KP_STORM = 4.67  # Kp >= 5- equivale a tempestade G1 na escala da NOAA
BLOCK = timedelta(hours=3)


def _latest_issued(rows: list[dict], cutoff: str) -> dict | None:
    issued = [r for r in rows if r["time"] <= cutoff]
    return max(issued, key=lambda r: r["time"]) if issued else None


def _mae(pairs: list[tuple[float, float]]) -> float | None:
    return round(sum(abs(a - b) for a, b in pairs) / len(pairs), 2) if pairs else None


def _day_summary(day: date, blocks: list[dict]) -> dict:
    complete = [b for b in blocks if b["observed"] is not None]
    both = [b for b in complete if b["ia"] and b["noaa"]]
    mae_ia = _mae([(b["ia"]["value"], b["observed"]) for b in both])
    mae_noaa = _mae([(b["noaa"]["value"], b["observed"]) for b in both])
    if mae_ia is None or len(both) < len(blocks) // 2:
        winner = None  # poucos blocos comparáveis para declarar vencedor
    elif abs(mae_ia - mae_noaa) < 0.05:
        winner = "empate"
    else:
        winner = "ia" if mae_ia < mae_noaa else "noaa"
    return {
        "date": day.isoformat(),
        "blocks_observed": len(complete),
        "blocks_compared": len(both),
        "mae_ia": mae_ia,
        "mae_noaa": mae_noaa,
        "winner": winner,
        "storm_observed": any(b["observed"] >= KP_STORM for b in complete),
        "storm_forecast_ia": any(b["ia"] and (b["ia"]["alert"] or b["ia"]["value"] >= KP_STORM) for b in blocks),
        "storm_forecast_noaa": any(b["noaa"] and b["noaa"]["value"] >= KP_STORM for b in blocks),
    }


def build(
    predictions: list[dict],
    noaa_forecasts: list[dict],
    kp_noaa: list[dict],
    kp_gfz: list[dict],
    day: date,
    now: datetime,
) -> dict:
    ours_by_block: dict[str, list[dict]] = defaultdict(list)
    for r in predictions:
        if r["target"] == "kp":
            ours_by_block[r["target_time"]].append(r)
    noaa_by_block: dict[str, list[dict]] = defaultdict(list)
    for r in noaa_forecasts:
        noaa_by_block[r["target_time"]].append(r)
    observed = {r["time"]: r["kp"] for r in kp_noaa if r.get("kp") is not None}
    observed.update({r["time"]: r["kp"] for r in kp_gfz if r.get("kp") is not None})  # GFZ prevalece

    now_s = format_time(now)
    days = []
    for d in (day - timedelta(days=1), day):
        start = datetime(d.year, d.month, d.day, tzinfo=now.tzinfo)
        blocks = []
        for i in range(8):
            b = start + i * BLOCK
            key = format_time(b)
            cutoff = min(key, now_s)
            ia = _latest_issued(ours_by_block.get(key, []), cutoff)
            noaa = _latest_issued(noaa_by_block.get(key, []), cutoff)
            done = b + BLOCK <= now
            blocks.append(
                {
                    "block": key,
                    "observed": observed.get(key) if done else None,
                    "ia": ia
                    and {
                        "value": ia["value"],
                        "probability": ia["storm_probability"],
                        "alert": bool(ia["storm_alert"]),
                        "issued": ia["time"],
                        "horizon_hours": ia["horizon_hours"],
                    },
                    "noaa": noaa
                    and {"value": noaa["kp"], "issued": noaa["time"], "scale": noaa.get("noaa_scale")},
                }
            )
        days.append({"summary": _day_summary(d, blocks), "blocks": blocks})
    return {
        "generated_at": now_s,
        "storm_threshold": KP_STORM,
        "days": days,
        "noaa_next": _noaa_next(noaa_forecasts, now),
    }


def _noaa_next(noaa_forecasts: list[dict], now: datetime) -> dict | None:
    """Próximo bloco previsto na última emissão da NOAA, mesmo que seja amanhã."""
    now_s = format_time(now)
    issued = [r for r in noaa_forecasts if r["time"] <= now_s]
    if not issued:
        return None
    last = max(r["time"] for r in issued)
    current_block = format_time(now.replace(hour=now.hour - now.hour % 3, minute=0, second=0, microsecond=0))
    upcoming = sorted(
        (r for r in issued if r["time"] == last and r["target_time"] >= current_block),
        key=lambda r: r["target_time"],
    )
    if not upcoming:
        return None
    r = upcoming[0]
    return {"block": r["target_time"], "value": r["kp"], "scale": r.get("noaa_scale"), "issued": r["time"]}
