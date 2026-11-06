from datetime import UTC, date, datetime

from futurespace.api import compare

NOW = datetime(2026, 11, 7, 10, 30, tzinfo=UTC)


def pred(issued: str, block: str, value: float, alert: bool = False) -> dict:
    return {
        "time": issued, "target": "kp", "horizon_hours": 3, "target_time": block,
        "value": value, "storm_probability": 0.1, "storm_alert": alert, "model_version": "v",
    }


def test_each_source_uses_its_last_forecast_before_the_block():
    block = "2026-11-06T21:00:00Z"
    result = compare.build(
        predictions=[
            pred("2026-11-06T18:00:00Z", block, 2.0),
            pred("2026-11-06T21:00:00Z", block, 3.0),  # emitida no início do bloco: vale
            pred("2026-11-06T22:00:00Z", block, 9.0),  # depois do início: ignorada
        ],
        noaa_forecasts=[
            {"time": "2026-11-06T19:00:00Z", "target_time": block, "kp": 5.0, "noaa_scale": "G1"},
            {"time": "2026-11-06T23:00:00Z", "target_time": block, "kp": 1.0, "noaa_scale": None},
        ],
        kp_noaa=[{"time": block, "kp": 3.0}],
        kp_gfz=[{"time": block, "kp": 3.333}],
        day=date(2026, 11, 7),
        now=NOW,
    )
    yesterday = result["days"][0]
    b = next(x for x in yesterday["blocks"] if x["block"] == block)
    assert b["ia"]["value"] == 3.0
    assert b["noaa"]["value"] == 5.0
    assert b["observed"] == 3.333  # GFZ prevalece sobre a estimativa da NOAA

    s = yesterday["summary"]
    assert s["blocks_compared"] == 1
    assert s["winner"] is None  # 1 bloco comparável não basta para declarar vencedor
    assert s["storm_forecast_noaa"] and not s["storm_forecast_ia"] and not s["storm_observed"]


def test_winner_and_future_blocks():
    blocks = [f"2026-11-06T{h:02d}:00:00Z" for h in range(0, 24, 3)]
    result = compare.build(
        predictions=[pred("2026-11-05T23:00:00Z", b, 2.0) for b in blocks],
        noaa_forecasts=[{"time": "2026-11-05T20:00:00Z", "target_time": b, "kp": 4.0} for b in blocks],
        kp_noaa=[{"time": b, "kp": 2.2} for b in blocks],
        kp_gfz=[],
        day=date(2026, 11, 7),
        now=NOW,
    )
    assert result["days"][0]["summary"]["winner"] == "ia"
    today = result["days"][1]["blocks"]
    assert today[3]["block"] == "2026-11-07T09:00:00Z"
    assert today[3]["observed"] is None  # bloco 9h-12h ainda em andamento às 10h30


def test_next_block_shows_both_sources_for_the_same_block():
    block = "2026-11-07T12:00:00Z"  # às 10h30 o bloco 9h-12h está em andamento; o próximo é o das 12h
    result = compare.build(
        predictions=[pred("2026-11-07T10:00:00Z", block, 2.3)],
        noaa_forecasts=[
            {"time": "2026-11-07T09:00:00Z", "target_time": block, "kp": 5.0, "noaa_scale": "G1"},
            {"time": "2026-11-07T08:00:00Z", "target_time": block, "kp": 3.0},  # emissão antiga
        ],
        kp_noaa=[], kp_gfz=[], day=date(2026, 11, 7), now=NOW,
    )
    nxt = result["next"]
    assert nxt["block"] == block
    assert nxt["ia"]["value"] == 2.3
    assert (nxt["noaa"]["value"], nxt["noaa"]["scale"]) == (5.0, "G1")
    assert [d["summary"]["date"] for d in result["days"]] == ["2026-11-06", "2026-11-07", "2026-11-08"]
