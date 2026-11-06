from datetime import UTC, datetime

from futurespace.partitions import labels_between, parse_time


def dt(s: str) -> datetime:
    return parse_time(s)


def test_labels_day():
    assert labels_between(dt("2026-10-30T12:00"), dt("2026-11-02T01:00"), "day") == [
        "2026-10-30",
        "2026-10-31",
        "2026-11-01",
        "2026-11-02",
    ]


def test_labels_month_crosses_year():
    assert labels_between(dt("2025-11-30"), dt("2026-02-01"), "month") == [
        "2025-11",
        "2025-12",
        "2026-01",
        "2026-02",
    ]


def test_labels_year_single():
    assert labels_between(dt("2024-05-10"), dt("2024-05-11"), "year") == ["2024"]


def test_parse_time_defaults_to_utc():
    assert parse_time("2024-05-10") == datetime(2024, 5, 10, tzinfo=UTC)
    assert parse_time("2024-05-10T03:00:00Z") == datetime(2024, 5, 10, 3, tzinfo=UTC)
