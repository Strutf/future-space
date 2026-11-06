from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime


def year_chunks(start: datetime, end: datetime, years: int = 1) -> Iterator[tuple[datetime, datetime]]:
    """Divide [start, end] em blocos de `years` anos para não estourar limites das fontes."""
    cur = start
    while cur < end:
        nxt = min(cur.replace(year=cur.year + years, month=1, day=1, hour=0, minute=0, second=0), end)
        yield cur, nxt
        cur = nxt
