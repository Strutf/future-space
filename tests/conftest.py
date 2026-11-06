from datetime import UTC, datetime

import pandas as pd
import pytest

from futurespace import catalog
from futurespace.pipeline.writer import write_catalog, write_dataset
from futurespace.storage import LocalStorage

NOW = datetime(2026, 11, 6, 12, tzinfo=UTC)


@pytest.fixture
def store(tmp_path):
    return LocalStorage(tmp_path)


@pytest.fixture
def populated(store):
    """Armazenamento com alguns dias de Kp e vento solar de duas sondas."""
    times = pd.date_range("2026-11-01", "2026-11-06", freq="3h", tz="UTC")
    kp = pd.DataFrame({"time": times, "kp": [i % 9 for i in range(len(times))], "station_count": 8})
    write_dataset(store, catalog.get("kp"), kp, now=NOW)

    minutes = pd.date_range("2026-11-05T00:00", "2026-11-06T23:59", freq="1min", tz="UTC")
    wind = pd.concat(
        pd.DataFrame({"time": minutes, "source": src, "active": src == "SOLAR1", "speed": 400.0})
        for src in ("SOLAR1", "ACE")
    )
    write_dataset(store, catalog.get("solar-wind-plasma"), wind, now=NOW)
    write_catalog(store, list(catalog.DATASETS), now=NOW)
    return store
