import pytest

from futurespace import catalog, cli
from futurespace.pipeline import connectors


@pytest.fixture
def local(tmp_path, monkeypatch):
    monkeypatch.setenv("FS_STORAGE", "local")
    monkeypatch.setenv("FS_DATA_DIR", str(tmp_path))


def _fail(*_):
    raise RuntimeError("handshake operation timed out")


def test_experimental_source_failure_does_not_fail_the_collection(local, monkeypatch, caplog):
    monkeypatch.setitem(connectors.CONNECTORS, "aurora-weather", _fail)
    status = cli._run([catalog.get("aurora-weather")], None, cli.datetime.now(cli.UTC))
    assert status == 0
    assert "fontes experimentais com falha" in caplog.text


def test_core_source_failure_still_fails_the_collection(local, monkeypatch):
    monkeypatch.setitem(connectors.CONNECTORS, "dst", _fail)
    monkeypatch.setitem(connectors.CONNECTORS, "aurora-weather", _fail)
    status = cli._run([catalog.get("dst"), catalog.get("aurora-weather")], None, cli.datetime.now(cli.UTC))
    assert status == 1


def test_only_aurora_sources_are_experimental():
    assert {d.id for d in catalog.DATASETS if not d.critical} == {"aurora-ovation", "aurora-weather"}
