"""Linha de comando: `futurespace update | backfill | datasets | serve`."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import UTC, datetime, timedelta

from futurespace import catalog
from futurespace.partitions import parse_time

log = logging.getLogger("futurespace")


def _error(message: str) -> None:
    """Loga o erro e, no GitHub Actions, também o publica como anotação da execução."""
    log.error(message)
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::error::{message}", flush=True)


def _run(datasets: list[catalog.Dataset], start: datetime | None, end: datetime) -> int:
    from futurespace import storage as storage_mod
    from futurespace.pipeline.connectors import CONNECTORS
    from futurespace.pipeline.http import make_client
    from futurespace.pipeline.writer import write_catalog, write_dataset

    try:
        store = storage_mod.from_env()
    except RuntimeError as exc:
        _error(f"configuração: {exc}")
        return 1

    failures = []
    written = []
    with make_client() as client:
        for ds in datasets:
            ds_start = start or end - timedelta(days=ds.update_days)
            try:
                df = CONNECTORS[ds.id](client, ds_start, end)
                result = write_dataset(store, ds, df, now=end)
            except Exception as exc:  # noqa: BLE001 - uma fonte fora do ar não deve parar as outras
                _error(f"{ds.id}: falhou: {exc}")
                failures.append(ds.id)
                continue
            log.info("%s: %d linhas em %s", ds.id, result.rows, ", ".join(result.partitions) or "-")
            if result.rows:
                written.append(ds.id)

    if written:
        try:
            write_catalog(store, list(catalog.DATASETS), now=end)
            store.commit(f"update {end:%Y-%m-%d %H:%M}Z: {', '.join(written)}")
        except Exception as exc:  # noqa: BLE001 - registra o motivo antes de falhar
            _error(f"falha ao gravar no armazenamento ({type(exc).__name__}): {exc}")
            return 1
    if failures:
        _error(f"datasets com falha: {', '.join(failures)}")
        return 1
    return 0


def _select(names: str | None, schedule: str) -> list[catalog.Dataset]:
    if names:
        return [catalog.get(n.strip()) for n in names.split(",") if n.strip()]
    if schedule == "all":
        return list(catalog.DATASETS)
    return [d for d in catalog.DATASETS if d.schedule == schedule]


def cmd_update(args: argparse.Namespace) -> int:
    datasets = _select(args.datasets, args.schedule)
    end = datetime.now(UTC)
    start = end - timedelta(days=args.days) if args.days else None
    return _run(datasets, start, end)


def cmd_backfill(args: argparse.Namespace) -> int:
    ds = catalog.get(args.dataset)
    if ds.schedule == "realtime":
        log.warning(
            "%s vem de um endpoint de tempo real: só existem os últimos dias na fonte", ds.id
        )
    end = parse_time(args.end) if args.end else datetime.now(UTC)
    return _run([ds], parse_time(args.start), end)


def cmd_train(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    import joblib

    from futurespace import storage as storage_mod
    from futurespace.ml.features import hourly_from_history, load_dataset
    from futurespace.ml.train import train_all

    store = storage_mod.from_env()
    hourly = hourly_from_history(load_dataset(store, "omni-hourly"), load_dataset(store, "kp-gfz"))
    trained, summary = train_all(hourly)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for t in trained:
        joblib.dump(t.model, out / f"{t.target}_h{t.horizon}.joblib", compress=3)
    (out / "report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n{'modelo':<8}{'RMSE modelo':>13}{'RMSE persist.':>15}{'TSS modelo':>12}{'TSS persist.':>14}{'eventos':>9}")
    for r in summary["models"]:
        m, p = r["model"], r["persistence"]
        name = f"{r['target']} +{r['horizon_hours']}h"
        print(
            f"{name:<8}{m['rmse']:>13.2f}{p['rmse']:>15.2f}"
            f"{m['storm']['tss']:>12.2f}{p['storm']['tss']:>14.2f}{m['storm']['events']:>9}"
        )
    print(f"\nModelos e relatório em {out}/")
    return 0


def cmd_datasets(args: argparse.Namespace) -> int:
    for ds in catalog.DATASETS:
        print(f"{ds.id:<20} {ds.schedule:<9} {ds.cadence:<6} {ds.title}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("futurespace.api.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="futurespace", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("update", help="coleta os dados recentes de cada fonte")
    p.add_argument("--schedule", choices=["realtime", "daily", "all"], default="realtime")
    p.add_argument("--datasets", help="lista separada por vírgula (ignora --schedule)")
    p.add_argument("--days", type=int, help="janela de busca (padrão: update_days do dataset)")
    p.set_defaults(func=cmd_update)

    p = sub.add_parser("backfill", help="baixa o histórico de um dataset")
    p.add_argument("dataset")
    p.add_argument("--start", required=True, help="ex.: 1963-01-01")
    p.add_argument("--end", help="padrão: agora")
    p.set_defaults(func=cmd_backfill)

    p = sub.add_parser("train", help="treina os modelos de previsão de Dst e Kp")
    p.add_argument("--out", default="models", help="pasta de saída dos modelos")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("datasets", help="lista os datasets do catálogo")
    p.set_defaults(func=cmd_datasets)

    p = sub.add_parser("serve", help="sobe a API localmente")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "httpx2", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
