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
        return [d for d in catalog.DATASETS if d.schedule != "derived"]
    return [d for d in catalog.DATASETS if d.schedule == schedule]


def cmd_update(args: argparse.Namespace) -> int:
    datasets = _select(args.datasets, args.schedule)
    end = datetime.now(UTC)
    start = end - timedelta(days=args.days) if args.days else None
    return _run(datasets, start, end)


def cmd_backfill(args: argparse.Namespace) -> int:
    ds = catalog.get(args.dataset)
    if not ds.historical:
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
    from futurespace.ml.serve import current_report, publish_bundle, record_rejected
    from futurespace.ml.train import compare, train_all

    store = storage_mod.from_env()
    # Revisão lida antes dos dados: as coletas não gravam ao mesmo tempo (grupo hf-writes).
    revision = store.data_revision()
    raw = {"omni-hourly": load_dataset(store, "omni-hourly"), "kp-gfz": load_dataset(store, "kp-gfz")}
    if not args.no_cme:
        raw["cme-cactus"] = load_dataset(store, "cme-cactus")
    hourly = hourly_from_history(raw["omni-hourly"], raw["kp-gfz"])
    if args.no_refit and (args.publish or args.publish_if_better):
        _error("--no-refit gera o modelo de benchmark (treinado até 2018); ele não deve ser publicado")
        return 1
    trained, summary = train_all(hourly, raw.get("cme-cactus"), refit_all=not args.no_refit)
    summary["reproducibility"]["data"] = {
        "storage": os.environ.get("FS_STORAGE", "local"),
        "repository": os.environ.get("FS_HF_REPO"),
        "revision": revision,
        "datasets": {
            name: {"rows": len(df), "start": str(df["time"].min()), "end": str(df["time"].max())}
            for name, df in raw.items()
        },
    }

    bundle = {
        "model_version": summary["model_version"],
        "features": summary["features"],
        "models": {
            t.name: {"regressor": t.regressor, "classifier": t.classifier, "threshold": t.threshold}
            for t in trained
        },
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out / "bundle.joblib", compress=3)
    (out / "report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(
        f"\n{'modelo':<9}{'RMSE':>7}{'persist.':>9} │{'TSS clf':>8}{'regr.':>7}{'persist.':>9}"
        f"{'POD':>6}{'FAR':>6}{'BSS':>6}{'eventos':>9}"
    )
    for r in summary["models"]:
        reg, storm = r["regression"], r["storm"]
        bss = 1 - storm["brier"] / storm["brier_climatology"]
        print(
            f"{r['target'] + ' +' + str(r['horizon_hours']) + 'h':<9}"
            f"{reg['model']['rmse']:>7.2f}{reg['persistence']['rmse']:>9.2f} │"
            f"{storm['classifier']['tss']:>8.2f}{storm['regression']['tss']:>7.2f}"
            f"{storm['persistence']['tss']:>9.2f}{storm['classifier']['pod']:>6.2f}"
            f"{storm['classifier']['far']:>6.2f}{bss:>6.2f}{storm['classifier']['events']:>9}"
        )
    print(f"\nModelos e relatório em {out}/")

    publish = args.publish
    if args.publish_if_better:
        current = current_report(store)
        if current is None:
            publish, reason = True, "nenhum modelo publicado ainda"
        else:
            publish, reason = compare(summary, current)
            reason = f"comparado com {current['model_version']}: {reason}"
        log.info("publicação %s: %s", "aprovada" if publish else "recusada", reason)
        if not publish:
            # O treino recusado também entra no histórico, com o motivo.
            record_rejected(store, summary, reason)
            store.commit(f"treino {summary['model_version']} não publicado")
            if os.environ.get("GITHUB_ACTIONS") == "true":
                print(f"::warning::Modelo novo não publicado ({reason})", flush=True)
    if publish:
        publish_bundle(store, out / "bundle.joblib", summary)
        store.commit(f"modelos {summary['model_version']}")
        log.info("publicado como %s", summary["model_version"])
    return 0


def cmd_publish_model(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from futurespace import storage as storage_mod
    from futurespace.ml.serve import publish_bundle

    folder = Path(args.dir)
    summary = json.loads((folder / "report.json").read_text())
    store = storage_mod.from_env()
    publish_bundle(store, folder / "bundle.joblib", summary)
    store.commit(f"modelos {summary['model_version']}")
    log.info("modelos %s publicados", summary["model_version"])
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from futurespace import storage as storage_mod
    from futurespace.ml.serve import NoModelPublished, predict
    from futurespace.pipeline.writer import write_catalog, write_dataset

    try:
        store = storage_mod.from_env()
        rows = predict(store)
    except NoModelPublished as exc:
        log.warning("previsão ignorada: %s", exc)
        return 0
    except RuntimeError as exc:
        _error(f"previsão: {exc}")
        return 1
    now = datetime.now(UTC)
    write_dataset(store, catalog.get("predictions"), rows, now=now)
    write_catalog(store, list(catalog.DATASETS), now=now)
    store.commit(f"previsões {rows['time'].iloc[0]:%Y-%m-%d %H:%M}Z")
    for r in rows.itertuples():
        alert = " ALERTA" if r.storm_alert else ""
        log.info("%s +%dh: %.2f (tempestade %.0f%%)%s", r.target, r.horizon_hours, r.value, 100 * r.storm_probability, alert)
    return 0


def cmd_aurora(args: argparse.Namespace) -> int:
    import pandas as pd

    from futurespace import storage as storage_mod
    from futurespace.aurora.visibility import forecast
    from futurespace.pipeline.writer import write_catalog, write_dataset

    store = storage_mod.from_env()
    rows = forecast(store)
    now = datetime.now(UTC)
    write_dataset(store, catalog.get("aurora-visibility"), rows, now=now)
    write_catalog(store, list(catalog.DATASETS), now=now)
    store.commit(f"aurora {rows['time'].iloc[0]:%Y-%m-%d %H:%M}Z")
    best = rows.sort_values("visibility_score", ascending=False).drop_duplicates("site").head(5)
    for r in best.itertuples():
        log.info(
            "%s: pontuação %.2f às %s (aurora %.2f, nuvens %s%%, Kp %.1f)",
            r.site, r.visibility_score, f"{r.target_time:%H}h UTC", r.aurora_score,
            "?" if pd.isna(r.cloud_cover) else f"{r.cloud_cover:.0f}", r.kp_expected,
        )
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    from futurespace import storage as storage_mod
    from futurespace.ml.serve import read_recent
    from futurespace.ml.verify import summarize, verify, write_summary
    from futurespace.pipeline.writer import write_catalog, write_dataset

    store = storage_mod.from_env()
    now = datetime.now(UTC)
    rows = verify(store, now=now, days=args.days)
    if rows.empty:
        log.info("nenhuma previsão com valor observado para verificar ainda")
        return 0
    ds = catalog.get("verification")
    write_dataset(store, ds, rows, now=now)
    history = read_recent(store, ds.id, now - timedelta(days=31), now)
    summary = summarize(history, now)
    write_summary(store, summary)
    write_catalog(store, list(catalog.DATASETS), now=now)
    store.commit(f"verificação {now:%Y-%m-%d %H:%M}Z")
    for g in summary["windows"]["7d"]:
        log.info(
            "%s +%dh (7 dias, n=%d): erro médio %s | persistência %s | NOAA %s",
            g["target"], g["horizon_hours"], g["n"], g["mae_model"], g["mae_persistence"], g["mae_noaa"],
        )
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from pathlib import Path

    import joblib

    from futurespace import storage as storage_mod
    from futurespace.ml.benchmark import run

    # Arquivos joblib executam código ao carregar: use só modelos treinados por você.
    bundle = joblib.load(Path(args.bundle))
    print(run(storage_mod.from_env(), bundle))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from futurespace import storage as storage_mod
    from futurespace.ml.evaluate import markdown
    from futurespace.ml.features import load_dataset

    # Só lê: usa o mesmo armazenamento da API (FS_DATA_URL, FS_HF_REPO ou a pasta local).
    rows = load_dataset(storage_mod.reader_from_env(), "verification")
    if args.start:
        rows = rows[rows["target_time"] >= parse_time(args.start)]
    if args.end:
        rows = rows[rows["target_time"] <= parse_time(args.end)]
    print(markdown(rows))
    return 0


def cmd_export_web(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    import joblib

    from futurespace import storage as storage_mod
    from futurespace.ml.export_web import build_replays, train_simulator
    from futurespace.ml.features import hourly_from_history, load_dataset

    store = storage_mod.from_env()
    hourly = hourly_from_history(load_dataset(store, "omni-hourly"), load_dataset(store, "kp-gfz"))
    out = Path(__file__).parent / "api" / "web" / "static"
    sim = train_simulator(hourly)
    (out / "sim-model.json").write_text(json.dumps(sim, separators=(",", ":")))
    replays = build_replays(hourly, load_dataset(store, "cme-cactus"), joblib.load(args.bundle))
    (out / "replays.json").write_text(json.dumps(replays, separators=(",", ":"), ensure_ascii=False))
    for name in ("sim-model.json", "replays.json"):
        log.info("%s: %.0f KB", name, (out / name).stat().st_size / 1024)
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
    p.add_argument("--no-cme", action="store_true", help="treina sem as variáveis de CME")
    p.add_argument(
        "--no-refit",
        action="store_true",
        help="não re-treina com todos os dados: o modelo fica treinado até 2018 (para o benchmark)",
    )
    p.add_argument("--publish", action="store_true", help="publica os modelos no armazenamento")
    p.add_argument(
        "--publish-if-better",
        action="store_true",
        help="publica só se a avaliação não for pior que a do modelo atual",
    )
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("verify", help="compara as previsões passadas com o que aconteceu")
    p.add_argument("--days", type=int, default=30, help="quantos dias de previsões verificar")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("publish-model", help="publica modelos já treinados no armazenamento")
    p.add_argument("--dir", default="models", help="pasta com bundle.joblib e report.json")
    p.set_defaults(func=cmd_publish_model)

    p = sub.add_parser("predict", help="emite as previsões com os dados mais recentes")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("aurora", help="previsão de visibilidade da aurora por cidade (0 a 24 h)")
    p.set_defaults(func=cmd_aurora)

    p = sub.add_parser("benchmark", help="benchmark histórico contra a previsão da NOAA (desde 2022)")
    p.add_argument("--bundle", default="models-2018/bundle.joblib", help="modelo de `train --no-refit`")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("evaluate", help="métricas das previsões verificadas (tabelas Markdown)")
    p.add_argument("--start", help="início do período (data-alvo, UTC)")
    p.add_argument("--end", help="fim do período (data-alvo, UTC)")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("export-web", help="gera os arquivos da parte interativa do site")
    p.add_argument("--bundle", default="models/bundle.joblib",
                   help="modelos para a Viagem no tempo (de preferência treinados só até 2018)")
    p.set_defaults(func=cmd_export_web)

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
