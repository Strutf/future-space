"""Piloto da calibração da aurora: sorteia momentos escuros e de céu limpo de uma câmera THEMIS
e baixa uma imagem de cada um.

1. Lê a classificação de nuvens da U. Calgary (AuroraX, a cada 10 min) para o período.
2. Mantém os horários de céu limpo (confiança >= 90%) com o Sol abaixo de -12°.
3. Sorteia N horários (semente fixa) e baixa o arquivo de 1 minuto (~2,5 MB, 20 quadros);
   guarda só o primeiro quadro (256x256, 16 bits) e descarta o resto.

Saída na pasta indicada: frames.npz (quadros, chave = AAAAMMDD_HHMM), samples.parquet e
clouds.parquet. As tarefas planejadas estão em themis_plan.json.

Uso: python research/aurora/sample_themis.py fsmi 2015-11-01 2016-03-01 500 <pasta_saída>
"""

from __future__ import annotations

import gzip
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from futurespace.aurora import astro

CALGARY = "https://data.phys.ucalgary.ca/sort_by_project/THEMIS/asi"
CLOUD = f"{CALGARY}/machine_learning/ucalgary_themis_asi_cloud_v2.0"
# Coordenadas das câmeras (aproximadas, para o cálculo da altura do Sol).
STATIONS = {
    "atha": (54.60, -113.64),
    "fsim": (61.76, -121.23),
    "fsmi": (60.03, -111.93),
    "gill": (56.38, -94.64),
    "inuv": (68.41, -133.77),
    "rank": (62.82, -92.11),
}
WORKERS = 4  # downloads simultâneos: rápido sem sobrecarregar o servidor


def cloud_day(client: httpx.Client, site: str, day: pd.Timestamp) -> pd.DataFrame:
    folder = f"{CLOUD}/{day:%Y/%m/%d}/"
    listing = client.get(folder)
    if listing.status_code != 200:
        return pd.DataFrame()
    names = re.findall(rf'href="({day:%Y%m%d}_{site}_[^"]+_classifications\.txt)"', listing.text)
    if not names:
        return pd.DataFrame()
    text = client.get(folder + names[0]).text
    rows = [line.split() for line in text.splitlines() if re.match(r"\d{4}-\d\d-\d\d", line)]
    df = pd.DataFrame(rows, columns=["date", "time", "pred", "label", "conf_clear", "conf_cloudy"])
    df["time"] = pd.to_datetime(df["date"] + " " + df["time"], utc=True)
    df["imager"] = names[0].split("_")[2]
    return df[["time", "imager", "label"]].assign(conf_clear=df["conf_clear"].astype(float))


def first_frame(raw: bytes) -> np.ndarray:
    """Primeiro quadro de um arquivo PGM (P5, 16 bits) com vários quadros concatenados."""
    data = gzip.decompress(raw)
    m = re.match(rb"P5\s+(?:#[^\n]*\n\s*)*(\d+)\s+(\d+)\s+(\d+)\s", data)
    if not m:
        raise ValueError("cabeçalho PGM inesperado")
    w, h, maxval = (int(x) for x in m.groups())
    dtype = ">u2" if maxval > 255 else "u1"
    return np.frombuffer(data, dtype=dtype, count=w * h, offset=m.end()).reshape(h, w).astype(np.uint16)


def main(site: str, start: str, end: str, n: int, out: str) -> None:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    lat, lon = STATIONS[site]
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        days = pd.date_range(start, end, freq="D", inclusive="left", tz="UTC")
        with ThreadPoolExecutor(WORKERS) as pool:
            clouds = pd.concat(pool.map(lambda d: cloud_day(client, site, d), days), ignore_index=True)
        if clouds.empty:
            print(f"sem classificação de nuvens para {site} entre {start} e {end}", flush=True)
            return
        clouds.to_parquet(out_dir / "clouds.parquet")
        sun = astro.sun_altitude(clouds["time"], lat, lon)
        ok = clouds[(clouds["label"] == "not_cloudy") & (clouds["conf_clear"] >= 90) & (sun < -12)]
        print(f"{len(clouds)} horários classificados, {len(ok)} escuros e limpos", flush=True)
        picks = ok.sample(n=min(n, len(ok)), random_state=0).sort_values("time")

        def fetch(r):
            name = f"{r.time:%Y%m%d_%H%M}_{site}_{r.imager}_full.pgm.gz"
            url = f"{CALGARY}/stream0/{r.time:%Y/%m/%d}/{site}_{r.imager}/ut{r.time:%H}/{name}"
            try:
                resp = client.get(url)
                resp.raise_for_status()
                return r, first_frame(resp.content)
            except (httpx.HTTPError, ValueError) as exc:
                print(f"falhou {name}: {exc}", flush=True)
                return r, None

        frames, done = {}, []
        with ThreadPoolExecutor(WORKERS) as pool:
            for r, frame in pool.map(fetch, picks.itertuples()):
                if frame is None:
                    continue
                frames[f"{r.time:%Y%m%d_%H%M}"] = frame
                done.append({"time": r.time, "site": site, "imager": r.imager, "conf_clear": r.conf_clear})
                if len(done) % 50 == 0:
                    print(f"{len(done)} imagens", flush=True)
    np.savez_compressed(out_dir / "frames.npz", **frames)
    pd.DataFrame(done).to_parquet(out_dir / "samples.parquet")
    print(f"fim: {len(done)} imagens", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
