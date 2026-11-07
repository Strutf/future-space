"""Monta a base da calibração da aurora a partir das imagens coletadas pelo aurora-sample.

1. Treina o classificador de imagens no OATH (baixado e conferido pelo SHA-256).
2. Classifica as imagens THEMIS guardadas no Hugging Face (research/aurora/themis/*).
3. Junta a cada momento: Kp (GFZ), vento solar (OMNI), latitude magnética e hora local
   magnética (dipolo), altura do Sol e da Lua, fase da Lua.
4. Para as nuvens: classificação da U. Calgary (nublado ou não) × cobertura de nuvens da
   reanálise ERA5 (Open-Meteo) na mesma hora e lugar.

Saída: <pasta>/aurora_samples.parquet e <pasta>/cloud_samples.parquet.
Uso: python build_dataset.py <repo_hf> <pasta_trabalho> <pasta_saída>
"""

from __future__ import annotations

import hashlib
import io
import sys
import tarfile
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from futurespace.aurora import astro
from futurespace.aurora.visibility import mag_lat
from futurespace.ml.features import load_dataset
from futurespace.storage import HttpStorage, hf_resolve_url

sys.path.insert(0, str(Path(__file__).parent))
from classifier import Features, preprocess_raw, train
from sample_themis import STATIONS

OATH_URL = "http://tid.uio.no/plasma/oath/oath_v1.1_20181026.tgz"
OATH_SHA256 = "d23da3c15f1c18b2c2272c36e3a2fedda916de9eff5f51ea0657009b00c9d814"
POLE_LAT, POLE_LON = 80.8, -72.7  # o mesmo dipolo do backend e do mapa do site
RAD = np.pi / 180


def get_oath(work: Path) -> Path:
    folder = work / "oath"
    if not (folder / "oath_v1.1").exists():
        raw = httpx.get(OATH_URL, timeout=600, follow_redirects=True).content
        if hashlib.sha256(raw).hexdigest() != OATH_SHA256:
            raise RuntimeError("OATH com checksum diferente do publicado")
        with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
            tar.extractall(folder, filter="data")
    return folder / "oath_v1.1"


def magnetic_longitude(lat, lon) -> np.ndarray:
    """Longitude no sistema do dipolo (só diferenças importam: a hora local magnética)."""
    lat, lon = np.asarray(lat) * RAD, np.asarray(lon) * RAD
    xyz = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat) * np.ones_like(lon)])
    pl, pn = POLE_LAT * RAD, POLE_LON * RAD
    rz = np.array([[np.cos(pn), np.sin(pn), 0], [-np.sin(pn), np.cos(pn), 0], [0, 0, 1]])
    ry = np.array([[np.sin(pl), 0, -np.cos(pl)], [0, 1, 0], [np.cos(pl), 0, np.sin(pl)]])
    x, y, _ = np.tensordot(ry @ rz, xyz, axes=1)
    return np.arctan2(y, x) / RAD


def magnetic_local_time(times, lat, lon) -> np.ndarray:
    d = astro._days(times)
    ra, dec = astro._equatorial(astro._sun_ecliptic(d), 0.0, astro._obliquity(d))
    gmst = (280.46061837 + 360.98564736629 * d) % 360
    sub_lon = ((ra / RAD - gmst + 180) % 360) - 180
    sub_mlon = magnetic_longitude(dec / RAD, sub_lon)
    return (12 + (magnetic_longitude(lat, lon) - sub_mlon) / 15) % 24


def download_samples(repo: str, work: Path) -> list[Path]:
    files = HfApi().list_repo_files(repo, repo_type="dataset")
    folders = sorted({f.rsplit("/", 1)[0] for f in files if f.startswith("research/aurora/themis/") and f.endswith("frames.npz")})
    out = []
    for folder in folders:
        for name in ("frames.npz", "samples.parquet", "clouds.parquet"):
            hf_hub_download(repo, f"{folder}/{name}", repo_type="dataset", local_dir=work / "hf")
        out.append(work / "hf" / folder)
    return out


def context(df: pd.DataFrame, kp: pd.DataFrame, omni: pd.DataFrame) -> pd.DataFrame:
    """Kp, vento solar, coordenadas magnéticas, Sol e Lua para cada (site, time)."""
    lat = df["site"].map(lambda s: STATIONS[s][0]).to_numpy()
    lon = df["site"].map(lambda s: STATIONS[s][1]).to_numpy()
    out = df.copy()
    out["mag_lat"] = mag_lat(lat, lon)
    out["mlt"] = magnetic_local_time(out["time"], lat, lon)
    out["sun_altitude"] = astro.sun_altitude(out["time"], lat, lon)
    out["moon_altitude"] = astro.moon_altitude(out["time"], lat, lon)
    out["moon_illumination"] = astro.moon_illumination(out["time"])
    kp_s = kp.set_index("time")["kp"].sort_index()
    out["kp"] = kp_s.reindex(out["time"].dt.floor("3h")).to_numpy()
    o = omni.set_index("time")[["bz_gsm", "by_gsm", "speed", "density"]].sort_index()
    o = o[~o.index.duplicated()].reindex(out["time"].dt.floor("h"))
    for col in o.columns:
        out[col] = o[col].to_numpy()
    bt = np.hypot(out["by_gsm"], out["bz_gsm"])
    clock = np.arctan2(out["by_gsm"], out["bz_gsm"])
    out["newell"] = out["speed"] ** (4 / 3) * bt ** (2 / 3) * np.abs(np.sin(clock / 2)) ** (8 / 3)
    out["night"] = out["site"] + "_" + (out["time"] - pd.Timedelta(hours=12)).dt.strftime("%Y%m%d")
    return out


def era5_clouds(site: str, start: str, end: str) -> pd.DataFrame:
    lat, lon = STATIONS[site]
    resp = httpx.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
                "hourly": "cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high", "timezone": "GMT"},
        timeout=120,
    )
    resp.raise_for_status()
    h = pd.DataFrame(resp.json()["hourly"])
    h["time"] = pd.to_datetime(h["time"], utc=True)
    return h


def main(repo: str, work: str, out: str) -> None:
    work_dir, out_dir = Path(work), Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Classificador (imprime a validação por noite no log)
    oath = get_oath(work_dir)
    train(oath, work_dir / "clf")
    import joblib

    clf = joblib.load(work_dir / "clf" / "aurora_classifier.joblib")
    features = Features()

    # 2-3. Imagens coletadas → rótulos + contexto
    store = HttpStorage(hf_resolve_url(repo))
    kp, omni = load_dataset(store, "kp-gfz"), load_dataset(store, "omni-hourly")
    samples, clouds = [], []
    for folder in download_samples(repo, work_dir):
        meta = pd.read_parquet(folder / "samples.parquet")
        frames = np.load(folder / "frames.npz")
        keys = [f"{t:%Y%m%d_%H%M}" for t in meta["time"]]
        probs = clf.predict_proba(features([preprocess_raw(frames[k]) for k in keys]))
        for i, c in enumerate(clf.classes_):
            meta[f"p_{c}"] = probs[:, i]
        meta["label"] = clf.classes_[probs.argmax(1)]
        samples.append(meta)
        site, start, end = folder.name.split("_")
        cl = pd.read_parquet(folder / "clouds.parquet").assign(site=site)
        cl["hour"] = cl["time"].dt.floor("h")
        era = era5_clouds(site, start, (pd.Timestamp(end) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
        clouds.append(cl.merge(era.rename(columns={"time": "hour"}), on="hour", how="left"))
        print(f"{folder.name}: {len(meta)} imagens, {meta['label'].value_counts().to_dict()}", flush=True)

    aurora = context(pd.concat(samples, ignore_index=True), kp, omni)
    aurora.to_parquet(out_dir / "aurora_samples.parquet")
    cloud = context(pd.concat(clouds, ignore_index=True).drop(columns=["hour"]), kp, omni)
    cloud = cloud[cloud["sun_altitude"] < -12]
    cloud.to_parquet(out_dir / "cloud_samples.parquet")
    print(f"base: {len(aurora)} imagens rotuladas, {len(cloud)} horários de nuvem", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
