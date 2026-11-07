"""Classificador de imagens all-sky: aurora, céu limpo sem aurora, ou outro (nuvem/Lua).

Treinado no OATH (Clausen & Nickisch, 2018; 5.824 imagens THEMIS rotuladas, CC BY 4.0).
Uma CNN pré-treinada (ResNet-18, ImageNet) extrai 512 características de cada imagem, e uma
regressão logística aprende as classes por cima. A validação separa as imagens por noite:
as fotos do OATH vêm em rajadas da mesma hora, e misturar a mesma noite no treino e no
teste inflaria o acerto.

Uso:
  python classifier.py train <pasta_oath> <saída>             # valida e salva o modelo
  python classifier.py apply <saída> <pasta_frames> <arquivo>  # classifica imagens .npy brutas
  python classifier.py finetune <pasta_oath>                  # compara: CNN ajustada (fine-tuning)
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLASSES = {"arc": "aurora", "diffuse": "aurora", "discrete": "aurora", "noaurora": "clear", "cloudy": "other", "moon": "other"}
# Recorte das imagens brutas THEMIS (256x256): o campo útil, sem as árvores do horizonte, tem
# raio ~100 px; no OATH ele ocupa um círculo de raio ~108 px.
RAW_RADIUS = 100
OATH_RADIUS = 108


def _csv_body(path: Path, header: str) -> str:
    lines = path.read_text().splitlines()
    return "\n".join(lines[next(i for i, line in enumerate(lines) if line.startswith(header)):])


def load_oath(folder: Path) -> pd.DataFrame:
    labels = pd.read_csv(io.StringIO(_csv_body(folder / "classifications/classifications.csv", "class2")), dtype={"picNum": str})
    rows = []
    for line in (folder / "images/files_origin.csv").read_text().splitlines():
        m = re.match(r"(\w{4}) (\d\d/\w{3}/\d{4} [\d:.]+),images/\w+/(\d{5})\.png", line.strip())
        if m:
            rows.append((m[1], pd.to_datetime(m[2], format="%d/%b/%Y %H:%M:%S.%f", utc=True), m[3]))
    origin = pd.DataFrame(rows, columns=["site", "time", "picNum"])
    df = origin.merge(labels, on="picNum")
    df["label"] = df["picType"].map(CLASSES)
    df["night"] = df["site"] + "_" + (df["time"] - pd.Timedelta(hours=12)).dt.strftime("%Y%m%d")
    df["path"] = [folder / "images/cropped_scaled" / f"{n}.png" for n in df["picNum"]]
    return df


def stretch(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(a[mask], [1, 99])
    out = np.clip((a - lo) / (hi - lo + 1e-9), 0, 1) * 255
    out[~mask] = 0
    return out.astype(np.uint8)


def preprocess_raw(frame: np.ndarray) -> np.ndarray:
    """Imagem bruta THEMIS (16 bits) → formato do OATH (8 bits, círculo recortado e esticado)."""
    c = 128
    crop = frame[c - RAW_RADIUS : c + RAW_RADIUS, c - RAW_RADIUS : c + RAW_RADIUS].astype(float)
    size = 2 * OATH_RADIUS
    resized = np.array(Image.fromarray(crop).resize((size, size), Image.BILINEAR))
    canvas = np.zeros((256, 256))
    off = (256 - size) // 2
    canvas[off : off + size, off : off + size] = resized
    yy, xx = np.mgrid[:256, :256]
    mask = np.hypot(yy - 127.5, xx - 127.5) <= OATH_RADIUS
    return stretch(canvas, mask)


class Features:
    def __init__(self):
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        model = torchvision.models.resnet18(weights=weights)
        model.fc = torch.nn.Identity()
        self.model = model.eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, images: list[np.ndarray], batch: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(images), batch):
            x = torch.tensor(np.stack(images[i : i + batch]), dtype=torch.float32) / 255
            x = x.unsqueeze(1).repeat(1, 3, 1, 1)
            x = torch.nn.functional.interpolate(x, size=224, mode="bilinear", align_corners=False)
            out.append(self.model((x - self.mean) / self.std).numpy())
        return np.concatenate(out)


def train(oath: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    df = load_oath(oath)
    images = [np.array(Image.open(p))[..., 1] for p in df["path"]]  # o OATH guarda a imagem no canal verde
    x = Features()(images)
    y = df["label"].to_numpy()
    groups = df["night"].to_numpy()

    pred = np.empty(len(y), dtype=object)
    prob = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(x, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))
        clf.fit(x[tr], y[tr])
        pred[te] = clf.predict(x[te])
        prob[te] = clf.predict_proba(x[te])[:, list(clf.classes_).index("aurora")]

    labels = ["aurora", "clear", "other"]
    cm = confusion_matrix(y, pred, labels=labels)
    print(f"{df['night'].nunique()} noites, {len(df)} imagens. Validação por noite (5 dobras):")
    print(f"acerto geral: {(pred == y).mean():.3f}")
    print(pd.DataFrame(cm, index=[f"real {k}" for k in labels], columns=[f"prev {k}" for k in labels]).to_string())
    clear_sky = np.isin(y, ["aurora", "clear"])
    binary = (pred[clear_sky] == "aurora") == (y[clear_sky] == "aurora")
    print(f"céu limpo, aurora × sem aurora: acerto {binary.mean():.3f}")

    final = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))
    final.fit(x, y)
    joblib.dump(final, out / "aurora_classifier.joblib")
    pd.DataFrame({"night": groups, "label": y, "pred": pred, "p_aurora": prob}).to_csv(out / "oath_cv.csv", index=False)


def apply(model_dir: Path, frames: Path, out_file: Path) -> None:
    clf = joblib.load(model_dir / "aurora_classifier.joblib")
    files = sorted(frames.glob("*.npy"))
    images = [preprocess_raw(np.load(f)) for f in files]
    probs = clf.predict_proba(Features()(images))
    result = pd.DataFrame(probs, columns=[f"p_{c}" for c in clf.classes_])
    result.insert(0, "file", [f.stem for f in files])
    result["pred"] = clf.classes_[probs.argmax(1)]
    result.to_csv(out_file, index=False)
    print(result["pred"].value_counts().to_string())




def finetune_cv(oath: Path, epochs: int = 4) -> None:
    """Opção B: ajusta a última camada residual da ResNet-18 (fine-tuning), mesma validação por noite."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    df = load_oath(oath)
    images = np.stack([np.array(Image.open(p))[..., 1] for p in df["path"]])
    labels = ["aurora", "clear", "other"]
    y = np.array([labels.index(v) for v in df["label"]])
    groups = df["night"].to_numpy()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def batch_tensor(idx, augment):
        x = torch.tensor(images[idx], dtype=torch.float32, device=device).unsqueeze(1) / 255
        if augment:  # rotação aleatória: a orientação do céu não muda a classe
            k = int(np.random.randint(4))
            x = torch.rot90(x, k, dims=(2, 3))
            if np.random.rand() < 0.5:
                x = torch.flip(x, dims=(3,))
        x = torch.nn.functional.interpolate(x.repeat(1, 3, 1, 1), size=224, mode="bilinear", align_corners=False)
        return (x - mean) / std

    pred = np.zeros(len(y), dtype=int)
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(images, y, groups)):
        torch.manual_seed(fold)
        np.random.seed(fold)
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Linear(512, 3)
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith(("layer4", "fc"))
        model.to(device)
        counts = np.bincount(y[tr], minlength=3)
        weight = torch.tensor(len(tr) / (3 * counts), dtype=torch.float32, device=device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-2)
        for _ in range(epochs):
            model.train()
            order = tr[np.random.permutation(len(tr))]
            for i in range(0, len(tr), 32):
                idx = order[i : i + 32]
                opt.zero_grad()
                loss = loss_fn(model(batch_tensor(idx, True)), torch.tensor(y[idx], device=device))
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            for i in range(0, len(te), 128):
                idx = te[i : i + 128]
                pred[idx] = model(batch_tensor(idx, False)).argmax(1).cpu().numpy()
        print(f"dobra {fold + 1}: acerto {(pred[te] == y[te]).mean():.3f}", flush=True)

    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    print(f"fine-tuning, acerto geral: {(pred == y).mean():.3f} (dispositivo: {device})")
    print(pd.DataFrame(cm, index=[f"real {k}" for k in labels], columns=[f"prev {k}" for k in labels]).to_string())
    clear_sky = np.isin(y, [0, 1])
    print(f"céu limpo, aurora × sem aurora: acerto {((pred[clear_sky] == 0) == (y[clear_sky] == 0)).mean():.3f}")


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(Path(sys.argv[2]), Path(sys.argv[3]))
    elif sys.argv[1] == "finetune":
        finetune_cv(Path(sys.argv[2]))
    else:
        apply(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
