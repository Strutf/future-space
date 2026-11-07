"""Backends de armazenamento. Todos trabalham com chaves no formato 'json/kp/2026.json'.

- LocalStorage: pasta no disco (desenvolvimento).
- HuggingFaceStorage: repositório de dataset no Hugging Face (gratuito, versionado,
  feito para treino de IA). As escritas são acumuladas e enviadas num único commit.
- S3Storage: qualquer serviço compatível com S3 (Cloudflare R2, Backblaze B2...).
- HttpStorage: somente leitura via HTTP, usado pela API em produção.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path

import httpx


class Storage(ABC):
    @abstractmethod
    def read(self, key: str) -> bytes | None:
        """Retorna o conteúdo ou None se a chave não existir."""

    def write(self, key: str, data: bytes) -> None:
        raise NotImplementedError(f"{type(self).__name__} é somente leitura")

    def commit(self, message: str) -> None:
        """Confirma as escritas pendentes. Backends sem transação não fazem nada."""

    def data_revision(self) -> str | None:
        """Identificador da versão atual dos dados, quando o backend tem histórico."""
        return None


class LocalStorage(Storage):
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def read(self, key: str) -> bytes | None:
        path = self.root / key
        return path.read_bytes() if path.exists() else None

    def write(self, key: str, data: bytes) -> None:
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)


class HttpStorage(Storage):
    def __init__(self, base_url: str, headers: dict[str, str] | None = None, timeout: float = 30):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(headers=headers, timeout=timeout, follow_redirects=True)

    def read(self, key: str) -> bytes | None:
        resp = self.client.get(f"{self.base_url}/{key}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.content


def hf_resolve_url(repo_id: str, revision: str = "main") -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}"


class HuggingFaceStorage(Storage):
    def __init__(self, repo_id: str, token: str | None = None, revision: str = "main"):
        from huggingface_hub import HfApi

        self.repo_id = repo_id
        self.revision = revision
        self.api = HfApi(token=token)
        headers = {"Authorization": f"Bearer {token}"} if token else None
        self._http = HttpStorage(hf_resolve_url(repo_id, revision), headers=headers)
        self._pending: dict[str, bytes] = {}

    def data_revision(self) -> str | None:
        """Commit do dataset no Hugging Face: com ele dá para baixar os dados exatamente como estavam."""
        return self.api.repo_info(self.repo_id, repo_type="dataset", revision=self.revision).sha

    def read(self, key: str) -> bytes | None:
        if key in self._pending:
            return self._pending[key]
        return self._http.read(key)

    def write(self, key: str, data: bytes) -> None:
        self._pending[key] = data

    def commit(self, message: str) -> None:
        if not self._pending:
            return
        from huggingface_hub import CommitOperationAdd

        ops = [CommitOperationAdd(path_in_repo=k, path_or_fileobj=v) for k, v in self._pending.items()]
        self.api.create_commit(
            repo_id=self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            operations=ops,
            commit_message=message,
        )
        self._pending.clear()


class S3Storage(Storage):
    def __init__(self, bucket: str, endpoint_url: str | None = None, prefix: str = ""):
        import boto3

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.s3 = boto3.client("s3", endpoint_url=endpoint_url)

    def _k(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def read(self, key: str) -> bytes | None:
        try:
            return self.s3.get_object(Bucket=self.bucket, Key=self._k(key))["Body"].read()
        except self.s3.exceptions.NoSuchKey:
            return None

    def write(self, key: str, data: bytes) -> None:
        self.s3.put_object(Bucket=self.bucket, Key=self._k(key), Body=data)


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"variável de ambiente {name} é obrigatória para este backend")
    return value


def from_env() -> Storage:
    """Backend de escrita do pipeline, escolhido por FS_STORAGE (local | hf | s3)."""
    kind = os.environ.get("FS_STORAGE", "local")
    if kind == "local":
        return LocalStorage(os.environ.get("FS_DATA_DIR", "data"))
    if kind == "hf":
        return HuggingFaceStorage(_require("FS_HF_REPO"), token=os.environ.get("HF_TOKEN"))
    if kind == "s3":
        return S3Storage(
            _require("FS_S3_BUCKET"),
            endpoint_url=os.environ.get("FS_S3_ENDPOINT_URL"),
            prefix=os.environ.get("FS_S3_PREFIX", ""),
        )
    raise RuntimeError(f"FS_STORAGE inválido: {kind!r} (use local, hf ou s3)")


def reader_from_env() -> Storage:
    """Backend de leitura da API: FS_DATA_URL > FS_HF_REPO > pasta local FS_DATA_DIR."""
    # Timeout curto: sob carga, é melhor falhar rápido do que acumular requisições presas.
    if url := os.environ.get("FS_DATA_URL"):
        return HttpStorage(url, timeout=10)
    if repo := os.environ.get("FS_HF_REPO"):
        return HttpStorage(hf_resolve_url(repo), timeout=10)
    return LocalStorage(os.environ.get("FS_DATA_DIR", "data"))
