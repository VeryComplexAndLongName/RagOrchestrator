from __future__ import annotations

import json
from typing import Iterable
from urllib import request


def _open_no_proxy(req: request.Request, timeout: int) -> bytes:
    opener = request.build_opener(request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as response:
        return response.read()


class OllamaEmbedder:
    """Production embedder backed by local Ollama /api/embed endpoint."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout_seconds: int = 60,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._dimensions: int | None = None

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            self._dimensions = len(self.embed("dimension probe"))
        return self._dimensions

    def embed(self, text: str) -> list[float]:
        payload = {
            "model": self.model,
            "input": text,
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self.base_url}/api/embed",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        body = json.loads(_open_no_proxy(req, timeout=self.timeout_seconds).decode("utf-8"))

        embeddings = body.get("embeddings") or []
        if not embeddings and body.get("embedding"):
            embeddings = [body["embedding"]]
        if not embeddings:
            return []
        vector = embeddings[0]
        if self._dimensions is None:
            self._dimensions = len(vector)
        return [float(value) for value in vector]

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        text_list = list(texts)
        if not text_list:
            return []

        payload = {
            "model": self.model,
            "input": text_list,
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self.base_url}/api/embed",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        body = json.loads(_open_no_proxy(req, timeout=self.timeout_seconds).decode("utf-8"))

        embeddings = body.get("embeddings") or []
        if not embeddings and body.get("embedding"):
            embeddings = [body["embedding"]]
        if not embeddings:
            return []
        if self._dimensions is None:
            self._dimensions = len(embeddings[0])
        return [[float(value) for value in vector] for vector in embeddings]

    @classmethod
    def list_models(cls, base_url: str = "http://localhost:11434", timeout_seconds: int = 20) -> list[str]:
        req = request.Request(
            url=f"{base_url.rstrip('/')}/api/tags",
            headers={"Content-Type": "application/json"},
            method="GET",
        )
        body = json.loads(_open_no_proxy(req, timeout=timeout_seconds).decode("utf-8"))
        return [str(item.get("name")) for item in body.get("models", []) if item.get("name")]
