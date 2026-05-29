from __future__ import annotations

from typing import Iterable, cast

from ragflow_orchestrator.errors import ProviderDependencyError


class HFEmbedder:
    """Hugging Face embedder via sentence-transformers."""

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ProviderDependencyError(
                "HFEmbedder requires optional dependency 'sentence-transformers'. "
                "Install with: pip install -e .[hf]"
            ) from exc

        self.model_name = model
        self.batch_size = max(1, int(batch_size))
        self.normalize_embeddings = normalize_embeddings
        model_kwargs = {"device": device} if device else {}
        self._model = SentenceTransformer(model_name_or_path=model, **model_kwargs)
        self._dimensions = int(self._model.get_sentence_embedding_dimension())

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, text: str) -> list[float]:
        vector = self._model.encode(
            text,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        return cast(list[float], vector.astype("float32").tolist())

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        text_list = list(texts)
        if not text_list:
            return []
        vectors = self._model.encode(
            text_list,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return cast(list[list[float]], vectors.astype("float32").tolist())
