"""Embeddings locales con sentence-transformers.

El modelo por defecto (multilingual-e5-base) tiene ventana de 512 tokens, suficiente para
chunks de ~1000 caracteres. Los modelos E5 exigen prefijos "query: " / "passage: ".
"""
from __future__ import annotations

from typing import Protocol

from rag.config import get_settings


class Embedder(Protocol):
    model_name: str

    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str | None = None, batch_size: int = 32):
        self.model_name = model_name or get_settings().embedding_model
        self.batch_size = batch_size
        self._model = None
        self._is_e5 = "e5" in self.model_name.lower()

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def max_tokens(self) -> int:
        return int(self.model.max_seq_length)

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer(text, add_special_tokens=True)["input_ids"])

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        if self._is_e5:
            texts = [f"passage: {t}" for t in texts]
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        if self._is_e5:
            text = f"query: {text}"
        return self._encode([text])[0]
