"""Vector store sobre ChromaDB persistente (distancia coseno)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from rag.embeddings import Embedder
from rag.ingestion.chunker import Chunk

COLLECTION_NAME = "documentos"
_BATCH = 128


@dataclass(frozen=True)
class SearchResult:
    text: str
    source: str
    page: int
    score: float  # similitud coseno en [-1, 1]; mayor = más relevante


class EmbeddingModelMismatch(RuntimeError):
    pass


class VectorStore:
    def __init__(self, persist_dir: Path | str, embedder: Embedder):
        self.embedder = embedder
        self._client = chromadb.PersistentClient(
            path=str(persist_dir), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._collection = self._client.get_or_create_collection(
            COLLECTION_NAME,
            metadata={"hnsw:space": "cosine", "embedding_model": embedder.model_name},
        )
        stored = (self._collection.metadata or {}).get("embedding_model")
        if stored and stored != embedder.model_name:
            raise EmbeddingModelMismatch(
                f"La base vectorial se creó con '{stored}' y ahora se usa "
                f"'{embedder.model_name}'. Borra la carpeta de la base para reindexar."
            )

    def count(self) -> int:
        return self._collection.count()

    def indexed_files(self) -> dict[str, str]:
        """Devuelve {archivo: fingerprint} de lo indexado."""
        data = self._collection.get(include=["metadatas"])
        return {m["source"]: m["fingerprint"] for m in data["metadatas"]}

    def delete_source(self, source: str) -> None:
        self._collection.delete(where={"source": source})

    def add_chunks(self, chunks: list[Chunk], fingerprint: str) -> None:
        for i in range(0, len(chunks), _BATCH):
            batch = chunks[i : i + _BATCH]
            self._collection.upsert(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                embeddings=self.embedder.embed_passages([c.text for c in batch]),
                metadatas=[{**c.metadata, "fingerprint": fingerprint} for c in batch],
            )

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if self.count() == 0:
            return []
        res = self._collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        return [
            SearchResult(doc, meta["source"], int(meta["page"]), 1.0 - dist)
            for doc, meta, dist in zip(
                res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]
