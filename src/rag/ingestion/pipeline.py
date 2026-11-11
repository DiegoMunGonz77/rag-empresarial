"""Ingesta: directorio de documentos -> chunks con metadatos.

La indexación en el vector store se añade en la etapa 3.
"""
from __future__ import annotations

from pathlib import Path

from rag.config import get_settings

from .chunker import Chunk, chunk_pages
from .loaders import load_directory


def ingest_directory(
    directory: Path | None = None,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    settings = get_settings()
    pages = load_directory(directory or settings.data_dir)
    return chunk_pages(
        pages,
        chunk_size if chunk_size is not None else settings.chunk_size,
        overlap if overlap is not None else settings.chunk_overlap,
    )
