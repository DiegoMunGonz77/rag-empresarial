"""Ingesta: directorio de documentos -> chunks -> vector store (indexación incremental)."""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

from rag.config import get_settings
from rag.vectorstore import VectorStore

from .chunker import Chunk, chunk_pages
from .loaders import SUPPORTED_EXTENSIONS, load_directory, load_document

logger = logging.getLogger(__name__)


def ingest_directory(
    directory: Path | None = None,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """Solo extracción + chunking (sin indexar)."""
    settings = get_settings()
    pages = load_directory(directory or settings.data_dir)
    return chunk_pages(
        pages,
        chunk_size if chunk_size is not None else settings.chunk_size,
        overlap if overlap is not None else settings.chunk_overlap,
    )


def file_fingerprint(path: Path, chunk_size: int, overlap: int) -> str:
    """Huella del archivo + parámetros de chunking: si cambia algo, se reindexa."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return f"{digest.hexdigest()}|{chunk_size}|{overlap}"


@dataclass
class IndexReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    chunks_indexed: int = 0


def index_directory(
    store: VectorStore,
    directory: Path | None = None,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> IndexReport:
    """Indexa de forma incremental: solo procesa archivos nuevos o modificados
    y elimina del índice los que ya no existen en el directorio."""
    settings = get_settings()
    directory = Path(directory or settings.data_dir)
    size = chunk_size if chunk_size is not None else settings.chunk_size
    ovl = overlap if overlap is not None else settings.chunk_overlap

    report = IndexReport()
    indexed = store.indexed_files()
    on_disk = {
        p.name: p
        for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    }

    for name in sorted(set(indexed) - set(on_disk)):
        store.delete_source(name)
        report.removed.append(name)

    for name, path in on_disk.items():
        fingerprint = file_fingerprint(path, size, ovl)
        if indexed.get(name) == fingerprint:
            report.unchanged.append(name)
            continue
        try:
            chunks = chunk_pages(load_document(path), size, ovl)
        except Exception:
            logger.exception("No se pudo procesar %s; se omite", name)
            report.failed.append(name)
            continue
        if name in indexed:
            store.delete_source(name)
        if chunks:
            store.add_chunks(chunks, fingerprint)
            report.chunks_indexed += len(chunks)
        (report.updated if name in indexed else report.added).append(name)
    return report
