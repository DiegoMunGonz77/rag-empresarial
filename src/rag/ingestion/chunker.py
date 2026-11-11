"""Chunking por caracteres con overlap. Los chunks nunca cruzan páginas,
de modo que la cita (archivo, página) de cada chunk es exacta."""
from __future__ import annotations

from dataclasses import dataclass

from .loaders import Page


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    page: int
    chunk_index: int  # posición dentro de la página

    @property
    def id(self) -> str:
        return f"{self.source}::p{self.page}::c{self.chunk_index}"

    @property
    def metadata(self) -> dict:
        return {"source": self.source, "page": self.page, "chunk_index": self.chunk_index}


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Divide `text` en trozos de a lo sumo `chunk_size` caracteres con `overlap` de solape.

    Prefiere cortar en salto de párrafo, de línea, fin de frase o espacio
    (en ese orden) dentro de la segunda mitad de la ventana.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size debe ser > 0")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap debe cumplir 0 <= overlap < chunk_size")

    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            end = _find_break(text, start, end)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        # Avanza dejando `overlap` caracteres; siempre progresa al menos 1.
        start = max(end - overlap, start + 1)
    return chunks


def _find_break(text: str, start: int, end: int) -> int:
    window_min = start + (end - start) // 2
    for sep in ("\n\n", "\n", ". ", " "):
        idx = text.rfind(sep, window_min, end)
        if idx != -1:
            return idx + len(sep)
    return end


def chunk_pages(pages: list[Page], chunk_size: int = 1000, overlap: int = 200) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        for i, piece in enumerate(split_text(page.text, chunk_size, overlap)):
            chunks.append(Chunk(piece, page.source, page.page, i))
    return chunks
