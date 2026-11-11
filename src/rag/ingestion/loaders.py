"""Extracción de texto de PDF, DOCX y TXT en unidades "página" con metadatos."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from docx import Document
from pypdf import PdfReader

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
_FORM_FEED = "\f"


@dataclass(frozen=True)
class Page:
    """Texto de una página con su origen."""

    text: str
    source: str  # nombre del archivo
    page: int  # 1-indexado


def load_pdf(path: Path) -> list[Page]:
    reader = PdfReader(str(path))
    pages = []
    for number, pdf_page in enumerate(reader.pages, start=1):
        text = (pdf_page.extract_text() or "").strip()
        if text:
            pages.append(Page(text, path.name, number))
    return pages


def load_docx(path: Path) -> list[Page]:
    """Word no tiene páginas fijas: se numeran según los saltos de página explícitos.

    Sin saltos explícitos todo el documento cuenta como página 1.
    Se incluyen párrafos y tablas en orden de aparición.
    """
    doc = Document(str(path))
    pages: list[list[str]] = [[]]

    def add_paragraph(paragraph) -> None:
        if paragraph._p.xpath('.//w:br[@w:type="page"]'):
            before, *rest = _split_on_page_break(paragraph)
            if before:
                pages[-1].append(before)
            for chunk in rest:
                pages.append([chunk] if chunk else [])
        elif paragraph.text.strip():
            pages[-1].append(paragraph.text.strip())

    body = doc.element.body
    for child in body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            from docx.text.paragraph import Paragraph

            add_paragraph(Paragraph(child, doc))
        elif tag == "tbl":
            from docx.table import Table

            for row in Table(child, doc).rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    pages[-1].append(" | ".join(cells))

    result = []
    for number, lines in enumerate(pages, start=1):
        text = "\n".join(lines).strip()
        if text:
            result.append(Page(text, path.name, number))
    return result


def _split_on_page_break(paragraph) -> list[str]:
    """Divide el texto de un párrafo en cada salto de página explícito."""
    parts, current = [], []
    for run in paragraph.runs:
        for node in run._r.iterchildren():
            tag = node.tag.rsplit("}", 1)[-1]
            if tag == "t":
                current.append(node.text or "")
            elif tag == "br" and node.get(
                "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}type"
            ) == "page":
                parts.append("".join(current).strip())
                current = []
    parts.append("".join(current).strip())
    return parts


def load_txt(path: Path) -> list[Page]:
    """TXT: una página por defecto; el carácter form feed (\\f) separa páginas."""
    raw = path.read_bytes()
    try:
        content = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        content = raw.decode("latin-1")
    pages = []
    for number, text in enumerate(content.split(_FORM_FEED), start=1):
        text = text.strip()
        if text:
            pages.append(Page(text, path.name, number))
    return pages


_LOADERS = {".pdf": load_pdf, ".docx": load_docx, ".txt": load_txt}


def load_document(path: Path) -> list[Page]:
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in _LOADERS:
        raise ValueError(f"Formato no soportado: {ext} ({path.name})")
    return _LOADERS[ext](path)


def load_directory(directory: Path) -> list[Page]:
    """Carga todos los documentos soportados de un directorio (no recursivo).

    Un archivo ilegible se registra y se omite; no detiene la ingesta.
    """
    pages: list[Page] = []
    for path in sorted(Path(directory).iterdir()):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS or not path.is_file():
            continue
        try:
            pages.extend(load_document(path))
        except Exception:
            logger.exception("No se pudo leer %s; se omite", path.name)
    return pages
