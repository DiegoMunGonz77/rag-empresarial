"""Retriever: búsqueda semántica + umbral de relevancia + agrupación de duplicados."""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from rag.config import get_settings
from rag.vectorstore import SearchResult, VectorStore

_CANDIDATE_FACTOR = 3  # se piden más candidatos que top_k para tener margen tras agrupar
_NUMBERS = re.compile(r"\d+(?:[.,]\d+)*")


@dataclass(frozen=True)
class SourceRef:
    """Procedencia de un pasaje: archivo y página."""

    file: str
    page: int

    def __str__(self) -> str:
        return f"{self.file}, p. {self.page}"


@dataclass(frozen=True)
class Passage:
    """Pasaje de contexto. Si el mismo texto aparece en varios archivos, `sources` los lista todos."""

    text: str
    sources: tuple[SourceRef, ...]
    score: float  # mejor similitud del grupo


_PAGE_LABEL_LINE = re.compile(r"^\s*\d{1,2}[.,]\d{1,3}\s*$", re.MULTILINE)


_SECTION_COUNTER = re.compile(r"\(\s*\d{1,2}\s*/\s*\d{1,2}\s*\)")


def _strip_page_labels(text: str) -> str:
    """Quita ruido de maquetación que cambia entre ediciones aunque el contenido sea el mismo:
    líneas con solo una etiqueta de página ("1.42", "5.16") y contadores de sección "(3/3)"."""
    return _SECTION_COUNTER.sub("", _PAGE_LABEL_LINE.sub("", text))


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", _strip_page_labels(text)).strip().lower()


def are_near_duplicates(a: str, b: str, threshold: float = 0.9) -> bool:
    """Casi idénticos: texto muy parecido Y los mismos números.

    La condición de números evita fusionar pasajes que cambian un dato entre ediciones
    (presiones, capacidades, plazos…): esos deben llegar al LLM por separado.
    """
    na, nb = _normalize(a), _normalize(b)
    if na == nb:
        return True
    if _NUMBERS.findall(na) != _NUMBERS.findall(nb):
        return False
    matcher = SequenceMatcher(None, na, nb, autojunk=False)
    return matcher.real_quick_ratio() >= threshold and matcher.ratio() >= threshold


def group_duplicates(results: list[SearchResult], threshold: float = 0.9) -> list[Passage]:
    """Agrupa resultados casi idénticos. `results` debe venir ordenado por relevancia
    descendente: el texto representante del grupo es el de mayor puntaje."""
    groups: list[tuple[SearchResult, list[SourceRef]]] = []
    for r in results:
        ref = SourceRef(r.source, r.page)
        for rep, refs in groups:
            if are_near_duplicates(rep.text, r.text, threshold):
                if ref not in refs:
                    refs.append(ref)
                break
        else:
            groups.append((r, [ref]))
    return [Passage(rep.text, tuple(refs), rep.score) for rep, refs in groups]


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        top_k: int | None = None,
        min_relevance: float | None = None,
        dedup_similarity: float | None = None,
    ):
        s = get_settings()
        self.store = store
        self.top_k = s.top_k if top_k is None else top_k
        self.min_relevance = s.min_relevance if min_relevance is None else min_relevance
        self.dedup_similarity = s.dedup_similarity if dedup_similarity is None else dedup_similarity

    def retrieve(self, query: str) -> list[Passage]:
        """Hasta `top_k` pasajes con similitud >= umbral, sin texto repetido entre archivos."""
        hits = self.store.search(query, self.top_k * _CANDIDATE_FACTOR)
        relevant = [h for h in hits if h.score >= self.min_relevance]
        return group_duplicates(relevant, self.dedup_similarity)[: self.top_k]
