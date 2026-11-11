"""Servicio RAG: recuperar -> construir prompt -> LLM -> respuesta con fuentes estructuradas."""
from __future__ import annotations

from dataclasses import dataclass, field

from rag.llm.base import LLMProvider
from rag.prompts import NOT_FOUND, SYSTEM_PROMPT, build_prompt
from rag.retriever import Passage, Retriever, SourceRef


@dataclass(frozen=True)
class Answer:
    text: str
    found: bool  # False => "no encontré esa información"
    # Fuentes salidas del retriever (no del texto del modelo); vacías si found es False
    sources: tuple[SourceRef, ...] = ()
    passages: tuple[Passage, ...] = field(default=(), repr=False)


def _is_not_found(text: str) -> bool:
    return text.strip().lower().startswith("no encontré esa información")


class RagService:
    def __init__(self, retriever: Retriever, llm: LLMProvider):
        self.retriever = retriever
        self.llm = llm

    def ask(self, question: str) -> Answer:
        question = question.strip()
        if not question:
            raise ValueError("La pregunta no puede estar vacía")

        passages = self.retriever.retrieve(question)
        if not passages:  # contexto insuficiente: ni siquiera se llama al LLM
            return Answer(NOT_FOUND, found=False)

        text = self.llm.generate(build_prompt(question, passages), system=SYSTEM_PROMPT).strip()
        if _is_not_found(text):
            return Answer(NOT_FOUND, found=False)

        sources: list[SourceRef] = []
        for p in passages:
            for ref in p.sources:
                if ref not in sources:
                    sources.append(ref)
        return Answer(text, found=True, sources=tuple(sources), passages=tuple(passages))
