import pytest

from rag.ingestion.chunker import Chunk
from rag.llm.base import LLMProvider
from rag.prompts import NOT_FOUND, SYSTEM_PROMPT, build_prompt
from rag.rag_service import RagService
from rag.retriever import Retriever, SourceRef, are_near_duplicates, group_duplicates
from rag.vectorstore import SearchResult, VectorStore
from tests.fakes import FakeEmbedder

AIRE = "La tecla 2 permite poner en marcha o detener el aire acondicionado del habitaculo"
AIRE_ESPACIOS = "La tecla 2  permite poner en marcha  o detener el aire\nacondicionado del habitaculo"
PRESION_19 = "La presion de los neumaticos delanteros es de 2.2 bar en carga normal del vehiculo"
PRESION_23 = "La presion de los neumaticos delanteros es de 2.5 bar en carga normal del vehiculo"


class FakeLLM(LLMProvider):
    name, model = "fake", "fake-1"

    def __init__(self, reply="Respuesta del modelo [x.pdf, p. 99]"):
        self.reply, self.calls = reply, []

    def generate(self, prompt, system=None):
        self.calls.append({"prompt": prompt, "system": system})
        return self.reply


@pytest.fixture
def store(tmp_path):
    s = VectorStore(tmp_path / "chroma", FakeEmbedder())
    s.add_chunks([Chunk(AIRE, "A2019.pdf", 68, 0)], "f")
    s.add_chunks([Chunk(AIRE_ESPACIOS, "A2023.pdf", 88, 0)], "f")
    s.add_chunks([Chunk(PRESION_19, "A2019.pdf", 120, 0)], "f")
    s.add_chunks([Chunk(PRESION_23, "A2023.pdf", 140, 0)], "f")
    s.add_chunks([Chunk("receta de paella con arroz y azafran", "cocina.txt", 1, 0)], "f")
    return s


def _service(store, llm=None, **kw):
    llm = llm or FakeLLM()
    return RagService(Retriever(store, top_k=5, min_relevance=0.3, **kw), llm), llm


# --- agrupación de duplicados --------------------------------------------------


def test_near_duplicates_ignores_whitespace_and_case():
    assert are_near_duplicates(AIRE, AIRE_ESPACIOS)
    assert are_near_duplicates("Hola  Mundo", "hola mundo")


def test_page_labels_do_not_prevent_grouping():
    # Los manuales empiezan cada página con su etiqueta ("1.42" vs "1.46")
    a = "1.42\nPara abrir la tapa A tire de la palanca 1 situada a la izquierda. Capacidad 38 litros."
    b = "1.46\nPara abrir la tapa A tire de la palanca 1 situada a la izquierda. Capacidad 38 litros."
    assert are_near_duplicates(a, b)
    # ...pero un dato real distinto sigue separándolos
    assert not are_near_duplicates(a, b.replace("38", "40"))


def test_section_counters_do_not_prevent_grouping():
    a = "calefaccion, ventilacion, aire acondicionado (3/3)\nLa tecla 2 enciende el aire. 38 litros"
    b = "calefaccion, ventilacion, aire acondicionado\nLa tecla 2 enciende el aire. 38 litros"
    assert are_near_duplicates(a, b)
    assert not are_near_duplicates(a, b.replace("38", "45"))


def test_changed_number_is_not_duplicate():
    assert not are_near_duplicates(PRESION_19, PRESION_23)


def test_different_text_is_not_duplicate():
    assert not are_near_duplicates(AIRE, "cambio de aceite cada diez mil kilometros del motor")


def test_group_duplicates_merges_sources_and_keeps_best_text():
    results = [
        SearchResult(AIRE, "A2019.pdf", 68, 0.9),
        SearchResult(PRESION_19, "A2019.pdf", 120, 0.8),
        SearchResult(AIRE_ESPACIOS, "A2023.pdf", 88, 0.88),
    ]
    passages = group_duplicates(results)
    assert len(passages) == 2
    assert passages[0].text == AIRE and passages[0].score == 0.9
    assert passages[0].sources == (SourceRef("A2019.pdf", 68), SourceRef("A2023.pdf", 88))


# --- retriever -----------------------------------------------------------------


def test_retriever_groups_across_files_and_applies_threshold(store):
    passages = Retriever(store, top_k=5, min_relevance=0.3).retrieve("tecla aire acondicionado")
    assert passages[0].sources == (SourceRef("A2019.pdf", 68), SourceRef("A2023.pdf", 88))
    assert all("paella" not in p.text for p in passages)  # bajo el umbral


def test_retriever_keeps_both_editions_when_data_differs(store):
    passages = Retriever(store, top_k=5, min_relevance=0.3).retrieve("presion neumaticos delanteros")
    texts = {p.text for p in passages}
    assert PRESION_19 in texts and PRESION_23 in texts


def test_retriever_returns_nothing_below_threshold(store):
    assert Retriever(store, min_relevance=0.99).retrieve("cualquier cosa rara") == []


def test_retriever_respects_top_k(store):
    assert len(Retriever(store, top_k=1, min_relevance=0.0).retrieve("presion aire")) == 1


# --- servicio ------------------------------------------------------------------


def test_answer_found_returns_sources_from_retriever_not_from_model(store):
    service, llm = _service(store)
    answer = service.ask("como se enciende el aire acondicionado con la tecla")
    assert answer.found
    # El modelo citó "x.pdf, p. 99", pero las fuentes salen del retriever
    assert "x.pdf" in answer.text
    assert SourceRef("x.pdf", 99) not in answer.sources
    assert answer.sources[:2] == (SourceRef("A2019.pdf", 68), SourceRef("A2023.pdf", 88))
    assert len(llm.calls) == 1


def test_duplicate_text_sent_to_llm_once_with_all_origins(store):
    service, llm = _service(store)
    service.ask("tecla aire acondicionado")
    prompt = llm.calls[0]["prompt"]
    assert prompt.count("permite poner en marcha") == 1
    assert "A2019.pdf, p. 68" in prompt and "A2023.pdf, p. 88" in prompt


def test_no_relevant_context_skips_llm_and_says_not_found(store):
    llm = FakeLLM()
    service = RagService(Retriever(store, min_relevance=0.99), llm)
    answer = service.ask("pregunta sin relacion alguna")
    assert not answer.found and answer.text == NOT_FOUND and answer.sources == ()
    assert llm.calls == []


def test_model_saying_not_found_clears_sources(store):
    service, _ = _service(store, FakeLLM(reply=NOT_FOUND))
    answer = service.ask("tecla aire acondicionado")
    assert not answer.found and answer.text == NOT_FOUND and answer.sources == ()


def test_empty_question_rejected(store):
    service, _ = _service(store)
    with pytest.raises(ValueError):
        service.ask("   ")


def test_prompt_contains_citation_rules_and_question(store):
    passages = Retriever(store, min_relevance=0.3).retrieve("tecla aire")
    prompt = build_prompt("¿Cómo enciendo el aire?", passages)
    assert "¿Cómo enciendo el aire?" in prompt and "Fragmento 1" in prompt
    assert "[archivo, p. N]" in SYSTEM_PROMPT and NOT_FOUND in SYSTEM_PROMPT
