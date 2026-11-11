import pytest

from rag.ingestion.chunker import Chunk
from rag.ingestion.pipeline import index_directory
from rag.vectorstore import EmbeddingModelMismatch, VectorStore
from tests.fakes import FakeEmbedder


@pytest.fixture
def store(tmp_path):
    return VectorStore(tmp_path / "chroma", FakeEmbedder())


def _chunks():
    return [
        Chunk("el aire acondicionado se enciende con la tecla dos", "a.pdf", 3, 0),
        Chunk("cambio de aceite del motor cada diez mil kilometros", "a.pdf", 7, 0),
        Chunk("politica de vacaciones quince dias al año", "b.txt", 1, 0),
    ]


def test_search_returns_most_relevant_with_metadata(store):
    store.add_chunks(_chunks(), "fp1")
    hits = store.search("como cambio el aceite del motor", top_k=2)
    assert hits[0].source == "a.pdf" and hits[0].page == 7
    assert hits[0].score > hits[1].score
    assert len(hits) == 2


def test_search_empty_store(store):
    assert store.search("lo que sea") == []


def test_upsert_is_idempotent(store):
    store.add_chunks(_chunks(), "fp1")
    store.add_chunks(_chunks(), "fp1")
    assert store.count() == 3


def test_persistence_across_instances(tmp_path):
    VectorStore(tmp_path / "c", FakeEmbedder()).add_chunks(_chunks(), "fp1")
    reopened = VectorStore(tmp_path / "c", FakeEmbedder())
    assert reopened.count() == 3
    assert reopened.indexed_files() == {"a.pdf": "fp1", "b.txt": "fp1"}


def test_delete_source(store):
    store.add_chunks(_chunks(), "fp1")
    store.delete_source("a.pdf")
    assert store.count() == 1
    assert set(store.indexed_files()) == {"b.txt"}


def test_embedding_model_mismatch_is_detected(tmp_path):
    VectorStore(tmp_path / "c", FakeEmbedder())

    class Other(FakeEmbedder):
        model_name = "otro-modelo"

    with pytest.raises(EmbeddingModelMismatch):
        VectorStore(tmp_path / "c", Other())


# --- indexación incremental ---------------------------------------------------


@pytest.fixture
def docs(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "uno.txt").write_text("frenos y neumaticos del vehiculo " * 5, encoding="utf-8")
    (d / "dos.txt").write_text("politica de vacaciones de la empresa " * 5, encoding="utf-8")
    return d


def test_incremental_indexing(store, docs):
    r1 = index_directory(store, docs, 100, 10)
    assert sorted(r1.added) == ["dos.txt", "uno.txt"] and r1.chunks_indexed > 0
    total = store.count()

    # Sin cambios: no se reprocesa nada
    r2 = index_directory(store, docs, 100, 10)
    assert sorted(r2.unchanged) == ["dos.txt", "uno.txt"]
    assert not (r2.added or r2.updated or r2.removed) and r2.chunks_indexed == 0
    assert store.count() == total

    # Archivo modificado: se reemplaza solo ese
    (docs / "uno.txt").write_text("bateria y luces del vehiculo " * 5, encoding="utf-8")
    r3 = index_directory(store, docs, 100, 10)
    assert r3.updated == ["uno.txt"] and r3.unchanged == ["dos.txt"]
    assert store.search("bateria luces", 1)[0].source == "uno.txt"
    assert all("frenos" not in h.text for h in store.search("frenos", 10) if h.source == "uno.txt")

    # Archivo nuevo y archivo eliminado
    (docs / "tres.txt").write_text("contrato de garantia extendida " * 5, encoding="utf-8")
    (docs / "dos.txt").unlink()
    r4 = index_directory(store, docs, 100, 10)
    assert r4.added == ["tres.txt"] and r4.removed == ["dos.txt"]
    assert set(store.indexed_files()) == {"uno.txt", "tres.txt"}


def test_changing_chunk_params_reindexes(store, docs):
    index_directory(store, docs, 100, 10)
    r = index_directory(store, docs, 60, 10)
    assert sorted(r.updated) == ["dos.txt", "uno.txt"]


def test_broken_file_does_not_stop_indexing(store, docs):
    (docs / "roto.pdf").write_bytes(b"no es un pdf")
    r = index_directory(store, docs, 100, 10)
    assert r.failed == ["roto.pdf"]
    assert sorted(r.added) == ["dos.txt", "uno.txt"]
