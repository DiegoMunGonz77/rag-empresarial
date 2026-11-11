import pytest

from rag.ingestion.chunker import chunk_pages, split_text
from rag.ingestion.loaders import Page
from rag.ingestion.pipeline import ingest_directory


def test_short_text_single_chunk():
    assert split_text("hola mundo", 100, 10) == ["hola mundo"]
    assert split_text("   ", 100, 10) == []


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"palabra{i}" for i in range(200))
    chunks = split_text(text, 100, 30)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    # Cada chunk comparte contexto con el anterior (overlap)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev.split()[-1] in nxt


def test_no_text_lost():
    text = " ".join(f"w{i}" for i in range(300))
    chunks = split_text(text, 80, 20)
    assert set(" ".join(chunks).split()) == set(text.split())


def test_hard_cut_without_separators_still_progresses():
    chunks = split_text("x" * 1000, 100, 20)
    assert all(len(c) <= 100 for c in chunks)
    assert len(chunks) >= 10


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1), (50, 80)])
def test_invalid_params(size, overlap):
    with pytest.raises(ValueError):
        split_text("texto", size, overlap)


def test_chunks_do_not_cross_pages_and_keep_metadata():
    pages = [Page("a " * 100, "f.txt", 1), Page("b " * 100, "f.txt", 2)]
    chunks = chunk_pages(pages, 60, 10)
    assert {c.page for c in chunks} == {1, 2}
    for c in chunks:
        assert set(c.text.split()) == ({"a"} if c.page == 1 else {"b"})
        assert c.metadata == {"source": "f.txt", "page": c.page, "chunk_index": c.chunk_index}
    assert len({c.id for c in chunks}) == len(chunks)  # ids únicos


def test_ingest_directory_end_to_end(docs_dir):
    chunks = ingest_directory(docs_dir, chunk_size=200, overlap=20)
    assert {c.source for c in chunks} == {"manual.pdf", "politica.docx", "notas.txt"}
    assert all(c.page >= 1 for c in chunks)
