import pytest

from tests.conftest import build_pdf
from rag.ingestion.loaders import (
    Page,
    collapse_dot_leaders,
    dehyphenate,
    load_directory,
    load_docx,
    load_document,
    load_pdf,
    load_txt,
)


def test_load_pdf_pages_and_metadata(pdf_file):
    pages = load_pdf(pdf_file)
    assert [p.page for p in pages] == [1, 2]
    assert all(p.source == "manual.pdf" for p in pages)
    assert "Primera pagina" in pages[0].text
    assert "frenos" in pages[1].text


def test_dehyphenate_joins_split_words_only():
    assert dehyphenate("los asientos de-\nlanteros") == "los asientos delanteros"
    assert dehyphenate("pa-  \n  rabrisas") == "parabrisas"
    assert dehyphenate("ver\r\nsiguiente") == "ver\r\nsiguiente"
    # No une si el guion es de lista, rango o precede a mayúscula/dígito
    assert dehyphenate("opciones:\n- primera") == "opciones:\n- primera"
    assert dehyphenate("modelo A-\n2019") == "modelo A-\n2019"
    assert dehyphenate("Mercedes-\nBenz") == "Mercedes-\nBenz"


def test_collapse_dot_leaders():
    src = "Escobillas . . . . . . . . . . . . 5.18\nFrenos.......... 5.20\nFin de frase. Otra."
    out = collapse_dot_leaders(src)
    assert out == "Escobillas ...  5.18\nFrenos ...  5.20\nFin de frase. Otra."
    assert collapse_dot_leaders("a... b") == "a... b"  # puntos suspensivos normales intactos


def test_load_pdf_dehyphenates(tmp_path):
    path = tmp_path / "guiones.pdf"
    path.write_bytes(build_pdf(["Los asientos de-\nlanteros se regulan"]))
    text = load_pdf(path)[0].text
    assert "delanteros" in text
    assert "de-" not in text


def test_load_docx_splits_on_page_break_and_reads_tables(docx_file):
    pages = load_docx(docx_file)
    assert [p.page for p in pages] == [1, 2]
    assert "Politica de vacaciones" in pages[0].text
    assert "Dias | 15" in pages[0].text
    assert "Fin de la pagina uno" in pages[0].text
    assert "Inicio de la pagina dos" in pages[1].text
    assert "Texto adicional" in pages[1].text
    assert all(p.source == "politica.docx" for p in pages)


def test_load_txt_form_feed_and_encoding(txt_file):
    pages = load_txt(txt_file)
    assert [p.page for p in pages] == [1, 2]
    assert "ñandú" in pages[1].text


def test_load_txt_latin1_fallback(tmp_path):
    path = tmp_path / "viejo.txt"
    path.write_bytes("canción".encode("latin-1"))
    assert load_txt(path)[0].text == "canción"


def test_load_document_dispatch_and_unsupported(tmp_path, txt_file):
    assert load_document(txt_file)[0].source == "notas.txt"
    bad = tmp_path / "datos.csv"
    bad.write_text("x")
    with pytest.raises(ValueError):
        load_document(bad)


def test_load_directory_skips_unsupported_and_broken(docs_dir):
    pages = load_directory(docs_dir)
    sources = {p.source for p in pages}
    assert sources == {"manual.pdf", "politica.docx", "notas.txt"}
    assert all(isinstance(p, Page) for p in pages)
