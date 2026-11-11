"""Fixtures que generan sus propios documentos de prueba en una carpeta temporal."""
import pytest
from docx import Document
from docx.enum.text import WD_BREAK


def build_pdf(page_texts: list[str]) -> bytes:
    """Construye un PDF válido mínimo con una línea de texto por página."""
    objs: list[bytes] = []
    n = len(page_texts)
    font_id = 3 + 2 * n
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    for i, text in enumerate(page_texts):
        content_id = 4 + 2 * i
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
            f"/Contents {content_id} 0 R >>".encode()
        )
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({safe}) Tj ET".encode("latin-1")
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return bytes(out)


@pytest.fixture
def pdf_file(tmp_path):
    path = tmp_path / "manual.pdf"
    path.write_bytes(build_pdf(["Primera pagina del manual", "Segunda pagina con frenos"]))
    return path


@pytest.fixture
def docx_file(tmp_path):
    path = tmp_path / "politica.docx"
    doc = Document()
    doc.add_paragraph("Politica de vacaciones")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Dias"
    table.rows[0].cells[1].text = "15"
    p = doc.add_paragraph("Fin de la pagina uno")
    p.add_run().add_break(WD_BREAK.PAGE)
    p.add_run("Inicio de la pagina dos")
    doc.add_paragraph("Texto adicional de la pagina dos")
    doc.save(path)
    return path


@pytest.fixture
def txt_file(tmp_path):
    path = tmp_path / "notas.txt"
    path.write_text("Linea uno\n\fPagina dos con acentos: añadir ñandú", encoding="utf-8")
    return path


@pytest.fixture
def docs_dir(tmp_path, pdf_file, docx_file, txt_file):
    (tmp_path / "ignorar.csv").write_text("a,b")
    (tmp_path / "roto.pdf").write_bytes(b"esto no es un pdf")
    return tmp_path
