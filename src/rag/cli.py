"""CLI: `python -m rag.cli index` y `python -m rag.cli search "pregunta"`."""
from __future__ import annotations

import argparse
import sys

from rag.config import get_settings
from rag.embeddings import SentenceTransformerEmbedder
from rag.ingestion.pipeline import index_directory
from rag.vectorstore import VectorStore


def _store() -> VectorStore:
    s = get_settings()
    return VectorStore(s.chroma_dir, SentenceTransformerEmbedder(s.embedding_model))


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="rag")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index", help="Indexa (incrementalmente) los documentos de data/")
    p_search = sub.add_parser("search", help="Búsqueda semántica en el índice")
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=get_settings().top_k)
    args = parser.parse_args(argv)

    store = _store()
    if args.cmd == "index":
        r = index_directory(store)
        print(
            f"nuevos={r.added} actualizados={r.updated} sin_cambios={r.unchanged} "
            f"eliminados={r.removed} fallidos={r.failed} chunks_indexados={r.chunks_indexed} "
            f"total_en_indice={store.count()}"
        )
    else:
        for i, hit in enumerate(store.search(args.query, args.k), start=1):
            print(f"\n[{i}] {hit.source}, p. {hit.page}  (similitud {hit.score:.3f})")
            print(hit.text[:400].replace("\n", " "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
