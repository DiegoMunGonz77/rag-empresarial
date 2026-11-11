"""CLI: index, search, llm y models.

  python -m rag.cli index
  python -m rag.cli search "pregunta"
  python -m rag.cli llm "pregunta corta" [--provider gemini|ollama|claude]
  python -m rag.cli models [--provider ...]
"""
from __future__ import annotations

import argparse
import sys

from rag.config import get_settings
from rag.llm import LLMError, ModelNotFoundError, get_provider


def _store():
    from rag.embeddings import SentenceTransformerEmbedder
    from rag.vectorstore import VectorStore

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
    p_llm = sub.add_parser("llm", help="Pregunta directa al proveedor LLM (sin RAG)")
    p_llm.add_argument("prompt")
    p_llm.add_argument("--provider", default=None)
    p_models = sub.add_parser("models", help="Lista los modelos disponibles del proveedor")
    p_models.add_argument("--provider", default=None)
    args = parser.parse_args(argv)

    if args.cmd == "index":
        from rag.ingestion.pipeline import index_directory

        store = _store()
        r = index_directory(store)
        print(
            f"nuevos={r.added} actualizados={r.updated} sin_cambios={r.unchanged} "
            f"eliminados={r.removed} fallidos={r.failed} chunks_indexados={r.chunks_indexed} "
            f"total_en_indice={store.count()}"
        )
    elif args.cmd == "search":
        for i, hit in enumerate(_store().search(args.query, args.k), start=1):
            print(f"\n[{i}] {hit.source}, p. {hit.page}  (similitud {hit.score:.3f})")
            print(hit.text[:400].replace("\n", " "))
    else:
        try:
            provider = get_provider(args.provider)
            if args.cmd == "models":
                print(f"Modelos de {provider.name} (configurado: {provider.model}):")
                print("\n".join(f"  {m}" for m in provider.list_models()))
            else:
                print(f"[{provider.name} / {provider.model}]")
                print(provider.generate(args.prompt))
        except ModelNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            print("Ejecuta `python -m rag.cli models` para ver los modelos disponibles.",
                  file=sys.stderr)
            return 2
        except LLMError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
