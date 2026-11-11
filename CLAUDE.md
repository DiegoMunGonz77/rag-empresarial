# CLAUDE.md — rag-empresarial

Sistema RAG empresarial: responde preguntas sobre PDF, DOCX y TXT de `data/`, citando archivo y página.

## Entorno
- Python 3.12 en `.venv` (3.14 del sistema no se usa: torch/sentence-transformers aún sin soporte estable).
- Dependencias con `pip` + `pyproject.toml`: `pip install -e ".[dev]"`. Extra opcional `claude`.
- Código en `src/rag/` (layout `src`); tests en `tests/` con `pythonpath=src`.
- Windows + PowerShell; `gh` autenticado por el usuario (no pedir ni manejar credenciales).

## Decisiones
- **Embeddings:** sentence-transformers, `paraphrase-multilingual-MiniLM-L12-v2`, locales.
- **Vector store:** ChromaDB persistente en `chroma_db/` (ignorado por git).
- **LLM:** interfaz `LLMProvider`. `GeminiProvider` por defecto (`GEMINI_API_KEY` en `.env`,
  modelo en `GEMINI_MODEL`, inicial `gemini-2.5-flash`). `OllamaProvider` y `ClaudeProvider` opcionales.
  Reintentos con backoff exponencial ante 429. Si la key da error de modelo, avisar al usuario.
- **Citas:** estructuradas desde el retriever (archivo + página de cada chunk), no dependen de que el LLM las escriba.
  Si el contexto no alcanza (umbral `MIN_RELEVANCE`), responder "no encontré esa información".
- **Config:** `rag.config.Settings` (pydantic-settings) lee `.env`; ver `.env.example`.
- **Secretos:** `.env` y `chroma_db/` en `.gitignore`. Nunca commitear keys.

## Flujo de trabajo
- Implementación por etapas, un commit por etapa, push a GitHub tras cada una (repo público `rag-empresarial`).
- Tras cada etapa: detenerse para revisión del usuario antes de seguir.
- Etapas: 1 base · 2 ingesta · 3 embeddings+vectorstore · 4 proveedores LLM · 5 servicio RAG ·
  6 API FastAPI · 7 UI Streamlit · 8 evaluación · 9 Docker+README.
- Etapa actual: 1 completada.
