# CLAUDE.md — rag-empresarial

Sistema RAG empresarial: responde preguntas sobre PDF, DOCX y TXT de `data/`, citando archivo y página.

## Entorno
- Python 3.12 en `.venv` (3.14 del sistema no se usa: torch/sentence-transformers aún sin soporte estable).
- Dependencias con `pip` + `pyproject.toml`: `pip install -e ".[dev]"`. Extra opcional `claude`.
- Código en `src/rag/` (layout `src`); tests en `tests/` con `pythonpath=src`.
- Windows + PowerShell; `gh` autenticado por el usuario (no pedir ni manejar credenciales).

## Decisiones
- **Embeddings:** sentence-transformers, `intfloat/multilingual-e5-base` (locales, 512 tokens, prefijos
  `query: `/`passage: ` aplicados en `SentenceTransformerEmbedder`). Se descartó
  `paraphrase-multilingual-MiniLM-L12-v2`: su ventana es de 128 tokens y truncaba el 75 % de los chunks de 1000 caracteres.
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
- Etapas 1, 2, 3 y 4 completadas.

## Proveedores LLM (etapa 4)
- `rag.llm`: `LLMProvider.generate(prompt, system=None)` + `list_models()`; `get_provider(name)` elige según `LLM_PROVIDER`.
- Errores propios: `LLMError`, `RateLimitError` (429, reintentable), `ModelNotFoundError`. Solo el 429 se reintenta
  (`call_with_retry`: backoff 1,2,4,… s + jitter, respeta `retry_after`, tope 60 s, `LLM_MAX_RETRIES`).
- Las keys son `SecretStr` (no salen en repr) y los mensajes de error las redactan. Nunca imprimir ni leer `.env` en voz alta.
- Timeout HTTP en Gemini = `LLM_TIMEOUT` (60 s).
- `python -m rag.cli llm "..."` y `python -m rag.cli models` para probar el proveedor.
- `gemini-2.5-flash` ya NO está disponible para la key del usuario (404). Medido: `gemini-3.5-flash` responde en ~5 s;
  `gemini-3.8-flash` fue lento (timeout 60 s) y `gemini-3.7-flash` dio 503. Modelo recomendado: `gemini-3.5-flash`.

## Vector store e indexación (etapa 3)
- `VectorStore` (ChromaDB persistente, coseno): `add_chunks`, `search`, `delete_source`, `indexed_files`.
  Guarda el modelo de embeddings en la colección; si cambia lanza `EmbeddingModelMismatch` (borrar `chroma_db/`).
- Indexación incremental (`index_directory`): huella = sha256 del archivo + chunk_size + overlap.
  Solo reindexa archivos nuevos/modificados y borra del índice los desaparecidos. Un archivo roto se omite.
- CLI: `python -m rag.cli index` / `python -m rag.cli search "pregunta"`.
- Tests usan `tests/fakes.py::FakeEmbedder` (sin descargar modelos).

## Ingesta (etapa 2)
- Loaders devuelven `Page(text, source, page)`. PDF: página real (pypdf). DOCX: se numera por saltos de
  página explícitos (si no hay, todo es página 1); incluye tablas. TXT: `\f` separa páginas; utf-8 con fallback latin-1.
- `load_directory` no es recursivo y omite (con log) archivos ilegibles.
- El chunker corta por caracteres (1000 / overlap 200 por defecto), prefiere límites de párrafo/frase y
  **nunca cruza páginas**, para que la cita de página sea exacta. Id de chunk: `archivo::pN::cK`.
- PDF: se unen palabras cortadas con guion (`dehyphenate`) y se colapsan los puntos guía de índices
  (`collapse_dot_leaders`; cada punto era un token y una página de índice llegaba a 834 tokens).
  Con eso, chunks de 1000/200 miden ≤286 tokens en los manuales reales: 0 % truncados con e5 (ventana 512).
  Si se cambia el modelo o el chunking, re-verificar tokens vs. ventana.
- `data/*.pdf` está en `.gitignore` (manuales con copyright); los tests generan sus propios archivos en `tmp_path`.
