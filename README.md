# rag-empresarial

Asistente RAG empresarial en Python que responde preguntas usando documentos PDF, Word (.docx) y TXT,
citando archivo y página.

> Proyecto en construcción por etapas. El README completo (arquitectura, instalación y uso) llegará en la etapa final.

## Instalación rápida (desarrollo)

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env   # y completa GEMINI_API_KEY
pytest
```
