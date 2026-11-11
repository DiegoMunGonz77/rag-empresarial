# rag-empresarial

Asistente RAG empresarial en Python que responde preguntas usando documentos PDF, Word (.docx) y TXT,
citando archivo y página.

> Proyecto en construcción por etapas. El README completo (arquitectura, instalación y uso) llegará en la etapa final.

## Documentos de ejemplo (`data/`)

Los PDF de `data/` son manuales con derechos de autor y **no se suben al repositorio**
(`data/*.pdf` está en `.gitignore`; solo se conserva `data/.gitkeep`).
Descárgalos tú desde la sección de manuales de usuario del sitio oficial de Renault de tu país
(busca "manual de utilización Renault Kwid") y guárdalos en `data/` con estos nombres:

- `Manual_Kwid_2019.pdf`
- `manual-kwid-2021-2022.pdf`
- `Manual_Kwid_2023.pdf`

Puedes añadir también tus propios `.pdf`, `.docx` y `.txt`.

## Instalación rápida (desarrollo)

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env   # y completa GEMINI_API_KEY
pytest
```
