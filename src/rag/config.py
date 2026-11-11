"""Configuración central del proyecto, cargada desde variables de entorno / .env."""
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM
    llm_provider: str = "gemini"  # gemini | ollama | claude
    gemini_api_key: SecretStr = SecretStr("")  # SecretStr: no aparece en repr ni logs
    gemini_model: str = "gemini-3.5-flash"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    anthropic_api_key: SecretStr = SecretStr("")
    claude_model: str = "claude-sonnet-5-5"
    llm_max_retries: int = 5
    llm_retry_base_delay: float = 1.0  # segundos; se duplica en cada reintento
    llm_timeout: float = 60.0

    # RAG
    data_dir: Path = PROJECT_ROOT / "data"
    chroma_dir: Path = PROJECT_ROOT / "chroma_db"
    embedding_model: str = "intfloat/multilingual-e5-base"  # ventana de 512 tokens
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    min_relevance: float = 0.80  # calibrado para multilingual-e5-base (ver CLAUDE.md)
    dedup_similarity: float = 0.9


@lru_cache
def get_settings() -> Settings:
    return Settings()
