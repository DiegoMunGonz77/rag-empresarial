from rag.config import Settings, get_settings

from .base import LLMError, LLMProvider, ModelNotFoundError, RateLimitError


def get_provider(name: str | None = None, settings: Settings | None = None) -> LLMProvider:
    """Crea el proveedor indicado (por defecto el de LLM_PROVIDER, que es 'gemini')."""
    settings = settings or get_settings()
    name = (name or settings.llm_provider).lower()
    if name == "gemini":
        from .gemini import GeminiProvider

        return GeminiProvider()
    if name == "ollama":
        from .ollama import OllamaProvider

        return OllamaProvider()
    if name == "claude":
        from .claude import ClaudeProvider

        return ClaudeProvider()
    raise ValueError(f"Proveedor LLM desconocido: '{name}' (usa gemini, ollama o claude)")


__all__ = [
    "LLMError",
    "LLMProvider",
    "ModelNotFoundError",
    "RateLimitError",
    "get_provider",
]
