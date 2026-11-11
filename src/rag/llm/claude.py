"""ClaudeProvider (opcional): requiere `pip install -e ".[claude]"` y ANTHROPIC_API_KEY."""
from __future__ import annotations

from rag.config import get_settings

from .base import (
    LLMError,
    LLMProvider,
    ModelNotFoundError,
    RateLimitError,
    ServiceUnavailableError,
)
from .retry import call_with_retry


class ClaudeProvider(LLMProvider):
    name = "claude"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int = 1024,
        max_retries: int | None = None,
        retry_base_delay: float | None = None,
        client=None,
    ):
        s = get_settings()
        self._api_key = api_key if api_key is not None else s.anthropic_api_key.get_secret_value()
        self.model = model or s.claude_model
        self.max_tokens = max_tokens
        self.max_retries = s.llm_max_retries if max_retries is None else max_retries
        self.retry_base_delay = (
            s.llm_retry_base_delay if retry_base_delay is None else retry_base_delay
        )
        if client is None:
            if not self._api_key:
                raise LLMError("Falta ANTHROPIC_API_KEY (defínela en .env)")
            try:
                import anthropic
            except ImportError:
                raise LLMError(
                    'Falta el paquete anthropic: pip install -e ".[claude]"'
                ) from None
            # max_retries=0: los reintentos los gestiona call_with_retry
            client = anthropic.Anthropic(api_key=self._api_key, max_retries=0)
        self._client = client

    def _redact(self, text: str) -> str:
        return text.replace(self._api_key, "***") if self._api_key else text

    def _generate_once(self, prompt: str, system: str | None) -> str:
        kwargs = {"system": system} if system else {}
        try:
            msg = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            text = self._redact(str(exc))
            if status in (429, 503, 529):  # 529 = overloaded en la API de Anthropic
                response = getattr(exc, "response", None)
                headers = getattr(response, "headers", None) or {}
                try:
                    retry_after = float(headers.get("retry-after"))
                except (TypeError, ValueError):
                    retry_after = None
                error_cls = RateLimitError if status == 429 else ServiceUnavailableError
                raise error_cls(f"Claude {status}: {text}", retry_after) from None
            if status == 404:
                raise ModelNotFoundError(f"Modelo '{self.model}' no disponible: {text}") from None
            if status is not None:
                raise LLMError(f"Error de Claude ({status}): {text}") from None
            raise
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")

    def generate(self, prompt: str, system: str | None = None) -> str:
        return call_with_retry(
            lambda: self._generate_once(prompt, system),
            max_retries=self.max_retries,
            base_delay=self.retry_base_delay,
        )

    def list_models(self) -> list[str]:
        return sorted(m.id for m in self._client.models.list())
