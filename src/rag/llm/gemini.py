"""GeminiProvider (proveedor por defecto) sobre el SDK google-genai."""
from __future__ import annotations

from rag.config import get_settings

from .base import LLMError, LLMProvider, ModelNotFoundError, RateLimitError
from .retry import call_with_retry


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_retries: int | None = None,
        retry_base_delay: float | None = None,
        client=None,
    ):
        s = get_settings()
        self._api_key = api_key if api_key is not None else s.gemini_api_key.get_secret_value()
        self.model = model or s.gemini_model
        self.max_retries = s.llm_max_retries if max_retries is None else max_retries
        self.retry_base_delay = (
            s.llm_retry_base_delay if retry_base_delay is None else retry_base_delay
        )
        if client is None:
            if not self._api_key:
                raise LLMError("Falta GEMINI_API_KEY (defínela en .env)")
            from google import genai
            from google.genai import types

            client = genai.Client(
                api_key=self._api_key,
                http_options=types.HttpOptions(timeout=int(s.llm_timeout * 1000)),  # ms
            )
        self._client = client

    def _redact(self, text: str) -> str:
        return text.replace(self._api_key, "***") if self._api_key else text

    def _translate(self, exc: Exception) -> Exception:
        """Convierte errores del SDK en errores propios, sin filtrar la key."""
        code = getattr(exc, "code", None)
        msg = self._redact(str(exc))
        if code == 429:
            return RateLimitError(f"Gemini 429 (cuota/límite): {msg}")
        if code == 404 or (code == 400 and "model" in msg.lower() and "not" in msg.lower()):
            return ModelNotFoundError(f"Modelo '{self.model}' no disponible: {msg}")
        return LLMError(f"Error de Gemini ({code}): {msg}")

    def _generate_once(self, prompt: str, system: str | None) -> str:
        from google.genai import errors, types

        config = types.GenerateContentConfig(system_instruction=system) if system else None
        try:
            response = self._client.models.generate_content(
                model=self.model, contents=prompt, config=config
            )
        except errors.APIError as exc:
            raise self._translate(exc) from None
        text = response.text
        if not text:
            raise LLMError("Gemini devolvió una respuesta vacía (posible bloqueo de seguridad)")
        return text

    def generate(self, prompt: str, system: str | None = None) -> str:
        return call_with_retry(
            lambda: self._generate_once(prompt, system),
            max_retries=self.max_retries,
            base_delay=self.retry_base_delay,
        )

    def list_models(self) -> list[str]:
        from google.genai import errors

        try:
            return sorted(
                m.name.removeprefix("models/")
                for m in self._client.models.list()
                if "generateContent" in (m.supported_actions or [])
            )
        except errors.APIError as exc:
            raise self._translate(exc) from None
