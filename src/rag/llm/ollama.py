"""OllamaProvider: modelos locales vía la API HTTP de Ollama."""
from __future__ import annotations

import requests

from rag.config import get_settings

from .base import (
    LLMError,
    LLMProvider,
    ModelNotFoundError,
    RateLimitError,
    ServiceUnavailableError,
)
from .retry import call_with_retry


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        max_retries: int | None = None,
        retry_base_delay: float | None = None,
        timeout: float | None = None,
        session=None,
    ):
        s = get_settings()
        self.base_url = (base_url or s.ollama_base_url).rstrip("/")
        self.model = model or s.ollama_model
        self.max_retries = s.llm_max_retries if max_retries is None else max_retries
        self.retry_base_delay = (
            s.llm_retry_base_delay if retry_base_delay is None else retry_base_delay
        )
        self.timeout = timeout or s.llm_timeout
        self._http = session or requests

    def _generate_once(self, prompt: str, system: str | None) -> str:
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system
        try:
            resp = self._http.post(
                f"{self.base_url}/api/generate", json=payload, timeout=self.timeout
            )
        except requests.RequestException as exc:
            raise LLMError(f"No se pudo conectar con Ollama en {self.base_url}: {exc}") from None
        if resp.status_code == 429:
            raise RateLimitError("Ollama 429", _retry_after(resp))
        if resp.status_code == 503:
            raise ServiceUnavailableError("Ollama 503", _retry_after(resp))
        if resp.status_code == 404:
            raise ModelNotFoundError(
                f"Modelo '{self.model}' no encontrado en Ollama (prueba `ollama pull {self.model}`)"
            )
        if resp.status_code >= 400:
            raise LLMError(f"Error de Ollama ({resp.status_code}): {resp.text[:200]}")
        return resp.json()["response"]

    def generate(self, prompt: str, system: str | None = None) -> str:
        return call_with_retry(
            lambda: self._generate_once(prompt, system),
            max_retries=self.max_retries,
            base_delay=self.retry_base_delay,
        )

    def list_models(self) -> list[str]:
        try:
            resp = self._http.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise LLMError(f"No se pudo listar modelos de Ollama: {exc}") from None
        return sorted(m["name"] for m in resp.json().get("models", []))


def _retry_after(resp) -> float | None:
    try:
        return float(resp.headers.get("Retry-After"))
    except (TypeError, ValueError):
        return None
