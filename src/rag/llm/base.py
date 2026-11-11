"""Interfaz común de proveedores LLM y sus errores."""
from __future__ import annotations

from abc import ABC, abstractmethod


class LLMError(RuntimeError):
    """Error genérico de un proveedor LLM (el mensaje nunca contiene credenciales)."""


class RetryableError(LLMError):
    """Error transitorio del servidor: tiene sentido reintentar con backoff."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class RateLimitError(RetryableError):
    """HTTP 429 / cuota excedida."""


class ServiceUnavailableError(RetryableError):
    """HTTP 503 (o 529 en Claude): servicio saturado o no disponible temporalmente."""


class ModelNotFoundError(LLMError):
    """El modelo configurado no existe o no está disponible para esta key."""


class LLMProvider(ABC):
    name: str
    model: str

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> str:
        """Devuelve el texto generado. Reintenta internamente ante 429."""

    def list_models(self) -> list[str]:
        raise NotImplementedError(f"{self.name} no soporta listar modelos")
