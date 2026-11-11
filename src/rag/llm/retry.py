"""Reintentos con backoff exponencial ante errores transitorios (429 y 503)."""
from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from typing import TypeVar

from .base import RetryableError

logger = logging.getLogger(__name__)
T = TypeVar("T")
MAX_DELAY = 60.0


def call_with_retry(
    fn: Callable[[], T],
    max_retries: int = 5,
    base_delay: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[], float] = lambda: random.uniform(0, 0.25),
) -> T:
    """Ejecuta `fn`; ante un error transitorio (429 o 503) espera y reintenta hasta
    `max_retries` veces.

    Espera = base_delay * 2**intento (+ jitter), o el `retry_after` que indique el servidor
    si es mayor. Tras agotar los reintentos relanza el último error.
    Cualquier otro error se propaga de inmediato.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except RetryableError as exc:
            if attempt >= max_retries:
                raise
            delay = min(max(base_delay * 2**attempt, exc.retry_after or 0.0) + jitter(), MAX_DELAY)
            logger.warning(
                "Error transitorio del proveedor (%s); reintento %d/%d en %.1fs",
                type(exc).__name__,
                attempt + 1,
                max_retries,
                delay,
            )
            sleep(delay)
            attempt += 1
