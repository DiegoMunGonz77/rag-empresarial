"""Embedder falso y determinista: bolsa de palabras hasheada. Sin descargas de modelos."""
import hashlib
import math
import re

DIM = 256


class FakeEmbedder:
    model_name = "fake-embedder"

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for word in re.findall(r"\w+", text.lower()):
            idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM
            v[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_passages(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)
