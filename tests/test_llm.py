from types import SimpleNamespace

import pytest
import requests
from google.genai import errors as genai_errors

from rag.config import Settings
from rag.llm import LLMError, ModelNotFoundError, RateLimitError, get_provider
from rag.llm.claude import ClaudeProvider
from rag.llm.gemini import GeminiProvider
from rag.llm.ollama import OllamaProvider
from rag.llm.retry import call_with_retry

SECRET = "AIzaSy-SECRETO-DE-PRUEBA-123"


# --- retry ---------------------------------------------------------------------


def test_retry_succeeds_after_429s_with_exponential_backoff():
    calls, sleeps = [], []

    def fn():
        calls.append(1)
        if len(calls) < 4:
            raise RateLimitError("429")
        return "ok"

    assert call_with_retry(fn, 5, 1.0, sleeps.append, jitter=lambda: 0) == "ok"
    assert sleeps == [1.0, 2.0, 4.0]


def test_retry_gives_up_after_max_retries():
    calls, sleeps = [], []

    def fn():
        calls.append(1)
        raise RateLimitError("429")

    with pytest.raises(RateLimitError):
        call_with_retry(fn, 3, 1.0, sleeps.append, jitter=lambda: 0)
    assert len(calls) == 4 and len(sleeps) == 3  # 1 intento + 3 reintentos


def test_retry_honors_retry_after_and_caps_delay():
    sleeps = []
    attempts = iter([RateLimitError("429", retry_after=10), RateLimitError("429", 999), "ok"])

    def fn():
        v = next(attempts)
        if isinstance(v, Exception):
            raise v
        return v

    call_with_retry(fn, 5, 1.0, sleeps.append, jitter=lambda: 0)
    assert sleeps == [10.0, 60.0]


def test_retry_does_not_retry_other_errors():
    calls = []

    def fn():
        calls.append(1)
        raise LLMError("500")

    with pytest.raises(LLMError):
        call_with_retry(fn, 5, 0, lambda s: None)
    assert len(calls) == 1


# --- Gemini --------------------------------------------------------------------


class FakeGeminiClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.models = self

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return SimpleNamespace(text=out)

    def list(self):
        return [
            SimpleNamespace(name="models/gemini-a", supported_actions=["generateContent"]),
            SimpleNamespace(name="models/embed-x", supported_actions=["embedContent"]),
        ]


def _gemini(outcomes, **kw):
    client = FakeGeminiClient(outcomes)
    return GeminiProvider(api_key=SECRET, client=client, retry_base_delay=0, **kw), client


def _api_error(code, msg="boom"):
    cls = genai_errors.ClientError if code < 500 else genai_errors.ServerError
    return cls(code, {"error": {"message": msg, "status": "X"}})


def test_gemini_generate_ok_passes_model_and_system():
    provider, client = _gemini(["hola"], model="gemini-test")
    assert provider.generate("pregunta", system="sé breve") == "hola"
    call = client.calls[0]
    assert call["model"] == "gemini-test" and call["contents"] == "pregunta"
    assert call["config"].system_instruction == "sé breve"


def test_gemini_retries_429_then_succeeds():
    provider, client = _gemini([_api_error(429), _api_error(429), "listo"])
    assert provider.generate("q") == "listo"
    assert len(client.calls) == 3


def test_gemini_429_exhausted_raises_rate_limit():
    provider, client = _gemini([_api_error(429)] * 10, max_retries=2)
    with pytest.raises(RateLimitError):
        provider.generate("q")
    assert len(client.calls) == 3


def test_gemini_404_is_model_not_found_without_retry():
    provider, client = _gemini([_api_error(404, "model not found")])
    with pytest.raises(ModelNotFoundError):
        provider.generate("q")
    assert len(client.calls) == 1


def test_gemini_other_errors_and_empty_response():
    provider, _ = _gemini([_api_error(500, "interno")])
    with pytest.raises(LLMError, match="500"):
        provider.generate("q")
    provider, _ = _gemini([""])
    with pytest.raises(LLMError, match="vacía"):
        provider.generate("q")


def test_gemini_never_leaks_api_key():
    provider, _ = _gemini([_api_error(500, f"clave inválida {SECRET}")])
    with pytest.raises(LLMError) as exc:
        provider.generate("q")
    assert SECRET not in str(exc.value)
    assert "***" in str(exc.value)


def test_gemini_list_models_only_generate_content():
    provider, _ = _gemini([])
    assert provider.list_models() == ["gemini-a"]


def test_gemini_requires_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        GeminiProvider(api_key="")


def test_settings_do_not_expose_secret_in_repr():
    s = Settings(_env_file=None, gemini_api_key=SECRET, anthropic_api_key=SECRET)
    assert SECRET not in repr(s) and SECRET not in str(s)
    assert s.gemini_api_key.get_secret_value() == SECRET


# --- Ollama --------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status=200, body=None, headers=None, text=""):
        self.status_code, self._body, self.headers, self.text = status, body or {}, headers or {}, text

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.posts = []

    def post(self, url, json, timeout):
        self.posts.append({"url": url, "json": json})
        out = self.responses.pop(0)
        if isinstance(out, Exception):
            raise out
        return out

    def get(self, url, timeout):
        return self.responses.pop(0)


def test_ollama_generate_and_payload():
    session = FakeSession([FakeResponse(body={"response": "hola local"})])
    p = OllamaProvider(base_url="http://x:1/", model="m", session=session, retry_base_delay=0)
    assert p.generate("q", system="s") == "hola local"
    assert session.posts[0]["url"] == "http://x:1/api/generate"
    assert session.posts[0]["json"] == {"model": "m", "prompt": "q", "stream": False, "system": "s"}


def test_ollama_retries_429_and_maps_errors():
    session = FakeSession([FakeResponse(429, headers={"Retry-After": "0"}),
                           FakeResponse(body={"response": "ok"})])
    p = OllamaProvider(session=session, retry_base_delay=0)
    assert p.generate("q") == "ok"

    with pytest.raises(ModelNotFoundError):
        OllamaProvider(session=FakeSession([FakeResponse(404)]), retry_base_delay=0).generate("q")
    with pytest.raises(LLMError, match="conectar"):
        OllamaProvider(
            session=FakeSession([requests.ConnectionError("down")]), retry_base_delay=0
        ).generate("q")
    with pytest.raises(LLMError, match="500"):
        OllamaProvider(
            session=FakeSession([FakeResponse(500, text="err")]), retry_base_delay=0
        ).generate("q")


def test_ollama_list_models():
    session = FakeSession([FakeResponse(body={"models": [{"name": "b"}, {"name": "a"}]})])
    assert OllamaProvider(session=session).list_models() == ["a", "b"]


# --- Claude --------------------------------------------------------------------


class FakeAnthropicError(Exception):
    def __init__(self, status, msg="x", headers=None):
        super().__init__(msg)
        self.status_code = status
        self.response = SimpleNamespace(headers=headers or {})


class FakeAnthropicClient:
    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=out)])


def test_claude_generate_system_and_retry():
    client = FakeAnthropicClient([FakeAnthropicError(429, headers={"retry-after": "0"}), "hola"])
    p = ClaudeProvider(api_key="k", client=client, retry_base_delay=0, model="m")
    assert p.generate("q", system="s") == "hola"
    assert len(client.calls) == 2
    assert client.calls[0]["system"] == "s" and client.calls[0]["model"] == "m"


def test_claude_error_mapping_and_redaction():
    p = ClaudeProvider(api_key=SECRET, client=FakeAnthropicClient([FakeAnthropicError(404)]))
    with pytest.raises(ModelNotFoundError):
        p.generate("q")
    p = ClaudeProvider(
        api_key=SECRET,
        client=FakeAnthropicClient([FakeAnthropicError(401, f"bad key {SECRET}")]),
    )
    with pytest.raises(LLMError) as exc:
        p.generate("q")
    assert SECRET not in str(exc.value)


# --- factory -------------------------------------------------------------------


def test_get_provider_default_is_gemini(monkeypatch):
    monkeypatch.setattr("rag.llm.gemini.GeminiProvider.__init__", lambda self: None)
    assert get_provider(settings=Settings(_env_file=None, llm_provider="gemini")).name == "gemini"


def test_get_provider_unknown_and_ollama():
    with pytest.raises(ValueError):
        get_provider("nada")
    assert get_provider("ollama").name == "ollama"
