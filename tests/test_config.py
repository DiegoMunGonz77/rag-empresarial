from rag.config import Settings


def test_defaults():
    s = Settings(_env_file=None)
    assert s.llm_provider == "gemini"
    assert s.gemini_model == "gemini-2.5-flash"
    assert s.chunk_overlap < s.chunk_size


def test_env_override(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "otro-modelo")
    assert Settings(_env_file=None).gemini_model == "otro-modelo"
