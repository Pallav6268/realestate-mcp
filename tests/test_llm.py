"""Provider selection from environment variables (no network calls)."""

import pytest

from assistant.llm import get_model

KEYS = ["ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "DEEPSEEK_API_KEY", "GROQ_API_KEY", "MODEL_ORDER"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)


def test_demo_mode_without_keys():
    assert get_model().live is False


def test_default_order_puts_gemini_before_deepseek_and_groq(monkeypatch):
    for key in ["GEMINI_API_KEY", "DEEPSEEK_API_KEY", "GROQ_API_KEY"]:
        monkeypatch.setenv(key, "test")
    choice = get_model()
    assert choice.label.startswith("Gemini · gemini-3.5-flash-lite (fallback: DeepSeek")
    assert len(choice.fallbacks) == 2


def test_model_order_overrides_and_skips(monkeypatch):
    for key in ["GEMINI_API_KEY", "DEEPSEEK_API_KEY", "GROQ_API_KEY"]:
        monkeypatch.setenv(key, "test")
    monkeypatch.setenv("MODEL_ORDER", "groq, gemini")
    choice = get_model()
    assert choice.label == "Groq · openai/gpt-oss-120b (fallback: Gemini · gemini-3.5-flash-lite)"
