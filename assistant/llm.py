"""Pick the model from environment variables.

ANTHROPIC_API_KEY  -> Claude   (model from ANTHROPIC_MODEL)
GEMINI_API_KEY     -> Gemini   (model from GEMINI_MODEL; GOOGLE_API_KEY also works)
DEEPSEEK_API_KEY   -> DeepSeek (model from DEEPSEEK_MODEL)
GROQ_API_KEY       -> Groq     (model from GROQ_MODEL)
none               -> rule-based demo model (no key, no cost)

Every provider with a key is used, in that order: the first is primary and the
rest are automatic fallbacks if a call fails (for example a rejected key or a
rate limit). MODEL_ORDER (e.g. "groq,gemini") overrides the order and can leave
providers out.

Tracing: set LANGSMITH_TRACING=true and LANGSMITH_API_KEY to send every run
to LangSmith (project from LANGSMITH_PROJECT).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter

from .demo_model import ScriptedToolPicker

DEFAULT_ANTHROPIC_MODEL = "claude-haiku-5-5"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_ORDER = ("claude", "gemini", "deepseek", "groq")
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
# Free tiers have per-minute token limits; a 429 tells the client how long to wait, so retry instead of failing.
MAX_RETRIES = 6


@dataclass
class ModelChoice:
    llm: BaseChatModel
    label: str
    live: bool
    fallbacks: list[BaseChatModel] = field(default_factory=list)


def _claude() -> tuple[BaseChatModel, str] | None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    from langchain_anthropic import ChatAnthropic

    name = os.getenv("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
    return ChatAnthropic(model=name, temperature=0, max_tokens=2048), f"Claude · {name}"


def _gemini() -> tuple[BaseChatModel, str] | None:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        return None
    from langchain_google_genai import ChatGoogleGenerativeAI

    name = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    # Free tier allows about 15 requests per minute; a tool loop can burst past it, so pace the calls.
    rpm = float(os.getenv("GEMINI_RPM", "14"))
    limiter = InMemoryRateLimiter(requests_per_second=rpm / 60, check_every_n_seconds=0.1, max_bucket_size=2)
    model = ChatGoogleGenerativeAI(
        model=name, google_api_key=key, temperature=0, max_retries=MAX_RETRIES, rate_limiter=limiter
    )
    return model, f"Gemini · {name}"


def _deepseek() -> tuple[BaseChatModel, str] | None:
    if not os.getenv("DEEPSEEK_API_KEY"):
        return None
    from langchain_deepseek import ChatDeepSeek

    name = os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
    return ChatDeepSeek(model=name, temperature=0, max_retries=MAX_RETRIES), f"DeepSeek · {name}"


def _groq() -> tuple[BaseChatModel, str] | None:
    if not os.getenv("GROQ_API_KEY"):
        return None
    from langchain_groq import ChatGroq

    name = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    return ChatGroq(model=name, temperature=0, max_retries=MAX_RETRIES), f"Groq · {name}"


PROVIDERS = {"claude": _claude, "gemini": _gemini, "deepseek": _deepseek, "groq": _groq}


def provider_order() -> list[str]:
    raw = os.getenv("MODEL_ORDER", "")
    order = [p.strip().lower() for p in raw.split(",") if p.strip().lower() in PROVIDERS]
    return order or list(DEFAULT_ORDER)


def available_models() -> list[tuple[BaseChatModel, str]]:
    """Every live provider that has a key, in priority order."""
    return [m for name in provider_order() if (m := PROVIDERS[name]()) is not None]


def get_model() -> ModelChoice:
    models = available_models()
    if not models:
        return ModelChoice(ScriptedToolPicker(), "Demo mode (rule-based tool picker, no API key)", False)
    (primary, label), rest = models[0], models[1:]
    if rest:
        label += " (fallback: " + ", ".join(lbl for _, lbl in rest) + ")"
    return ModelChoice(primary, label, True, [m for m, _ in rest])
