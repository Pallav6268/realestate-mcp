"""The RERA Act 2016 (official text, reused from Week 2) for the legal specialist."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from rank_bm25 import BM25Okapi

SECTIONS_FILE = Path(__file__).resolve().parent / "data" / "rera" / "sections.json"


@lru_cache(maxsize=1)
def _sections() -> list[dict]:
    return json.loads(SECTIONS_FILE.read_text(encoding="utf-8"))


def _tokens(text: str) -> list[str]:
    return [t[:6] for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2]


@lru_cache(maxsize=1)
def _index() -> BM25Okapi:
    return BM25Okapi([_tokens(f"{s['title']} {s['text']}") for s in _sections()])


def read_rera_section(section: str) -> dict:
    wanted = str(section).strip().lower().removeprefix("section").strip()
    for s in _sections():
        if s["section"] == wanted:
            return {"section": s["section"], "title": s["title"], "text": s["text"][:4000]}
    return {"error": f"Section {section} not found in the Act."}


def search_rera(query: str, k: int = 3) -> list[dict]:
    scores = _index().get_scores(_tokens(query))
    best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [
        {
            "section": _sections()[i]["section"],
            "title": _sections()[i]["title"],
            "excerpt": _sections()[i]["text"][:500],
        }
        for i in best
        if scores[i] > 0
    ]
