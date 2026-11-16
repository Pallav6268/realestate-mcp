"""Buyer helpdesk documents (reused from Week 2): fictional Riverside Heights project documents.

Each ``## Heading`` in a project Markdown file is one citable passage, for example ``payment_plan-3``.
Search is BM25 keyword ranking; Week 2 added embeddings, left out here to keep the server light.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from rank_bm25 import BM25Okapi

PROJECT_DIR = Path(__file__).resolve().parent / "data" / "project"


@lru_cache(maxsize=1)
def passages() -> list[dict]:
    found: list[dict] = []
    for path in sorted(PROJECT_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        doc_title = re.search(r"^# (.+)$", text, re.M)
        doc_name = doc_title.group(1).split("—")[-1].strip() if doc_title else path.stem
        sections = re.findall(r"^## (.+?)\n(.*?)(?=^## |\Z)", text, re.M | re.S)
        for n, (heading, body) in enumerate(sections, 1):
            found.append({"id": f"{path.stem}-{n}", "title": f"{doc_name} › {heading.strip()}", "text": body.strip()})
    return found


STOPWORDS = set(
    "the and for are but not you your yours can will what when where which who whom how why does did have has had "
    "this that these those with from into about any all our its was were been being they them their there then than "
    "much many get got also may might shall should would could just only out per".split()
)


def _tokens(text: str) -> list[str]:
    return [t[:6] for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2 and t not in STOPWORDS]


@lru_cache(maxsize=1)
def _index() -> BM25Okapi:
    return BM25Okapi([_tokens(f"{p['title']} {p['text']}") for p in passages()])


# Everyday words buyers use, mapped to the words the documents use. Keyword search cannot see synonyms.
SYNONYMS = {
    "late": "delay",
    "delayed": "delay",
    "complain": "complaint grievance",
    "complaint": "grievance regulator",
    "money": "refund",
    "back": "refund",
    "keys": "possession handover",
    "flat": "apartment",
    "tax": "gst",
}


def search(query: str, k: int = 3) -> list[dict]:
    words = re.findall(r"[a-z0-9]+", query.lower())
    expanded = " ".join([query, *(SYNONYMS[w] for w in words if w in SYNONYMS)])
    scores = _index().get_scores(_tokens(expanded))
    best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [{**passages()[i], "score": round(float(scores[i]), 2)} for i in best if scores[i] > 0]


def documents() -> list[dict]:
    return [{"id": p["id"], "title": p["title"]} for p in passages()]
