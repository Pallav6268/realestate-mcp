"""Agreements for sale (reused from Week 1): split Markdown into clauses and compare them with the template.

Each clause is a level-2 heading such as ``## 5. Possession``; its key is the title in snake_case
(``possession``), so the same clause can be looked up in the template and in any agreement. Amounts,
percentages, durations, areas and dates are extracted with regular expressions and compared exactly.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

AGREEMENTS_DIR = Path(__file__).resolve().parent / "data" / "agreements"
TEMPLATE_ID = "template"

_HEADING = re.compile(r"^##\s+(?:\d+\.\s*)?(.+?)\s*$", re.MULTILINE)
_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("date", re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTHS})\s+(\d{{4}})\b")),
    ("money", re.compile(r"(?:₹|Rs\.?|INR)\s?(\d[\d,]*(?:\.\d+)?)")),
    ("percent", re.compile(r"(\d+(?:\.\d+)?)\s?%")),
    ("area", re.compile(r"(\d[\d,]*(?:\.\d+)?)\s?(?:sq\.?\s?ft|square feet)", re.IGNORECASE)),
    ("duration", re.compile(r"\b(\d+)\s+(day|month|year)s?\b", re.IGNORECASE)),
]
PLACEHOLDER = re.compile(r"\{\{\s*[a-z_]+\s*\}\}")


@dataclass
class Clause:
    key: str
    title: str
    text: str


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    clauses: dict[str, Clause] = field(default_factory=dict)


def slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


def parse_markdown(doc_id: str, text: str) -> Document:
    first_line = text.strip().splitlines()[0] if text.strip() else doc_id
    doc = Document(doc_id=doc_id, title=first_line.lstrip("# ").strip() or doc_id, text=text)
    matches = list(_HEADING.finditer(text))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        title = match.group(1).strip()
        doc.clauses[slugify(title)] = Clause(slugify(title), title, text[match.end() : end].strip())
    return doc


@lru_cache(maxsize=1)
def documents() -> dict[str, Document]:
    return {p.stem: parse_markdown(p.stem, p.read_text(encoding="utf-8")) for p in sorted(AGREEMENTS_DIR.glob("*.md"))}


def get(doc_id: str) -> Document:
    docs = documents()
    if doc_id not in docs:
        raise ValueError(f"Unknown document '{doc_id}'. Known: {sorted(docs)}")
    return docs[doc_id]


def agreement_ids() -> list[str]:
    return [d for d in documents() if d != TEMPLATE_ID]


# ------------------------------------------------------------------ value check
def extract_values(text: str) -> list[str]:
    """Normalised values found in ``text``, for example ``'12 months'`` or ``'₹500000'``."""
    text = PLACEHOLDER.sub(" ", text)
    found: list[str] = []
    consumed: list[tuple[int, int]] = []
    for kind, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            if not all(m.end() <= s or m.start() >= e for s, e in consumed):
                continue
            consumed.append(m.span())
            if kind == "date":
                found.append(f"{int(m.group(1))} {m.group(2)} {m.group(3)}")
            elif kind == "money":
                found.append("₹" + m.group(1).replace(",", ""))
            elif kind == "percent":
                found.append(f"{m.group(1)}%")
            elif kind == "area":
                found.append(m.group(1).replace(",", "") + " sq ft")
            else:
                found.append(f"{m.group(1)} {m.group(2).lower()}s")
    return sorted(found)


def diff_values(template_text: str, agreement_text: str) -> dict:
    t, a = Counter(extract_values(template_text)), Counter(extract_values(agreement_text))
    variable = bool(PLACEHOLDER.search(template_text))
    only_t, only_a = sorted((t - a).elements()), sorted((a - t).elements())
    # In a clause with placeholders (unit, price), extra agreement values are the filled-in fields.
    return {
        "template_values": sorted(t.elements()),
        "agreement_values": sorted(a.elements()),
        "only_in_template": only_t,
        "only_in_agreement": only_a,
        "variable_clause": variable,
        "values_deviate": bool(only_t) or (bool(only_a) and not variable),
    }


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def same_wording(template_text: str, agreement_text: str) -> bool:
    """True if the agreement repeats the template word for word, with any {{placeholder}} filled in."""
    parts = PLACEHOLDER.split(_normalise(template_text))
    # A filled field is a short value (a name, unit number or amount), not an extra phrase.
    field = r"((?:(?!, |; |\. ).){1,60}?)"
    pattern = field.join(re.escape(p) for p in parts)
    return re.fullmatch(pattern, _normalise(agreement_text)) is not None


def review_clause(agreement_id: str, clause_key: str) -> dict:
    template = get(TEMPLATE_ID).clauses.get(clause_key)
    if template is None:
        raise ValueError(f"'{clause_key}' is not a template clause. Known: {list(get(TEMPLATE_ID).clauses)}")
    agreement = get(agreement_id).clauses.get(clause_key)
    if agreement is None:
        return {"clause": clause_key, "template_text": template.text, "agreement_text": None, "missing": True}
    return {
        "clause": clause_key,
        "template_text": template.text,
        "agreement_text": agreement.text,
        "missing": False,
        "value_check": diff_values(template.text, agreement.text),
    }


def scan(agreement_id: str) -> dict:
    """Clause-by-clause status against the template, without judging wording."""
    if agreement_id == TEMPLATE_ID:
        raise ValueError("Pick an agreement, not the template.")
    doc, template = get(agreement_id), get(TEMPLATE_ID)
    rows = []
    for key, t in template.clauses.items():
        a = doc.clauses.get(key)
        if a is None:
            rows.append({"clause": key, "title": t.title, "status": "missing"})
            continue
        values = diff_values(t.text, a.text)
        if values["values_deviate"]:
            status = "values differ"
        elif not same_wording(t.text, a.text):
            status = "wording differs"
        else:
            status = "filled-in fields only" if values["variable_clause"] else "matches template"
        row = {"clause": key, "title": t.title, "status": status}
        if values["values_deviate"]:
            row["only_in_template"], row["only_in_agreement"] = values["only_in_template"], values["only_in_agreement"]
        rows.append(row)
    rows += [
        {"clause": k, "title": c.title, "status": "not in template"}
        for k, c in doc.clauses.items()
        if k not in template.clauses
    ]
    flagged = [r["clause"] for r in rows if r["status"] not in ("matches template", "filled-in fields only")]
    return {
        "agreement_id": agreement_id,
        "clauses": rows,
        "flagged": flagged,
        "note": "Values and missing clauses are compared exactly. 'wording differs' means the text changed but "
        "the numbers did not: read the clause with review_clause to judge it. 'filled-in fields only' means the "
        "template wording is unchanged apart from its placeholders (buyer, unit, price).",
    }
