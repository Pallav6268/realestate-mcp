"""A rule-based stand-in for the language model, used when no API key is set.

It picks one MCP tool from keywords in the question, calls it through the same MCP session as a live
model would, and summarises the result from a template. It shows the protocol working end to end;
it does not understand the question the way a language model does.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from .content import parse

AGREEMENT = re.compile(r"\b([ABC])-?(\d{3})\b", re.I)
PARCEL = re.compile(r"\bparcel[\s-]?([abc])\b", re.I)
SECTION = re.compile(r"\bsection\s+(\d+[A-Z]?)\b", re.I)
PRICE = re.compile(r"₹\s?([\d,]{4,})(?!\s*(?:crore|cr\b|lakh))")
EXTRA = re.compile(r"₹\s?(\d+(?:\.\d+)?)\s*(crore|cr\b|lakh)", re.I)
CLAUSES = {
    "possession": "possession",
    "maintenance": "maintenance",
    "cancel": "cancellation_by_allottee",
    "refund": "cancellation_by_allottee",
    "parking": "car_parking",
    "arbitration": "dispute_resolution",
    "dispute": "dispute_resolution",
    "payment": "payment_schedule",
    "delay": "delay_compensation",
}


def pick_tool(question: str) -> tuple[str, dict[str, Any]]:
    """Choose one tool and its arguments from keywords."""
    low = question.lower()
    if m := AGREEMENT.search(question):
        agreement = f"{m.group(1).upper()}-{m.group(2)}"
        clause = next((key for word, key in CLAUSES.items() if word in low), None)
        if clause:
            return "review_clause", {"agreement_id": agreement, "clause_key": clause}
        return "scan_agreement", {"agreement_id": agreement}
    if m := PARCEL.search(question):
        args: dict[str, Any] = {"parcel_id": f"parcel-{m.group(1).lower()}"}
        if p := PRICE.search(question):
            args["sale_price_per_sqft"] = float(p.group(1).replace(",", ""))
        if x := EXTRA.search(question):
            args["extra_costs_inr"] = float(x.group(1)) * (1e5 if x.group(2).lower() == "lakh" else 1e7)
        return "assess_parcel", args
    if "parcel" in low or "land" in low:
        return "list_parcels", {}
    if "agreement" in low:
        return "list_agreements", {}
    if m := SECTION.search(question):
        return "read_rera_section", {"section": m.group(1)}
    if "rera" in low or " act" in f" {low}":
        return "search_rera_act", {"query": question}
    return "search_project_docs", {"query": question}


def _summary(name: str, data: Any) -> str:
    if isinstance(data, str):
        return data[:600]
    if name == "scan_agreement":
        flagged = [r for r in data["clauses"] if r["clause"] in data["flagged"]]
        lines = [f"- **{r['title']}**: {r['status']}" for r in flagged]
        return f"{data['agreement_id']} has {len(flagged)} clauses to look at:\n" + "\n".join(lines)
    if name == "review_clause":
        if data["missing"]:
            return f"The **{data['clause']}** clause is missing. The template says: {data['template_text']}"
        v = data["value_check"]
        return (
            f"Template: {data['template_text']}\n\nAgreement: {data['agreement_text']}\n\n"
            f"Values only in the template: {v['only_in_template'] or 'none'}; "
            f"only in the agreement: {v['only_in_agreement'] or 'none'}."
        )
    if name == "assess_parcel":
        rec, fin = data["recommendation"], data["financials"]
        extra = fin["cost_breakdown_inr"].get("extra_what_if", 0)
        with_extra = f" with ₹{extra / 1e7:,.2f} crore of extra costs" if extra else ""
        return (
            f"**{rec['label']}** for {data['parcel']['name']} ({rec['rule']}). Margin {fin['margin_pct']}% at "
            f"₹{fin['sale_price_per_sqft_carpet']:,}/sq ft{with_extra}; breakeven "
            f"₹{fin['breakeven_price_per_sqft_carpet']:,}."
        )
    if name in ("search_project_docs", "search_rera_act"):
        if not data["results"]:
            return "Nothing in the documents matches that."
        top = data["results"][0]
        cite = top.get("id") or f"Section {top.get('section')}"
        return f"Closest match ({cite}, {top['title']}): {(top.get('text') or top.get('excerpt', ''))[:400]}"
    if name == "read_rera_section":
        return f"Section {data['section']}. {data['title']}: {data['text'][:500]}"
    return json.dumps(data, ensure_ascii=False)[:600]


class ScriptedToolPicker(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "scripted-tool-picker"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedToolPicker:
        return self

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    def _next(self, messages: list[BaseMessage]) -> AIMessage:
        last_human = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1)
        results = [m for m in messages[last_human + 1 :] if isinstance(m, ToolMessage)]
        if not results:
            name, args = pick_tool(str(messages[last_human].content) if last_human >= 0 else "")
            return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "demo_1", "type": "tool_call"}])
        msg = results[-1]
        data = parse(msg.content)
        return AIMessage(content=_summary(msg.name or "", data) + "\n\n_Demo mode: one tool, picked by keywords._")
