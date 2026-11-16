"""Real-estate MCP server: the tools from my Week 1-3 agents, available to any MCP client.

Run it over stdio for Claude Desktop, Cursor or any other MCP client:

    realestate-mcp                       # stdio (default)
    realestate-mcp --transport http      # streamable HTTP on 127.0.0.1:8000/mcp

Every tool is read-only and deterministic: the numbers come from plain Python, never from a model.
The data is fictional, apart from the RERA Act 2016 text.
"""

from __future__ import annotations

import argparse
import json
from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from . import __version__, agreements, feasibility, helpdesk, rera

INSTRUCTIONS = """Tools for Indian residential real estate, built on fictional sample data plus the official
RERA Act 2016 text.

- Agreements for sale: list_agreements, scan_agreement (clause-by-clause status against the developer's
  template), review_clause (both texts plus an exact value check). Judge wording yourself; the tools only
  compare values and find missing clauses.
- Buyer questions about the Riverside Heights project: search_project_docs. Cite the passage id.
- RERA Act 2016: search_rera_act, read_rera_section. Cite the section number.
- Land feasibility: list_parcels, assess_parcel (finance model, comparables, site and title rubrics,
  RERA registration check, weighted score and recommendation). Quote its numbers; do not compute your own.

All data except the Act is fictional. Nothing here is legal or investment advice."""

mcp = FastMCP("realestate-mcp", instructions=INSTRUCTIONS, log_level="WARNING")
READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)

AgreementId = Annotated[str, Field(description="Agreement ID from list_agreements, e.g. 'B-204'", max_length=40)]
ParcelId = Annotated[str, Field(description="Parcel ID from list_parcels, e.g. 'parcel-b'", max_length=40)]
Query = Annotated[str, Field(description="What to look for, in plain words", min_length=2, max_length=300)]
TopK = Annotated[int, Field(description="How many results to return", ge=1, le=8)]


def _parcel(parcel_id: str) -> dict[str, Any]:
    parcels = feasibility.load_parcels()
    if parcel_id not in parcels:
        raise ValueError(f"Unknown parcel '{parcel_id}'. Known: {sorted(parcels)}")
    return parcels[parcel_id]


# ------------------------------------------------------------------ agreements (Week 1)
@mcp.tool(title="List agreements", annotations=READ_ONLY)
def list_agreements() -> dict[str, Any]:
    """List the sample agreements for sale (IDs and titles). The developer's standard template has the ID 'template'."""
    docs = [{"id": d.doc_id, "title": d.title, "clauses": len(d.clauses)} for d in agreements.documents().values()]
    return {"agreements": docs, "template_id": agreements.TEMPLATE_ID}


@mcp.tool(title="Scan an agreement", annotations=READ_ONLY)
def scan_agreement(agreement_id: AgreementId) -> dict[str, Any]:
    """Compare every clause of an agreement with the template. Each clause gets a status: matches template,
    filled-in fields only, values differ (with the values on each side), wording differs, missing, or not in
    template. Use review_clause on the flagged clauses to read the text and judge the impact."""
    return agreements.scan(agreement_id)


@mcp.tool(title="Review one clause", annotations=READ_ONLY)
def review_clause(
    agreement_id: AgreementId,
    clause_key: Annotated[str, Field(description="Clause key, e.g. 'possession' or 'car_parking'", max_length=60)],
) -> dict[str, Any]:
    """The template text and the agreement text of one clause, plus an exact check of amounts, percentages,
    durations, areas and dates. The check does not judge wording; you do."""
    return agreements.review_clause(agreement_id, clause_key)


# ------------------------------------------------------------------ buyer helpdesk (Week 2)
@mcp.tool(title="Search project documents", annotations=READ_ONLY)
def search_project_docs(query: Query, k: TopK = 4) -> dict[str, Any]:
    """Keyword search over the Riverside Heights project documents (FAQ, payment plan, possession, cancellation
    and refund, maintenance, escalation contacts). Returns passages with an id to cite. If nothing fits, try
    the words the documents would use (e.g. 'delay compensation', 'forfeiture', 'corpus fund')."""
    return {"query": query, "results": helpdesk.search(query, k)}


# ------------------------------------------------------------------ RERA Act
@mcp.tool(title="Search the RERA Act", annotations=READ_ONLY)
def search_rera_act(query: Query, k: TopK = 3) -> dict[str, Any]:
    """Keyword search over the Real Estate (Regulation and Development) Act, 2016. Returns section numbers,
    titles and an excerpt; read the full section with read_rera_section before quoting it."""
    return {"query": query, "results": rera.search_rera(query, k)}


@mcp.tool(title="Read a RERA section", annotations=READ_ONLY)
def read_rera_section(
    section: Annotated[str, Field(description="Section number, e.g. '18' or '3'", max_length=12)],
) -> dict[str, Any]:
    """The text of one section of the RERA Act 2016 (first 4,000 characters)."""
    result = rera.read_rera_section(section)
    if "error" in result:
        raise ValueError(result["error"])
    return result


# ------------------------------------------------------------------ land feasibility (Week 3)
@mcp.tool(title="List land parcels", annotations=READ_ONLY)
def list_parcels() -> dict[str, Any]:
    """List the sample land parcels with their size, asking price and planned product."""
    parcels = [
        {
            "id": p["id"],
            "name": p["name"],
            "summary": p["summary"],
            "area_acres": p["land"]["area_acres"],
            "asking_price": feasibility.crore(p["land"]["asking_price_inr"]),
            "product": p["plan"]["product"],
        }
        for p in feasibility.load_parcels().values()
    ]
    return {"parcels": parcels}


@mcp.tool(title="Assess a land parcel", annotations=READ_ONLY)
def assess_parcel(
    parcel_id: ParcelId,
    sale_price_per_sqft: Annotated[
        float | None, Field(description="What-if sale price, ₹ per sq ft of carpet area", gt=0, le=100_000)
    ] = None,
    construction_cost_per_sqft: Annotated[
        float | None, Field(description="What-if construction cost, ₹ per sq ft built-up", gt=0, le=50_000)
    ] = None,
    extra_costs_inr: Annotated[
        float, Field(description="One-off costs not in the model, in ₹ (e.g. 10000000 for ₹1 crore)", ge=0, le=1e10)
    ] = 0,
) -> dict[str, Any]:
    """Full feasibility check of one parcel: revenue, cost breakdown, margin, annualised ROI and breakeven;
    margin if prices fall or costs rise 10%; comparable sales; site and title rubrics; RERA registration;
    rubric scores, weighted score and Go / Conditional go / No-go with the legal veto applied; and conflicts
    worth questioning. Pass a sale price, construction cost or extra cost to test a what-if."""
    p = _parcel(parcel_id)
    fin = feasibility.run_financials(p, sale_price_per_sqft, construction_cost_per_sqft, extra_costs_inr)
    title = feasibility.title_checklist(p)
    scores = feasibility.rubric_scores(p, sale_price_per_sqft, construction_cost_per_sqft)
    return {
        "parcel": {"id": p["id"], "name": p["name"], "summary": p["summary"]},
        "financials": fin,
        "sensitivity": feasibility.sensitivity(p, sale_price_per_sqft, construction_cost_per_sqft),
        "comparables": feasibility.comps_stats(p, fin["sale_price_per_sqft_carpet"]),
        "site": feasibility.site_risk(p),
        "title": title,
        "rera_registration": feasibility.rera_check(p),
        "rubric_scores": scores,
        "recommendation": feasibility.recommendation(scores, title["verdict"]),
        "conflicts": feasibility.tensions(p, sale_price_per_sqft),
        "method": "Rubric scores only. In the Week 3 panel each specialist may move its score by up to 2 points "
        "with a reason; here no model is involved. Extra costs change the financials but not the rubric scores.",
    }


# ------------------------------------------------------------------ resources
@mcp.resource("agreement://{agreement_id}", title="Agreement text", mime_type="text/markdown")
def agreement_text(agreement_id: str) -> str:
    """Full Markdown text of an agreement for sale, or of the template ('template')."""
    return agreements.get(agreement_id).text


@mcp.resource("parcel://{parcel_id}", title="Parcel dossier", mime_type="application/json")
def parcel_dossier(parcel_id: str) -> str:
    """The raw dossier of a land parcel: land, plan, title, site and market facts."""
    return json.dumps(_parcel(parcel_id), ensure_ascii=False, indent=1)


@mcp.resource("project://documents", title="Project document index", mime_type="application/json")
def project_documents() -> str:
    """Index of the Riverside Heights project document passages (id and title)."""
    return json.dumps(helpdesk.documents(), ensure_ascii=False, indent=1)


# ------------------------------------------------------------------ prompts
@mcp.prompt(title="Review an agreement")
def review_agreement(agreement_id: str) -> str:
    """Find every deviation in an agreement for sale and rate its severity."""
    return (
        f"Review agreement {agreement_id} against the developer's template.\n"
        f"1. Call scan_agreement('{agreement_id}').\n"
        "2. For every flagged clause, call review_clause and read both texts.\n"
        "3. Reply with a table: clause, template says, agreement says, severity (high = shifts money, risk or rights "
        "against the buyer; medium = should be reviewed; low = cosmetic), and one line on the impact.\n"
        "Quote values exactly as the tools return them. Do not report clauses the scan marked as matching."
    )


@mcp.prompt(title="Land feasibility memo")
def feasibility_memo(parcel_id: str) -> str:
    """Write a short investment-committee memo for a land parcel."""
    return (
        f"Call assess_parcel('{parcel_id}') and write a one-page memo for an investment committee:\n"
        "- Recommendation and weighted score, with the rule that produced it.\n"
        "- Three to five reasons, each with the number behind it.\n"
        "- The conflicts the tool lists, and what you would ask about each.\n"
        "- Conditions before signing.\n"
        "Use only numbers from the tool output, in ₹ crore or lakh. To test a different price or an extra cost, "
        "call assess_parcel again with those arguments rather than working it out yourself."
    )


@mcp.prompt(title="Answer a buyer")
def answer_buyer(question: str) -> str:
    """Answer a home buyer's question from the project documents and the RERA Act, with citations."""
    return (
        f"A buyer at Riverside Heights asks: {question}\n"
        "Search the project documents first (search_project_docs) and, where the law matters, the RERA Act "
        "(search_rera_act, then read_rera_section). Answer in plain words in under 150 words, cite passage ids "
        "and section numbers, and say so if the documents do not cover the question."
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="realestate-mcp", description=__doc__.split("\n\n")[0])
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--version", action="version", version=f"realestate-mcp {__version__}")
    args = parser.parse_args()
    if args.transport == "http":
        mcp.settings.host, mcp.settings.port = args.host, args.port
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
