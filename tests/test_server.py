"""The MCP server, tested through a real client session (in memory, same protocol messages as stdio)."""

import asyncio

from mcp.shared.memory import create_connected_server_and_client_session

from realestate_mcp.server import mcp

TOOLS = {
    "list_agreements",
    "scan_agreement",
    "review_clause",
    "search_project_docs",
    "search_rera_act",
    "read_rera_section",
    "list_parcels",
    "assess_parcel",
}


def session_run(fn):
    async def main():
        async with create_connected_server_and_client_session(mcp._mcp_server) as session:
            return await fn(session)

    return asyncio.run(main())


def test_tool_list_is_complete_described_and_read_only():
    async def check(s):
        return (await s.list_tools()).tools

    tools = session_run(check)
    assert {t.name for t in tools} == TOOLS
    for t in tools:
        assert t.description and len(t.description) > 40, t.name
        assert t.annotations.readOnlyHint is True and t.annotations.destructiveHint is False
        assert t.outputSchema is not None, t.name  # structured output


def test_every_tool_answers_with_structured_content():
    calls = {
        "list_agreements": {},
        "scan_agreement": {"agreement_id": "A-101"},
        "review_clause": {"agreement_id": "A-101", "clause_key": "possession"},
        "search_project_docs": {"query": "maintenance charges"},
        "search_rera_act": {"query": "registration of real estate project"},
        "read_rera_section": {"section": "3"},
        "list_parcels": {},
        "assess_parcel": {"parcel_id": "parcel-a"},
    }

    async def run(s):
        return {name: await s.call_tool(name, args) for name, args in calls.items()}

    results = session_run(run)
    assert set(results) == TOOLS
    for name, r in results.items():
        assert not r.isError, (name, r.content)
        assert isinstance(r.structuredContent, dict) and r.structuredContent, name


def test_assess_parcel_what_if_and_verdicts():
    async def run(s):
        what_if = await s.call_tool(
            "assess_parcel", {"parcel_id": "parcel-b", "sale_price_per_sqft": 5500, "extra_costs_inr": 10_000_000}
        )
        verdicts = {}
        for pid in ("parcel-a", "parcel-b", "parcel-c"):
            verdicts[pid] = (await s.call_tool("assess_parcel", {"parcel_id": pid})).structuredContent
        return what_if.structuredContent, verdicts

    what_if, verdicts = session_run(run)
    assert what_if["financials"]["margin_pct"] == 13.7
    assert what_if["financials"]["breakeven_price_per_sqft_carpet"] == 4745
    labels = {pid: v["recommendation"]["label"] for pid, v in verdicts.items()}
    assert labels == {"parcel-a": "Go", "parcel-b": "Conditional go", "parcel-c": "No-go"}
    assert "forces No-go" in verdicts["parcel-c"]["recommendation"]["rule"]  # legal veto


def test_bad_input_returns_tool_errors_not_crashes():
    async def run(s):
        return [
            await s.call_tool("assess_parcel", {"parcel_id": "parcel-z"}),
            await s.call_tool("assess_parcel", {"parcel_id": "parcel-a", "sale_price_per_sqft": -5}),
            await s.call_tool("read_rera_section", {"section": "999"}),
            await s.call_tool("review_clause", {"agreement_id": "B-204", "clause_key": "nope"}),
            await s.call_tool("search_project_docs", {"query": "x" * 400}),
        ]

    for r in session_run(run):
        assert r.isError


def test_resources_and_prompts():
    async def run(s):
        templates = [t.uriTemplate for t in (await s.list_resource_templates()).resourceTemplates]
        static = [str(r.uri) for r in (await s.list_resources()).resources]
        agreement = (await s.read_resource("agreement://B-204")).contents[0].text
        parcel = (await s.read_resource("parcel://parcel-c")).contents[0].text
        prompts = [p.name for p in (await s.list_prompts()).prompts]
        memo = (await s.get_prompt("feasibility_memo", {"parcel_id": "parcel-b"})).messages[0].content.text
        return templates, static, agreement, parcel, prompts, memo

    templates, static, agreement, parcel, prompts, memo = session_run(run)
    assert templates == ["agreement://{agreement_id}", "parcel://{parcel_id}"]
    assert static == ["project://documents"]
    assert agreement.startswith("# Agreement for Sale — Apartment B-204")
    assert '"id": "parcel-c"' in parcel
    assert prompts == ["review_agreement", "feasibility_memo", "answer_buyer"]
    assert "assess_parcel('parcel-b')" in memo


def test_server_sends_instructions_at_connection():
    async def run(s):
        return s

    # The in-memory helper initialises for us; check the instructions the server will send.
    assert "assess_parcel" in mcp.instructions and "fictional" in mcp.instructions
