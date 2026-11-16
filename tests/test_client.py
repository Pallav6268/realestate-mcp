"""The Streamlit client's MCP path: a real server subprocess over stdio, with scripted models."""

import asyncio

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from assistant import agent as G
from assistant.demo_model import ScriptedToolPicker, pick_tool


class TwoToolModel(BaseChatModel):
    """Asks for two tools at once, then answers from their results."""

    @property
    def _llm_type(self) -> str:
        return "two-tool"

    def bind_tools(self, tools, **kwargs):
        self._names = [t.name for t in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if not any(isinstance(m, ToolMessage) for m in messages):
            calls = [
                {"name": "scan_agreement", "args": {"agreement_id": "B-204"}, "id": "c1", "type": "tool_call"},
                {"name": "read_rera_section", "args": {"section": "18"}, "id": "c2", "type": "tool_call"},
            ]
            msg = AIMessage(content="", tool_calls=calls)
        else:
            msg = AIMessage(content="B-204 changes the cancellation clause; Section 18 covers delay.")
        return ChatResult(generations=[ChatGeneration(message=msg)])


class Broken(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "broken"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise RuntimeError("429 rate limit")


def test_describe_over_stdio():
    info = asyncio.run(G.describe())
    assert info["name"] == "realestate-mcp"
    assert len(info["tools"]) == 8 and all(t["read_only"] for t in info["tools"])
    assert "fictional" in info["instructions"]


def test_direct_call_over_stdio():
    is_error, result = asyncio.run(G.call("scan_agreement", {"agreement_id": "B-204"}))
    assert not is_error and result["flagged"] == ["cancellation_by_allottee", "car_parking", "dispute_resolution"]
    is_error, result = asyncio.run(G.call("assess_parcel", {"parcel_id": "nope"}))
    assert is_error and "Unknown parcel" in str(result)


def test_resource_and_prompt_over_stdio():
    assert "Car Parking" in asyncio.run(G.read_resource("agreement://template"))
    assert "scan_agreement('B-204')" in asyncio.run(G.get_prompt("review_agreement", {"agreement_id": "B-204"}))


def test_agent_runs_parallel_mcp_calls():
    out = asyncio.run(G.ask("What changed in B-204?", [], [(TwoToolModel(), "Two-tool")]))
    assert [c["tool"] for c in out["calls"]] == ["scan_agreement", "read_rera_section"]
    assert out["calls"][0]["result"]["agreement_id"] == "B-204"
    assert out["calls"][1]["result"]["section"] == "18"
    assert out["answered_by"] == "Two-tool" and out["answer"].startswith("B-204")


def test_fallback_to_next_provider_is_recorded():
    out = asyncio.run(G.ask("Assess parcel C", [], [(Broken(), "Broken"), (ScriptedToolPicker(), "Demo")]))
    assert out["answered_by"] == "Demo"
    assert out["fallback_errors"] and "429" in out["fallback_errors"][0]
    assert "No-go" in out["answer"]


def test_all_providers_failing_raises_a_clear_error():
    with pytest.raises(Exception, match="429"):
        asyncio.run(G.ask("hello", [], [(Broken(), "Broken")]))


def test_tool_errors_go_back_to_the_model_instead_of_crashing():
    out = asyncio.run(G.ask("What does section 999 say?", [], [(ScriptedToolPicker(), "Demo")]))
    assert out["calls"][0]["error"] is True


@pytest.mark.parametrize(
    ("question", "tool"),
    [
        ("What changed in B-204?", "scan_agreement"),
        ("Is parking the same in A-101?", "review_clause"),
        ("Assess parcel B at ₹5,500", "assess_parcel"),
        ("What does section 18 say?", "read_rera_section"),
        ("refund if I cancel", "search_project_docs"),
    ],
)
def test_demo_picker(question, tool):
    assert pick_tool(question)[0] == tool


def test_demo_picker_reads_price_and_extra_cost():
    name, args = pick_tool("Parcel B: what's the margin at ₹5,500 per sq ft if drainage costs ₹1 crore?")
    assert args == {"parcel_id": "parcel-b", "sale_price_per_sqft": 5500.0, "extra_costs_inr": 10_000_000.0}
    out = asyncio.run(G.ask("Parcel B at ₹5,500 with ₹1 crore drainage", [], [(ScriptedToolPicker(), "Demo")]))
    assert "13.7%" in out["answer"] and "₹4,745" in out["answer"]
