"""The MCP client: connects to the real-estate server over stdio and runs a LangGraph tool loop on its tools.

The assistant knows nothing about real estate itself. On every question it starts the server as a
subprocess, reads the server's instructions and tool list over the MCP protocol, and lets the model
call those tools. The same server works unchanged in Claude Desktop, Cursor or any other MCP client.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .content import parse, text_of

ROOT = Path(__file__).resolve().parent.parent
MAX_STEPS = 16  # graph steps per question: about 7 rounds of tool calls

CLIENT_RULES = """You are a real-estate assistant. Everything you know comes from the tools of the MCP server
described above.
- Call tools before answering. Never answer from memory, and never compute a number yourself: if a different
  price or cost is asked about, call the tool again with that input.
- Quote figures exactly as the tools return them, in ₹ crore or lakh where the tool does.
- Cite what you used: passage ids (e.g. payment_plan-3), RERA section numbers, agreement and clause names.
- If the tools do not cover the question, say so in one sentence.
- Keep answers short and plain: a few sentences or a small table."""


class ProvidersUnavailable(RuntimeError):
    """Every configured model provider failed for this step."""


def server_params() -> StdioServerParameters:
    """Start the server the same way Claude Desktop would: as a subprocess speaking MCP over stdio."""
    return StdioServerParameters(command=sys.executable, args=["-m", "realestate_mcp"], cwd=str(ROOT))


@asynccontextmanager
async def connect():
    """An initialised MCP session with the server, plus its initialize result (name, version, instructions)."""
    async with stdio_client(server_params()) as (read, write), ClientSession(read, write) as session:
        init = await session.initialize()
        yield session, init


async def describe() -> dict:
    """What the server offers, read over the protocol."""
    async with connect() as (session, init):
        tools = (await session.list_tools()).tools
        resources = (await session.list_resources()).resources
        templates = (await session.list_resource_templates()).resourceTemplates
        prompts = (await session.list_prompts()).prompts
    return {
        "name": init.serverInfo.name,
        "version": init.serverInfo.version,
        "protocol": init.protocolVersion,
        "instructions": init.instructions or "",
        "tools": [
            {
                "name": t.name,
                "title": t.title or t.name,
                "description": t.description or "",
                "input_schema": t.inputSchema,
                "read_only": bool(t.annotations and t.annotations.readOnlyHint),
            }
            for t in tools
        ],
        "resources": [str(r.uri) for r in resources] + [t.uriTemplate for t in templates],
        "prompts": [
            {"name": p.name, "description": p.description or "", "arguments": [a.name for a in p.arguments or []]}
            for p in prompts
        ],
    }


async def call(name: str, arguments: dict) -> tuple[bool, Any]:
    """Call one tool directly. Returns (is_error, result)."""
    async with connect() as (session, _):
        result = await session.call_tool(name, arguments)
    if result.structuredContent is not None and not result.isError:
        return False, result.structuredContent
    return bool(result.isError), parse(result.content)


async def read_resource(uri: str) -> str:
    async with connect() as (session, _):
        result = await session.read_resource(uri)
    return "\n".join(getattr(c, "text", "") for c in result.contents)


async def get_prompt(name: str, arguments: dict) -> str:
    async with connect() as (session, _):
        result = await session.get_prompt(name, arguments)
    return "\n\n".join(text_of(m.content) for m in result.messages)


# ------------------------------------------------------------------ the agent
def build_graph(models: list[tuple[BaseChatModel, str]], tools: list[BaseTool], system: str):
    """reason -> call MCP tools -> observe, with provider fallback on every model step."""

    async def agent(state: MessagesState) -> dict:
        errors: list[str] = []
        for model, label in models:
            try:
                msg = await model.bind_tools(tools).ainvoke([SystemMessage(system), *state["messages"]])
            except Exception as exc:  # rate limit, rejected key, provider outage, bad tool format
                errors.append(f"{label}: {type(exc).__name__}: {str(exc)[:200]}")
                continue
            msg.response_metadata = {**(msg.response_metadata or {}), "answered_by": label, "fallback_errors": errors}
            return {"messages": [msg]}
        raise ProvidersUnavailable(" | ".join(errors))

    def route(state: MessagesState) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else END

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, ["tools", END])
    graph.add_edge("tools", "agent")
    return graph.compile()


def trace(messages: list[BaseMessage]) -> list[dict]:
    """Every MCP tool call in order, with its arguments and result."""
    results = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}
    calls = []
    for m in messages:
        if isinstance(m, AIMessage):
            for c in m.tool_calls:
                r = results.get(c["id"])
                calls.append(
                    {
                        "tool": c["name"],
                        "args": c["args"],
                        "error": bool(r is not None and r.status == "error"),
                        "result": parse(r.content) if r is not None else None,
                    }
                )
    return calls


def _leaf(exc: BaseException) -> BaseException:
    """The real error inside the task groups that the MCP client wraps around everything."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return exc


async def ask(question: str, history: list[BaseMessage], models: list[tuple[BaseChatModel, str]]) -> dict:
    """Answer one question. ``history`` holds earlier Human/AI turns (final answers only)."""
    try:
        return await _ask(question, history, models)
    except BaseExceptionGroup as group:
        raise _leaf(group) from None


async def _ask(question: str, history: list[BaseMessage], models: list[tuple[BaseChatModel, str]]) -> dict:
    async with connect() as (session, init):
        tools = await load_mcp_tools(session)
        system = f"{init.instructions or ''}\n\n{CLIENT_RULES}"
        graph = build_graph(models, tools, system)
        start = [*history, HumanMessage(question)]
        out = await graph.ainvoke({"messages": start}, {"recursion_limit": MAX_STEPS})
    new = out["messages"][len(start) :]
    final = new[-1] if new else AIMessage(content="")
    meta = final.response_metadata or {}
    errors = [
        e for m in new if isinstance(m, AIMessage) for e in (m.response_metadata or {}).get("fallback_errors", [])
    ]
    return {
        "answer": text_of(final.content).strip(),
        "calls": trace(new),
        "answered_by": meta.get("answered_by", ""),
        "fallback_errors": list(dict.fromkeys(errors)),
    }
