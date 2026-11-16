"""Streamlit client for the real-estate MCP server.

Every question, tool call, resource read and prompt on this page goes to the server over the MCP protocol
(stdio), exactly as it would from Claude Desktop or Cursor.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from assistant import agent as G
from assistant.demo_model import ScriptedToolPicker
from assistant.llm import available_models

st.set_page_config(page_title="Real-estate MCP server", page_icon="🔌", layout="wide")

ROOT = Path(__file__).parent
REPO = "Pallav6268/realestate-mcp"
GITHUB = f"https://github.com/{REPO}"
LINKEDIN = "https://www.linkedin.com/in/pallav-kumar-ai"
WEEKS = {
    "Week 1 · Clause Deviation Agent": "https://clause-deviation-agent.streamlit.app",
    "Week 2 · Buyer Helpdesk Agent": "https://rera-buyer-agent.streamlit.app",
    "Week 3 · Land Feasibility Panel": "https://land-feasibility-panel.streamlit.app",
}
MODEL_SETTINGS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_MODEL",
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_MODEL",
    "GROQ_API_KEY",
    "GROQ_MODEL",
    "MODEL_ORDER",
)
EXAMPLES = [
    "What changed in agreement B-204, and how serious is each change?",
    "Parcel B: what's the margin at ₹5,500 per sq ft if drainage costs ₹1 crore?",
    "I want to cancel my booking. How much do I lose and when do I get the refund?",
    "My possession is late. What am I owed under RERA?",
]
TOOL_EXAMPLES = {
    "scan_agreement": {"agreement_id": "B-204"},
    "review_clause": {"agreement_id": "B-204", "clause_key": "cancellation_by_allottee"},
    "search_project_docs": {"query": "refund after cancellation", "k": 3},
    "search_rera_act": {"query": "compensation for delay in possession", "k": 3},
    "read_rera_section": {"section": "18"},
    "assess_parcel": {"parcel_id": "parcel-b", "sale_price_per_sqft": 5500, "extra_costs_inr": 10000000},
}
RESOURCE_URIS = [
    "agreement://template",
    "agreement://A-101",
    "agreement://B-204",
    "agreement://C-307",
    "parcel://parcel-a",
    "parcel://parcel-b",
    "parcel://parcel-c",
    "project://documents",
]
PROMPT_EXAMPLES = {
    "review_agreement": {"agreement_id": "B-204"},
    "feasibility_memo": {"parcel_id": "parcel-b"},
    "answer_buyer": {"question": "Can I transfer my booking to my brother?"},
}
DESKTOP_CONFIG = {"mcpServers": {"realestate": {"command": "realestate-mcp"}}}


def load_secrets_into_env() -> None:
    try:
        secrets = st.secrets.to_dict()
    except Exception:  # no secrets file (local run)
        return

    def put(mapping: dict) -> None:
        for key, value in mapping.items():
            if isinstance(value, dict):
                put(value)
            elif isinstance(value, str) and value.strip():
                os.environ[key.strip().upper()] = value.strip()

    put(secrets)


def settings_fingerprint() -> tuple:
    files = sorted([*(ROOT / "assistant").glob("*.py"), *(ROOT / "realestate_mcp").rglob("*.py")])
    code = hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest()[:12]
    return (code, *(hashlib.sha256(os.getenv(n, "").encode()).hexdigest()[:12] for n in MODEL_SETTINGS))


@st.cache_resource
def event_loop() -> asyncio.AbstractEventLoop:
    """One long-lived loop for every MCP session and model call. The cached model clients (Gemini's in
    particular) bind to the loop they first ran on, so a fresh asyncio.run() per question breaks them."""
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True, name="mcp-loop").start()
    return loop


def run(coro):
    return asyncio.run_coroutine_threadsafe(coro, event_loop()).result()


@st.cache_resource
def load_models(settings: tuple) -> tuple[list, str, bool]:
    models = available_models()
    if not models:
        return [(ScriptedToolPicker(), "Demo mode")], "Demo mode (rule-based tool picker, no API key)", False
    label = models[0][1] + (" (fallback: " + ", ".join(lbl for _, lbl in models[1:]) + ")" if models[1:] else "")
    return models, label, True


@st.cache_data(show_spinner="Connecting to the MCP server…")
def server_info(settings: tuple) -> dict:
    return run(G.describe())


def show_calls(calls: list[dict]) -> None:
    for c in calls:
        args = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in c["args"].items())
        st.markdown(f"{'⚠️' if c['error'] else '🔧'} `{c['tool']}({args})`")
        if isinstance(c["result"], (dict, list)):
            st.json(c["result"], expanded=False)
        else:
            st.code(str(c["result"])[:3000], language=None)


def show_answer(turn: dict) -> None:
    st.markdown(turn["answer"] or "_No answer._")
    n = len(turn["calls"])
    meta = f"Answered by {turn['answered_by']} · {n} MCP tool call{'s' if n != 1 else ''}"
    if turn["fallback_errors"]:
        meta += f" · fell back after: {'; '.join(turn['fallback_errors'])[:300]}"
    st.caption(meta)
    if n:
        with st.expander(f"MCP calls ({n})"):
            show_calls(turn["calls"])


# ------------------------------------------------------------------ page
load_secrets_into_env()
settings = settings_fingerprint()
models, model_label, live = load_models(settings)
info = server_info(settings)

with st.sidebar:
    st.subheader("MCP server")
    st.markdown(
        f"**{info['name']}**  \nMCP protocol `{info['protocol']}`  \n"
        f"{len(info['tools'])} tools · {len(info['resources'])} resources · {len(info['prompts'])} prompts"
    )
    st.caption("Connected over stdio: the page starts the server as a subprocess for every request.")
    st.subheader("Model")
    st.caption(model_label)
    if not live:
        st.info("No API key set, so a rule-based picker calls one tool per question. The tool explorer works fully.")
    st.divider()
    st.markdown("**Earlier weeks**  \n" + "  \n".join(f"[{k}]({v})" for k, v in WEEKS.items()))
    st.markdown(f"[Code on GitHub]({GITHUB}) · [Pallav Kumar]({LINKEDIN})")

st.title("🔌 Real-estate MCP server")
st.markdown(
    "Week 4 of my public AI-agent portfolio. One **MCP server** exposes the tools behind my first three agents: "
    "agreement review, buyer helpdesk search, the RERA Act and land feasibility. Any MCP client can use them. "
    "This page is one client; Claude Desktop is another."
)

ask_tab, explore_tab, desktop_tab = st.tabs(["💬 Ask", "🧰 Tool explorer", "🖥️ Use it in Claude Desktop"])

# ------------------------------------------------------------------ ask
with ask_tab:
    if "turns" not in st.session_state:
        st.session_state.turns = []

    def queue(question: str) -> None:
        st.session_state.pending = question

    cols = st.columns(len(EXAMPLES))
    for col, example in zip(cols, EXAMPLES, strict=True):
        col.button(example, on_click=queue, args=(example,), width="stretch")

    conversation = st.container()
    typed = st.chat_input("Ask about the agreements, the project, the RERA Act or the land parcels")
    question = typed or st.session_state.pop("pending", None)
    with conversation:
        for turn in st.session_state.turns:
            with st.chat_message("user"):
                st.markdown(turn["question"])
            with st.chat_message("assistant"):
                show_answer(turn)
        if question:
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                history = []
                for t in st.session_state.turns[-3:]:
                    history += [HumanMessage(t["question"]), AIMessage(t["answer"])]
                try:
                    with st.spinner("Calling MCP tools…"):
                        turn = run(G.ask(question, history, models))
                except Exception as exc:  # most often a free-tier limit on every provider
                    st.error(
                        "The AI providers did not answer, so nothing was generated. On free tiers this is usually a "
                        "temporary overload or a per-minute limit: try again in a minute. The tool explorer still "
                        f"works.\n\n{type(exc).__name__}: {str(exc)[:500]}"
                    )
                else:
                    turn["question"] = question
                    st.session_state.turns.append(turn)
                    show_answer(turn)

    if st.session_state.turns:
        st.button("Clear chat", on_click=lambda: st.session_state.update(turns=[]), type="tertiary")

# ------------------------------------------------------------------ explorer
with explore_tab:
    st.markdown(
        "Call the server's tools directly, without a model in between. This is what an MCP client sees: "
        "a tool list with JSON schemas, structured results, resources and prompt templates."
    )
    with st.expander("Server instructions (sent to every client at connection)"):
        st.markdown(info["instructions"])

    tools = {t["name"]: t for t in info["tools"]}
    names = list(tools)
    name = st.selectbox(
        "Tool",
        names,
        index=names.index("assess_parcel") if "assess_parcel" in names else 0,
        format_func=lambda n: f"{tools[n]['title']}  ·  {n}",
        key="tool_pick",
    )
    tool = tools[name]
    st.markdown(tool["description"])
    st.caption("Read-only" if tool["read_only"] else "")
    with st.expander("Input schema"):
        st.json(tool["input_schema"])
    with st.form("call_tool"):
        raw = st.text_area(
            "Arguments (JSON)", json.dumps(TOOL_EXAMPLES.get(name, {}), ensure_ascii=False), key=f"args_{name}"
        )
        submitted = st.form_submit_button("Call tool", type="primary")
    if submitted:
        try:
            arguments = json.loads(raw or "{}")
        except json.JSONDecodeError as exc:
            st.error(f"Arguments are not valid JSON: {exc}")
        else:
            is_error, result = run(G.call(name, arguments))
            if is_error:
                st.error(result if isinstance(result, str) else json.dumps(result))
            else:
                st.json(result)

    left, right = st.columns(2)
    with left:
        st.subheader("Resources")
        uri = st.selectbox("URI", RESOURCE_URIS)
        if st.button("Read resource"):
            text = run(G.read_resource(uri))
            lang = "json" if uri.startswith(("parcel", "project")) else "markdown"
            st.code(text[:6000], language=lang, wrap_lines=True)
    with right:
        st.subheader("Prompts")
        prompts = {p["name"]: p for p in info["prompts"]}
        pname = st.selectbox("Prompt", list(prompts))
        st.caption(prompts[pname]["description"])
        pargs = st.text_input("Arguments (JSON)", json.dumps(PROMPT_EXAMPLES.get(pname, {})), key=f"pargs_{pname}")
        if st.button("Get prompt"):
            try:
                st.code(run(G.get_prompt(pname, json.loads(pargs or "{}"))), language=None, wrap_lines=True)
            except Exception as exc:
                st.error(f"{type(exc).__name__}: {exc}")

# ------------------------------------------------------------------ desktop
with desktop_tab:
    st.markdown(
        "The same server runs on your own computer inside Claude Desktop (or Cursor, VS Code, or any MCP client). "
        "It needs [uv](https://docs.astral.sh/uv/) and git; no API key, because the tools do not call a model."
    )
    st.markdown("1. Install it once:")
    st.code(f"uv tool install git+{GITHUB}", language="bash")
    st.markdown(
        "2. In Claude Desktop open **Settings → Developer → Edit Config**, add this to "
        "`claude_desktop_config.json`, and restart Claude Desktop. If it can't find the command, use the full path "
        "from `where realestate-mcp` (Windows) or `which realestate-mcp` (macOS/Linux)."
    )
    st.code(json.dumps(DESKTOP_CONFIG, indent=2), language="json")
    st.markdown(
        '3. In a new chat, try *"Use the realestate tools to review agreement B-204"* or the '
        "**Land feasibility memo** prompt with `parcel-b`."
    )
    st.caption(
        "Running it with `uvx --from git+...` in the config also works, but it re-fetches the repo on every launch "
        "and can take longer than Claude Desktop waits for a server to start."
    )
    st.markdown("Or run it yourself:")
    st.code(
        "realestate-mcp                     # stdio, for MCP clients\n"
        "realestate-mcp --transport http    # streamable HTTP at http://127.0.0.1:8000/mcp",
        language="bash",
    )
