import json
import re
from pathlib import Path
from typing import Any

import streamlit as st

from src.bot import NovaCartPolicyBot

st.set_page_config(
    page_title="NovaCart Policy Assistant",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Subtle custom styling for comfortable reading and clean cards
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }
    .stChatMessage {
        border-radius: 12px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.75rem;
    }
    .citation-pill {
        display: inline-block;
        background-color: rgba(49, 130, 206, 0.12);
        border: 1px solid rgba(49, 130, 206, 0.35);
        border-radius: 6px;
        padding: 2px 8px;
        margin: 2px 4px 2px 0;
        font-family: monospace;
        font-size: 0.82rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def extract_structured_response(result: Any) -> dict[str, Any]:
    """
    Safely unpacks BotResponse (Pydantic v2/v1), dict, or raw string
    so it never renders as a single-line repr() dump.
    """
    if isinstance(result, str):
        # Fallback if a stringified Pydantic repr was passed
        match = re.search(r"answer=(['\"])(.*?)\1\s+citations_used=", result, flags=re.DOTALL)
        if match:
            raw_answer = match.group(2).encode("utf-8").decode("unicode_escape")
            return {"answer": raw_answer}
        return {"answer": result}

    if isinstance(result, dict):
        data = dict(result)
    elif hasattr(result, "model_dump"):
        data = result.model_dump()
    elif hasattr(result, "dict"):
        data = result.dict()
    else:
        data = {"answer": getattr(result, "answer", str(result))}

    # BotResponse exposes compatibility properties, but Pydantic serialization
    # contains the underlying field names. Normalize both shapes for the UI.
    if "citations_used" not in data:
        data["citations_used"] = data.get("citations", [])
    if "mandatory_disclosures" not in data:
        data["mandatory_disclosures"] = data.get("disclosures", [])
    if "query_context" not in data:
        data["query_context"] = data.get("context")
    if "active_policies" not in data:
        resolution = data.get("resolution") or {}
        data["active_policies"] = resolution.get("active", []) if isinstance(resolution, dict) else []

    return data


def format_markdown_readability(text: str) -> str:
    """
    Ensures markdown headings, numbered steps, and bullet points render on
    separate lines even if the LLM emitted inline formatting.
    """
    if not text:
        return "_No response generated._"

    # Unescape literal \n sequences if double-encoded
    cleaned = text.replace("\\n", "\n").strip()

    # Ensure blank lines before Markdown headings and numbered/bulleted lists
    cleaned = re.sub(r"([^\n])\s+(#{2,4}\s+)", r"\1\n\n\2", cleaned)
    cleaned = re.sub(r"([^\n])\s+(\*\*Step\s+\d+[:\*])", r"\1\n\n\2", cleaned)
    cleaned = re.sub(r"([.:])\s+(\d+\.\s+\*\*)", r"\1\n\2", cleaned)
    cleaned = re.sub(r"([.:])\s+(-\s+\*\*)", r"\1\n\2", cleaned)

    return cleaned


def render_assistant_turn(payload: dict[str, Any]) -> None:
    """Renders the formatted answer, disclosures, citations, and audit tabs cleanly."""
    answer_md = format_markdown_readability(payload.get("answer", ""))
    disclosures = payload.get("mandatory_disclosures") or []
    citations = payload.get("citations_used") or []
    active_policies = payload.get("active_policies") or []
    calc_trace = payload.get("calculation_trace")
    query_ctx = payload.get("query_context")

    # 1. Highlight mandatory policy version / conflict notices prominently
    for disc in disclosures:
        d_type = (
            disc.get("kind", disc.get("type", ""))
            if isinstance(disc, dict)
            else getattr(disc, "kind", getattr(disc, "type", ""))
        )
        msg = (
            disc.get("instruction") or disc.get("message") or str(disc)
            if isinstance(disc, dict)
            else getattr(disc, "instruction", str(disc))
        )
        if msg and msg.lower() in answer_md.lower():
            continue
        if d_type == "unresolved_conflict":
            st.warning(f"**Policy Conflict Notice:** {msg}", icon="⚠️")
        else:
            st.info(f"**Policy Version Notice:** {msg}", icon="ℹ️")

    # 2. Main nicely formatted Markdown answer
    st.markdown(answer_md)

    # 3. Clean Citation Pills right below the answer
    if citations:
        pills_html = "".join(
            f'<span class="citation-pill">📄 {c}</span>' for c in dict.fromkeys(citations)
        )
        st.markdown(f"**Sources Cited:**<br>{pills_html}", unsafe_allow_html=True)

    # 4. Comfortable tabbed inspection drawer for math & policy resolution details
    has_details = any([calc_trace, active_policies, query_ctx])
    if has_details:
        with st.expander("View Calculation Breakdown & Active Policy Details", expanded=False):
            tab_names = []
            if calc_trace:
                tab_names.append("🧮 Math Breakdown")
            if active_policies:
                tab_names.append("📚 Active Policies")
            if query_ctx:
                tab_names.append("🔎 Extracted Query Info")

            tabs = st.tabs(tab_names)
            idx = 0

            if calc_trace:
                with tabs[idx]:
                    if isinstance(calc_trace, dict) and "steps" in calc_trace:
                        for step in calc_trace["steps"]:
                            st.markdown(f"- {step}")
                    else:
                        st.json(calc_trace)
                idx += 1

            if active_policies:
                with tabs[idx]:
                    st.dataframe(active_policies, use_container_width=True, hide_index=True)
                idx += 1

            if query_ctx:
                with tabs[idx]:
                    st.json(query_ctx)


# --- Session State Initialization ---
if "bot" not in st.session_state:
    st.session_state.bot = NovaCartPolicyBot()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar: Corpus Status & One-Click Test Scenarios ---
with st.sidebar:
    st.header("🛒 NovaCart Policy Bot")
    st.caption("Grounded strictly in NovaCart policies (`00`–`21`)")

    report_path = Path("processed/validation_report.json")
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        col1, col2 = st.columns(2)
        col1.metric("Policies", report.get("documents_processed", 22))
        col2.metric("Rule Chunks", report.get("chunks_generated", "—"))
    else:
        st.error("Missing `processed/validation_report.json`. Run `python preprocess_policies.py`.")

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("Quick Test Scenarios")
    scenarios = {
        "📦 Promo + Return Refund (2026)": (
            "Today is 2026-06-15. I ordered a shirt for Rs. 2,500 using promo code "
            "STYLE15 and want to return it because I changed my mind. How much refund will I get?"
        ),
        "⚠️ COD Limit Conflict (Rs. 45,000)": (
            "Contact date: 2026-05-10. Can I place a Cash on Delivery order worth Rs. 45,000 in Lahore?"
        ),
        "🌍 International UK Order (GST-Free)": (
            "On 2026-04-20, I am ordering electronics worth Rs. 11,800 to be delivered to the UK. "
            "What is my total payable?"
        ),
        "🛠️ Prorated Warranty Refund": (
            "Today is 2026-04-12. My electronics item (paid Rs. 24,000, 12-month warranty) was "
            "delivered on 2026-01-10 and cannot be repaired or replaced. What is my warranty refund?"
        ),
        "🏔️ Remote Area Shipping (Gilgit)": (
            "Today is 2026-05-05. What are the shipping fees and delivery timeline for a Rs. 3,000 order to Gilgit?"
        ),
    }

    for label, prompt_text in scenarios.items():
        if st.button(label, use_container_width=True):
            st.session_state.pending_prompt = prompt_text

# --- Main Chat Area ---
st.title("NovaCart Policy Assistant")
st.markdown(
    "Ask any question about **returns, refunds, shipping fees, promo codes, warranties, or payments**. "
    "Include the **contact date** and **order date** for accurate version resolution."
)

if not st.session_state.messages:
    st.info(
        "💡 **Tip:** Pick a scenario from the left sidebar or type a question below with a date "
        "(e.g., *'Today is 2026-06-15...'*)."
    )

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_assistant_turn(msg["payload"])

# Capture user input
user_input = st.chat_input("Ask a NovaCart policy question (e.g., Today is 2026-06-15...)...")
if "pending_prompt" in st.session_state:
    user_input = st.session_state.pop("pending_prompt")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Checking active policies, cross-references, and exact Rupee math..."):
            try:
                raw_result = st.session_state.bot.answer(user_input)
                payload = extract_structured_response(raw_result)
            except Exception as exc:
                payload = {"answer": f"**Error processing request:** `{exc}`"}

            render_assistant_turn(payload)

    st.session_state.messages.append({"role": "assistant", "payload": payload})