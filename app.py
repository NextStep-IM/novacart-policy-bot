import json
from pathlib import Path
import streamlit as st

# Adjust this import to match your Step 3/4 bot entrypoint in src/bot.py
from src.bot import NovaCartPolicyBot

st.set_page_config(page_title="NovaCart Policy Q&A Bot", page_icon="🛒", layout="wide")

st.title("🛒 NovaCart Policy Q&A Bot")
st.caption("Tech Fusion '26 · Challenge A — Grounded strictly in NovaCart Policy Files (00–21)")

# Initialize the bot once in session state
if "bot" not in st.session_state:
    st.session_state.bot = NovaCartPolicyBot()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar: Helpful sample trap questions & corpus status
with st.sidebar:
    st.header("System Status")
    report_path = Path("processed/validation_report.json")
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        st.success(f"Corpus Status: {report.get('status', 'READY')}")
        st.write(f"• Documents loaded: **{report.get('documents_processed', 22)}**")
        st.write(f"• Policy chunks: **{report.get('chunks_generated', 'N/A')}**")
        st.write(f"• Injections blocked: **{report.get('injections_removed', {}).get('total', 0)}**")
    else:
        st.warning("Run `python preprocess_policies.py` first.")

    st.divider()
    st.header("Sample Test Queries")
    sample_prompts = [
        "Today is 2026-06-15. I ordered a shirt for Rs. 2,500 using code STYLE15 and want to return it because I changed my mind. How much refund will I get?",
        "Contact date: 2026-05-10. Can I place a Cash on Delivery order worth Rs. 45,000 in Lahore?",
        "On 2026-04-20, I am ordering electronics worth Rs. 11,800 to be delivered to the UK. What is my total payable?",
        "Today is 2026-04-12. My electronics item (paid Rs. 24,000) was delivered on 2026-01-10 and cannot be repaired or replaced. What is my warranty refund?",
    ]
    for i, sample in enumerate(sample_prompts):
        if st.button(f"Test Case {i+1}", key=f"sample_{i}", use_container_width=True):
            st.session_state.pending_prompt = sample

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("debug"):
            with st.expander("🔍 View Active Policies, Citations & Math Trace"):
                st.json(msg["debug"])

# Handle input from either chat box or sidebar test button
user_input = st.chat_input("Ask a NovaCart policy question (include contact/order dates)...")
if "pending_prompt" in st.session_state:
    user_input = st.session_state.pop("pending_prompt")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Checking active policies, cross-references, and exact math..."):
            result = st.session_state.bot.answer(user_input)

            # Handle whether bot.answer() returns a Pydantic object, dict, or plain string
            if isinstance(result, str):
                answer_text = result
                debug_data = None
            elif isinstance(result, dict):
                answer_text = result.get("answer", str(result))
                debug_data = result
            else:
                answer_text = getattr(result, "answer", str(result))
                debug_data = result.model_dump() if hasattr(result, "model_dump") else None

            st.markdown(answer_text)
            if debug_data:
                with st.expander("🔍 View Active Policies, Citations & Math Trace"):
                    st.json(debug_data)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "debug": debug_data,
    })
