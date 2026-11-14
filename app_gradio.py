"""Material-style Gradio chat UI for the NovaCart Policy Assistant (`python app_gradio.py`)."""
from __future__ import annotations

import html
import inspect
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any

import gradio as gr

PRIMARY, PRIMARY_DARK = "#3F51B5", "#9FA8DA"
INFO, WARNING, SUCCESS = "#1A73E8", "#B26A00", "#1E8E3E"
MAX_CHARS, SLOW_SECONDS, POLL_SECONDS, MAX_WIDTH_PX = 1500, 20, 1.5, 880
LOADING_TEXTS = ("Checking our policies...", "Working out the numbers...", "Double-checking the details...")
PROCESSED = ("processed/policy_documents.json", "processed/policy_chunks.json")
SETUP_MSG = "**Setup needed.** The policy index is missing. Run `python preprocess_policies.py`, then restart."
ERROR_MSG = "Sorry, something went wrong while looking that up. Please try again."
NOT_MONEY = ("day", "days", "month", "months", "year", "years", "percent", "rate", "count", "number", "version", "pct")
EXAMPLES = {
    "Return refund with promo": "Today is 2026-06-15. I ordered a shirt for Rs. 2,500 using promo code STYLE15 and want to return it because I changed my mind. How much refund will I get?",
    "COD limit": "Contact date: 2026-05-10. Can I place a Cash on Delivery order worth Rs. 45,000 in Lahore?",
    "International order": "On 2026-04-20, I am ordering electronics worth Rs. 11,800 to be delivered to the UK. What is my total payable?",
    "Warranty refund": "Today is 2026-04-12. My electronics item (paid Rs. 24,000, 12-month warranty) was delivered on 2026-01-10 and cannot be repaired or replaced. What is my warranty refund?",
    "Remote-area shipping": "Today is 2026-05-05. What are the shipping fees and delivery timeline for a Rs. 3,000 order to Gilgit?",
}
CITE_RE = re.compile(r"[\w.-]+\.md, Sections? \d+(?:\.\d+)*")

CSS = f"""
:root {{ --nc-primary:{PRIMARY}; --nc-bg:#F6F7FB; --nc-card:#FFFFFF; --nc-text:#1F2330; --nc-muted:#5B6275;
  --nc-info:{INFO}; --nc-warn:{WARNING}; --nc-ok:{SUCCESS}; --nc-tint:#E8EAF6;
  --nc-e1:0 1px 2px rgba(20,24,40,.12),0 1px 3px rgba(20,24,40,.08); --nc-e2:0 2px 6px rgba(20,24,40,.14),0 4px 12px rgba(20,24,40,.08); }}
.dark {{ --nc-primary:{PRIMARY_DARK}; --nc-bg:#121212; --nc-card:#1E1E24; --nc-text:#ECEDF2; --nc-muted:#A9AEBD;
  --nc-info:#8AB4F8; --nc-warn:#FDD663; --nc-ok:#81C995; --nc-tint:#2B2F4A; }}
body, .gradio-container {{ background:var(--nc-bg)!important; color:var(--nc-text); font-size:16px; line-height:1.5;
  font-family:'Roboto','Inter',system-ui,-apple-system,'Segoe UI',sans-serif; }}
footer {{ display:none!important; }}
.gradio-container {{ max-width:{MAX_WIDTH_PX}px!important; margin:0 auto!important; padding:0 12px!important; }}
.nc-bar {{ display:flex; align-items:center; gap:12px; padding:12px 4px; }}
.nc-bar h1 {{ font-size:24px; margin:0; flex:1; color:var(--nc-text); }}
.nc-greet {{ background:var(--nc-card); border-radius:16px; padding:24px; box-shadow:var(--nc-e1); margin-bottom:8px; }}
.nc-greet p {{ margin:4px 0; }} .nc-tip, .nc-cap {{ color:var(--nc-muted); font-size:13px; }}
.nc-chips button, .nc-pill {{ border-radius:999px!important; min-height:44px; }}
.nc-chips button {{ background:var(--nc-tint)!important; color:var(--nc-text)!important; border:none!important; transition:box-shadow .18s; }}
.nc-chips button:hover {{ box-shadow:var(--nc-e2); }}
#nc-send {{ background:var(--nc-primary)!important; color:#fff!important; border-radius:999px!important; min-height:44px; }}
.dark #nc-send {{ color:#121212!important; }}
#nc-clear {{ margin-left:auto; min-width:44px; min-height:44px; border-radius:999px!important; }}
.message.bot, .bot.message {{ background:var(--nc-card)!important; border-radius:16px!important; box-shadow:var(--nc-e1); max-width:85%; }}
.message.user, .user.message {{ background:var(--nc-tint)!important; border-radius:16px!important; max-width:85%; }}
.nc-cite {{ font-size:12px; color:var(--nc-muted); white-space:nowrap; }}
.nc-call {{ display:flex; gap:8px; padding:12px 16px; border-radius:12px; margin-bottom:8px; border-left:4px solid; background:var(--nc-bg); }}
.nc-call.info {{ border-color:var(--nc-info); }} .nc-call.warn {{ border-color:var(--nc-warn); }}
.nc-chip {{ display:inline-block; margin:2px 4px 2px 0; padding:4px 12px; border-radius:999px; background:var(--nc-tint); font-size:13px; overflow-wrap:anywhere; }}
.nc-det summary {{ cursor:pointer; color:var(--nc-primary); font-size:14px; min-height:44px; display:flex; align-items:center; }}
.nc-table {{ width:100%; font-size:14px; border-collapse:collapse; display:block; overflow-x:auto; }}
.nc-table td, .nc-table th {{ padding:6px 8px; text-align:left; overflow-wrap:anywhere; }}
.nc-table th {{ color:var(--nc-muted); font-weight:500; }}
.nc-dots span {{ display:inline-block; width:8px; height:8px; margin-right:4px; border-radius:50%; background:var(--nc-primary); animation:nc-b 1s infinite; }}
.nc-dots span:nth-child(2) {{ animation-delay:.15s; }} .nc-dots span:nth-child(3) {{ animation-delay:.3s; }}
@keyframes nc-b {{ 0%,80%,100% {{ opacity:.25; }} 40% {{ opacity:1; }} }}
:focus-visible {{ outline:2px solid var(--nc-primary)!important; outline-offset:2px; }}
button, .message, details {{ transition:all .18s ease; }}
@media (prefers-reduced-motion:reduce) {{ * {{ animation:none!important; transition:none!important; }} }}
@media (max-width:480px) {{ .nc-bar h1 {{ font-size:20px; }} .message.bot, .message.user {{ max-width:95%; }} }}
"""


class NovaCartTheme(gr.themes.Soft):
    """Indigo Material-like theme layered on Gradio's Soft theme."""

    def __init__(self) -> None:
        super().__init__(primary_hue="indigo", neutral_hue="slate",
                         font=[gr.themes.GoogleFont("Roboto"), "system-ui", "sans-serif"], radius_size="lg")


# ---------- pure helpers ----------
def friendly_source(citation: str) -> str:
    """'14_store_credit.md, Section 3.1' -> 'Store credit · Section 3.1' (display only)."""
    name, _, section = citation.partition(", ")
    label = re.sub(r"^\d+_", "", name.removesuffix(".md")).replace("_", " ").strip()
    label = label[:1].upper() + label[1:]
    return f"{label} · {section}" if section else label


def humanize_key(key: str) -> str:
    """'step_3_discount_amount' -> 'Discount amount'."""
    text = re.sub(r"^step_?\d+_?", "", str(key)).replace("_", " ").strip() or str(key)
    return text[:1].upper() + text[1:]


def format_value(key: str, value: Any) -> str:
    """Format a trace value: whole numbers as 'Rs. 1,932' unless the key is not money."""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float)):
        if set(re.findall(r"[a-z]+", key.lower())) & set(NOT_MONEY):
            return f"{value:,}"
        return f"Rs. {round(value):,}"
    if isinstance(value, (list, tuple)):
        return ", ".join(format_value(key, v) for v in value)
    return str(value)


def _flatten(prefix: str, data: Any) -> list[tuple[str, str]]:
    if isinstance(data, dict):
        return [row for k, v in data.items()
                for row in _flatten(f"{prefix} · {humanize_key(k)}" if prefix else humanize_key(k), v)]
    return [(prefix, format_value(prefix, data))]


def render_math_table(trace: list[Any]) -> str:
    """Render calculation traces as a two-column HTML table (escaped)."""
    rows = []
    for item in trace or []:
        name = item.get("name", "") if isinstance(item, dict) else getattr(item, "name", "")
        result = item.get("result", {}) if isinstance(item, dict) else getattr(item, "result", {})
        rows.append(f'<tr><th colspan="2">{html.escape(humanize_key(name))}</th></tr>')
        rows += [f"<tr><td>{html.escape(k)}</td><td>{html.escape(v)}</td></tr>" for k, v in _flatten("", result)]
    return f'<table class="nc-table">{"".join(rows)}</table>' if rows else ""


def mute_citations(text: str) -> str:
    """Wrap verbatim citations in a muted, non-wrapping span without changing their text."""
    return CITE_RE.sub(lambda m: f'<span class="nc-cite">{m.group(0)}</span>', text)


def pending_disclosures(disclosures: list[Any], answer: str) -> list[tuple[str, str]]:
    """(kind, message) pairs not already covered by the answer; `missing_date` is skipped."""
    seen, out = answer.lower(), []
    for d in disclosures or []:
        kind, msg = (d.get("kind", ""), d.get("message", "")) if isinstance(d, dict) else (d.kind, d.message)
        if kind != "missing_date" and msg and msg.lower() not in seen and (kind, msg) not in out:
            out.append((kind, msg))
    return out


def format_answer_html(payload: dict[str, Any]) -> str:
    """Compose the structured assistant message (answer, callouts, sources, math)."""
    answer = str(payload.get("answer") or "_No response generated._").replace("\\n", "\n")
    parts = []
    for kind, msg in pending_disclosures(payload.get("disclosures", []), answer):
        warn = kind == "unresolved_conflict"
        title, icon = ("Our policies disagree", "⚠️") if warn else ("Good to know", "ℹ️")
        parts.append(f'<div class="nc-call {"warn" if warn else "info"}"><span>{icon}</span>'
                     f"<div><b>{title}</b><br>{html.escape(msg)}</div></div>")
    parts.append(mute_citations(answer))
    cites = list(dict.fromkeys(payload.get("citations", [])))
    if cites:
        chips = "".join(f'<span class="nc-chip" title="{html.escape(c)}">📄 {html.escape(friendly_source(c))}</span>' for c in cites)
        parts.append(f'<details class="nc-det"><summary>Where this comes from ({len(cites)})</summary>{chips}</details>')
    table = render_math_table(payload.get("calculation_trace", []))
    if table:
        parts.append(f'<details class="nc-det"><summary>How it was calculated</summary>{table}</details>')
    return "\n\n".join(parts)


def needs_more_info(payload: dict[str, Any]) -> bool:
    """True when the bot is asking the customer for a missing detail."""
    kinds = [d.get("kind") if isinstance(d, dict) else d.kind for d in payload.get("disclosures", [])]
    traces = payload.get("calculation_trace", [])
    return "missing_date" in kinds or any(getattr(t, "missing_inputs", None) for t in traces)


# ---------- bot access ----------
_bot: Any = None


def get_bot() -> Any:
    """Lazy module-level singleton; raises FileNotFoundError if the index is missing."""
    global _bot
    if _bot is None:
        if not all(Path(p).exists() for p in PROCESSED):
            raise FileNotFoundError(SETUP_MSG)
        from src.bot import NovaCartPolicyBot
        _bot = NovaCartPolicyBot()
    return _bot


def ask_bot(query: str) -> dict[str, Any]:
    """Run the bot and return a plain dict payload for rendering."""
    res = get_bot().answer(query)
    return {"answer": res.answer, "citations": res.citations, "disclosures": res.disclosures,
            "calculation_trace": res.calculation_trace, "injection_detected": res.injection_detected}


def typing_html(text: str) -> str:
    """Animated typing indicator with status text."""
    return f'<div class="nc-dots" role="status" aria-live="polite"><span></span><span></span><span></span> {html.escape(text)}</div>'


# ---------- handlers ----------
def respond(message: str, history: list[dict], last_q: str):
    """Generator: echo the question, show a typing indicator, then the structured answer."""
    text = (message or "").strip()
    hide = gr.update(visible=False)
    if not text:
        yield history, gr.update(), gr.update(), gr.update(), last_q, hide, hide, "Type a question first."
        return
    if len(text) > MAX_CHARS:
        yield history, gr.update(), gr.update(), gr.update(), last_q, hide, hide, f"Please shorten your question to {MAX_CHARS} characters."
        return
    history = history + [{"role": "user", "content": html.escape(text)}]
    busy = gr.update(interactive=False)
    shown = lambda t: history + [{"role": "assistant", "content": typing_html(t)}]  # noqa: E731
    yield shown(LOADING_TEXTS[0]), gr.update(value="", interactive=False), busy, hide, text, hide, hide, ""
    pool, error, payload = ThreadPoolExecutor(max_workers=1), False, {}
    future, tick = pool.submit(ask_bot, text), 0
    while True:
        try:
            payload = future.result(timeout=POLL_SECONDS)
            break
        except FutureTimeout:
            tick += 1
            label = "Still working on it..." if tick * POLL_SECONDS > SLOW_SECONDS else LOADING_TEXTS[tick % len(LOADING_TEXTS)]
            yield shown(label), gr.update(), gr.update(), hide, text, hide, hide, ""
        except FileNotFoundError:
            payload, error = {"answer": SETUP_MSG}, True
            break
        except Exception:  # noqa: BLE001 - never leak internals to the customer
            payload, error = {"answer": ERROR_MSG}, True
            break
    pool.shutdown(wait=False)
    history = history + [{"role": "assistant", "content": format_answer_html(payload)}]
    yield (history, gr.update(interactive=True), gr.update(interactive=True), hide, text,
           gr.update(visible=error), gr.update(visible=needs_more_info(payload) and not error), "")


def clear_chat():
    """Reset the conversation and show the empty state again."""
    hide = gr.update(visible=False)
    return [], gr.update(value="", interactive=True), gr.update(visible=True), "", hide, hide, ""


def count_chars(text: str) -> str:
    """Live character counter."""
    return f"{len(text or '')}/{MAX_CHARS}"


def build_app() -> gr.Blocks:
    """Assemble the Blocks UI."""
    legacy = "css" in inspect.signature(gr.Blocks.__init__).parameters
    kw = {"theme": NovaCartTheme(), "css": CSS} if legacy else {}
    with gr.Blocks(title="NovaCart Policy Assistant", **kw) as demo:
        last_q = gr.State("")
        gr.HTML('<div class="nc-bar"><span style="font-size:28px" aria-hidden="true">🛒</span>'
                '<h1>NovaCart Policy Assistant</h1></div>')
        clear_btn = gr.Button("🗑️ Clear chat", elem_id="nc-clear", size="sm", variant="secondary")
        with gr.Column(visible=True) as empty:
            gr.HTML('<div class="nc-greet"><p><b>Hi! Ask me about returns, refunds, shipping, promo codes, '
                    'warranty or payments.</b></p><p class="nc-tip">Include your order date and city for the most accurate answer.</p></div>')
            with gr.Row(elem_classes="nc-chips"):
                chips = [gr.Button(label, size="sm") for label in EXAMPLES]
        ckw = {"type": "messages"} if "type" in inspect.signature(gr.Chatbot.__init__).parameters else {}
        chat = gr.Chatbot(height=480, render_markdown=True, autoscroll=True, label="Conversation", **ckw)
        with gr.Row():
            retry = gr.Button("Try again", visible=False, size="sm")
            edit = gr.Button("Edit and resend", visible=False, size="sm")
        with gr.Row():
            box = gr.Textbox(placeholder="e.g. On 2026-06-15 in Karachi I bought a shirt for Rs. 2,400... how much will I get back?",
                             lines=1, max_lines=6, show_label=False, container=False, scale=8, autofocus=True)
            send = gr.Button("Send", elem_id="nc-send", variant="primary", scale=1, min_width=96)
        hint = gr.Markdown("", elem_classes="nc-cap")
        counter = gr.Markdown(count_chars(""), elem_classes="nc-cap")
        gr.Markdown("Each question is answered on its own, so include the details (date, city, item) every time.", elem_classes="nc-cap")

        outs = [chat, box, send, empty, last_q, retry, edit, hint]
        pre = lambda: gr.update(visible=False)  # noqa: E731
        for trigger in (box.submit, send.click):
            trigger(respond, [box, chat, last_q], outs, concurrency_limit=4).then(pre, None, empty)
        for chip, text in zip(chips, EXAMPLES.values()):
            chip.click(lambda t=text: t, None, box).then(respond, [box, chat, last_q], outs, concurrency_limit=4).then(pre, None, empty)
        retry.click(lambda q: q, last_q, box).then(respond, [box, chat, last_q], outs)
        edit.click(lambda q: (q, gr.update(visible=False)), last_q, [box, edit])
        box.change(count_chars, box, counter, show_progress="hidden")
        clear_btn.click(clear_chat, None, [chat, box, empty, last_q, retry, edit, hint])
        demo.queue()
    demo._nc_launch_kw = {} if legacy else {"theme": NovaCartTheme(), "css": CSS}  # Gradio 6 takes these at launch
    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(**app._nc_launch_kw)
