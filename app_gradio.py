"""Gradio chat UI for the NovaCart Policy Assistant (`python app_gradio.py`)."""
from __future__ import annotations

import html
import inspect
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any

import gradio as gr

PRIMARY, PRIMARY_DARK = "#0B6E69", "#3CCFC4"
ACCENT, ACCENT_DARK = "#F5B400", "#FFD25A"
MAX_CHARS, SLOW_SECONDS, POLL_SECONDS, MAX_WIDTH_PX = 1500, 20, 1.5, 860
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
ICONS = ("↩️", "💵", "🌍", "🛠️", "🏔️")
CITE_RE = re.compile(r"[\w.-]+\.md, Sections? \d+(?:\.\d+)*")

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&family=DM+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
:root {{ --p:{PRIMARY}; --a:{ACCENT}; --ink:#10231F; --muted:#5A6B68; --bg:#F3F6F6; --card:#FFFFFF; --line:#DCE5E3;
  --soft:#E3F1EF; --user:#0B6E69; --user-ink:#FFFFFF; --info:#0B6E69; --warn:#9A5B00; --warn-bg:#FFF4D6; --info-bg:#E3F1EF;
  --shadow:0 1px 2px rgba(16,35,31,.06),0 8px 24px -8px rgba(16,35,31,.14);
  --head:'Bricolage Grotesque','DM Sans',system-ui,sans-serif; --body:'DM Sans',system-ui,-apple-system,'Segoe UI',sans-serif;
  --mono:'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,monospace; }}
.dark {{ --p:{PRIMARY_DARK}; --a:{ACCENT_DARK}; --ink:#E8F2F0; --muted:#98ABA7; --bg:#0B1413; --card:#13211F; --line:#233835;
  --soft:#183230; --user:#3CCFC4; --user-ink:#06201E; --info:#3CCFC4; --warn:#FFD25A; --warn-bg:#2E2610; --info-bg:#163230;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px -10px rgba(0,0,0,.6); }}
body, .gradio-container {{ font-family:var(--body)!important; font-size:16px; line-height:1.55; color:var(--ink);
  background:radial-gradient(900px 380px at 80% -10%,color-mix(in srgb,var(--p) 14%,transparent),transparent),var(--bg)!important; }}
footer {{ display:none!important; }}
.gradio-container {{ max-width:{MAX_WIDTH_PX}px!important; margin:0 auto!important; padding:0 16px 24px!important; }}
.nc-bar {{ display:flex; align-items:center; gap:12px; padding:20px 2px 8px; }}
.nc-logo {{ width:40px; height:40px; border-radius:12px; display:grid; place-items:center; font-size:22px;
  background:linear-gradient(135deg,var(--p),color-mix(in srgb,var(--p) 55%,#000)); box-shadow:var(--shadow); }}
.nc-bar h1 {{ font:800 24px/1.1 var(--head); letter-spacing:-.02em; margin:0; flex:1; color:var(--ink); }}
.nc-bar small {{ display:block; font:500 13px var(--body); color:var(--muted); letter-spacing:0; margin-top:2px; }}
#nc-clear {{ margin-left:auto; min-height:44px; border-radius:999px!important; background:transparent!important;
  border:1px solid var(--line)!important; color:var(--muted)!important; }}
#nc-clear:hover {{ color:var(--ink)!important; border-color:var(--p)!important; }}
.nc-hero {{ padding:28px 4px 12px; }}
.nc-hero h2 {{ font:800 clamp(28px,5.5vw,44px)/1.05 var(--head); letter-spacing:-.03em; margin:0 0 10px; color:var(--ink); max-width:16ch; }}
.nc-hero p {{ margin:0; color:var(--muted); max-width:52ch; }}
.nc-chips {{ display:grid!important; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:10px!important; }}
.nc-chips button {{ min-height:56px; justify-content:flex-start!important; text-align:left; border-radius:16px!important; background:var(--card)!important;
  color:var(--ink)!important; border:1px solid var(--line)!important; font-weight:500; box-shadow:none; transition:transform .15s,border-color .15s,box-shadow .15s; }}
.nc-chips button:hover {{ border-color:var(--p)!important; box-shadow:var(--shadow); transform:translateY(-1px); }}
.message.user, .user.message, .message-row .user {{ background:var(--user)!important; color:var(--user-ink)!important; border:none!important;
  border-radius:20px 20px 6px 20px!important; max-width:85%; }}
.message.bot, .bot.message, .message-row .bot {{ background:var(--card)!important; color:var(--ink)!important; border:1px solid var(--line)!important;
  border-radius:20px 20px 20px 6px!important; box-shadow:var(--shadow); max-width:88%; }}
.nc-cite {{ font:400 12px var(--mono); color:var(--muted); white-space:nowrap; background:var(--soft); padding:1px 6px; border-radius:6px; }}
.nc-call {{ display:flex; gap:10px; padding:12px 14px; border-radius:14px; margin-bottom:10px; font-size:14px; }}
.nc-call.info {{ background:var(--info-bg); }} .nc-call.warn {{ background:var(--warn-bg); }}
.nc-call b {{ font-family:var(--head); }}
.nc-chip {{ display:inline-block; margin:3px 6px 3px 0; padding:4px 12px; border-radius:999px; background:var(--soft); font-size:13px; overflow-wrap:anywhere; }}
.nc-det {{ border-top:1px dashed var(--line); margin-top:10px; }}
.nc-det summary {{ cursor:pointer; color:var(--p); font:600 14px var(--body); min-height:44px; display:flex; align-items:center; }}
.nc-table {{ width:100%; font:400 13px var(--mono); border-collapse:collapse; display:block; overflow-x:auto; padding:8px 12px; margin-bottom:8px;
  background:var(--bg); border-radius:12px; border:1px solid var(--line); }}
.nc-table td, .nc-table th {{ padding:5px 4px; text-align:left; overflow-wrap:anywhere; border-bottom:1px dotted var(--line); }}
.nc-table td:last-child {{ text-align:right; font-weight:500; white-space:nowrap; }}
.nc-table th {{ font:800 14px var(--head); color:var(--ink); padding-top:10px; border-bottom:1px solid var(--line); }}
.nc-dots span {{ display:inline-block; width:8px; height:8px; margin-right:4px; border-radius:50%; background:var(--p); animation:nc-b 1s infinite; }}
.nc-dots span:nth-child(2) {{ animation-delay:.15s; }} .nc-dots span:nth-child(3) {{ animation-delay:.3s; }}
@keyframes nc-b {{ 0%,80%,100% {{ opacity:.25; transform:translateY(0); }} 40% {{ opacity:1; transform:translateY(-3px); }} }}
.nc-composer {{ background:var(--card); border:1px solid var(--line); border-radius:24px; padding:8px 8px 8px 16px; box-shadow:var(--shadow); align-items:center; }}
.nc-composer textarea {{ background:transparent!important; border:none!important; box-shadow:none!important; font-size:16px!important; }}
#nc-send {{ background:var(--p)!important; color:#fff!important; border:none!important; border-radius:999px!important; min-height:44px; font-weight:600; }}
.dark #nc-send {{ color:#06201E!important; }}
#nc-send:hover {{ filter:brightness(1.1); }}
.nc-cap {{ color:var(--muted); font-size:13px; }}
:focus-visible {{ outline:2px solid var(--p)!important; outline-offset:2px; }}
@media (prefers-reduced-motion:reduce) {{ * {{ animation:none!important; transition:none!important; }} }}
@media (max-width:480px) {{ .nc-bar h1 {{ font-size:20px; }} .message.bot, .message.user {{ max-width:96%; }} .nc-hero h2 {{ max-width:none; }} }}
"""


class NovaCartTheme(gr.themes.Soft):
    """Teal theme layered on Gradio's Soft theme; most styling lives in CSS."""

    def __init__(self) -> None:
        super().__init__(primary_hue="teal", neutral_hue="slate", radius_size="lg",
                         font=[gr.themes.GoogleFont("DM Sans"), "system-ui", "sans-serif"])


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
        with gr.Row(elem_classes="nc-barrow"):
            gr.HTML('<div class="nc-bar"><div class="nc-logo" aria-hidden="true">🛒</div>'
                    '<h1>NovaCart Policy Assistant<small>Answers grounded in our written policies</small></h1></div>')
            clear_btn = gr.Button("Clear chat", elem_id="nc-clear", size="sm", scale=0, min_width=110)
        with gr.Column(visible=True) as empty:
            gr.HTML('<div class="nc-hero"><h2>What can I help you sort out?</h2>'
                    '<p>Ask about returns, refunds, shipping, promo codes, warranty or payments. '
                    'Include your order date and city for the most accurate answer.</p></div>')
            with gr.Row(elem_classes="nc-chips"):
                chips = [gr.Button(f"{icon}  {label}", size="sm") for icon, label in zip(ICONS, EXAMPLES)]
        ckw = {"type": "messages"} if "type" in inspect.signature(gr.Chatbot.__init__).parameters else {}
        chat = gr.Chatbot(height=480, render_markdown=True, autoscroll=True, label="Conversation", **ckw)
        with gr.Row():
            retry = gr.Button("Try again", visible=False, size="sm")
            edit = gr.Button("Edit and resend", visible=False, size="sm")
        with gr.Row(elem_classes="nc-composer"):
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
