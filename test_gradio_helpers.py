"""Tests for pure helpers in app_gradio.py."""
from types import SimpleNamespace

from app_gradio import (friendly_source, format_answer_html, mute_citations,
                        pending_disclosures, render_math_table)


def test_friendly_source():
    assert friendly_source("14_store_credit.md, Section 3.1") == "Store credit · Section 3.1"


def test_math_table_humanizes_and_escapes():
    html_out = render_math_table([{"name": "refund", "result": {"step_3_discount": 1932, "note": "<b>x</b>"}}])
    assert "Rs. 1,932" in html_out and "Discount" in html_out
    assert "step_3" not in html_out and "<b>x</b>" not in html_out


def test_disclosure_dedup_and_skip():
    discs = [SimpleNamespace(kind="superseded_version", message="Version 2 applies."),
             SimpleNamespace(kind="missing_date", message="Need a date."),
             SimpleNamespace(kind="unresolved_conflict", message="Files disagree.")]
    out = pending_disclosures(discs, "Note: version 2 applies. See below.")
    assert out == [("unresolved_conflict", "Files disagree.")]


def test_citation_text_unchanged():
    cite = "06_shipping_fees_and_free_shipping.md, Section 2.2"
    out = mute_citations(f"Fee applies ({cite}).")
    assert cite in out and "nc-cite" in out
    assert cite in format_answer_html({"answer": f"x ({cite})"})
