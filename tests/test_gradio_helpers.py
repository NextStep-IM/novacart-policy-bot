import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app_gradio import _disclosure_covered, friendly_source, render_math_table


def test_friendly_source():
    assert friendly_source("14_store_credit.md, Section 3.1") == "Store credit · Section 3.1"


def test_render_math_table_formats_money_and_hides_internal_step_prefix():
    result = render_math_table([{"name": "checkout", "result": {"step_3_discount": 500}}])
    assert "Discount" in result
    assert "Rs. 500" in result
    assert "step_3_discount" not in result


def test_disclosure_is_deduplicated_when_files_are_already_in_answer():
    disclosure = {
        "kind": "superseded_version",
        "files": ["01_returns_policy.md", "02_returns_policy_2025.md"],
        "message": "The files differ.",
    }
    answer = "I used 01_returns_policy.md and 02_returns_policy_2025.md because the files differ."
    assert _disclosure_covered(answer, disclosure)
