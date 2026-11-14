import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bot import CITATION_RE, NovaCartPolicyBot


RAW = (
    "Your store credit would be Rs. 1,932 (14_store_credit.md, Section 3.1).\n\n"
    "* Paid price: Rs. 2,040 (01_returns_policy.md, Section 5.1).\n"
    "* Store-credit bonus: Rs. 92 (14_store_credit.md, Section 3.1). Important notes: The files differ: "
    "02_returns_policy_2025.md and 01_returns_policy.md. (01_returns_policy.md, Section 4.2, "
    "02_returns_policy_2025.md, Section 4.2)"
)


def test_polish_splits_bullets_and_adds_friendly_heading():
    out = NovaCartPolicyBot._polish_answer(RAW)
    assert "\n- Paid price" in out and "\n- Store-credit bonus" in out
    assert "\n\n**A quick note on our policies**\n" in out
    assert "Important notes" not in out


def test_polish_never_changes_numbers_files_or_citations():
    out = NovaCartPolicyBot._polish_answer(RAW)
    assert sorted(set(CITATION_RE.findall(out))) == sorted(set(CITATION_RE.findall(RAW)))
    for token in ("Rs. 1,932", "Rs. 2,040", "Rs. 92", "02_returns_policy_2025.md", "01_returns_policy.md"):
        assert token in out


def test_polish_is_idempotent():
    once = NovaCartPolicyBot._polish_answer(RAW)
    assert NovaCartPolicyBot._polish_answer(once) == once
