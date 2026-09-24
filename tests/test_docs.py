"""Guard rails for the documentation: numbers must come from results/, no placeholders."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
SUMMARY = ROOT / "results" / "summary_tables.md"
DOCS = ["README.md", "RESUME_BULLETS.md", "LINKEDIN_POST.md", "INTERVIEW_PREP.md", "MORNING_REPORT.md",
        "PROGRESS.md", "DECISIONS.md", "figures/CAPTIONS.md"]


def _readme_block() -> str:
    text = README.read_text(encoding="utf-8")
    match = re.search(r"<!-- results:start -->\n(.*?)<!-- results:end -->", text, flags=re.S)
    assert match, "README.md is missing the <!-- results:start/end --> block"
    return match.group(1)


def test_readme_results_match_generated_tables() -> None:
    assert _readme_block().strip() == SUMMARY.read_text(encoding="utf-8").strip()


@pytest.mark.parametrize("name", DOCS)
def test_no_placeholders_in_docs(name: str) -> None:
    path = ROOT / name
    if not path.exists():
        pytest.skip(f"{name} not written yet")
    text = path.read_text(encoding="utf-8")
    for marker in ("TODO", "TBD", "FIXME", "XXX", "lorem ipsum", "PLACEHOLDER"):
        assert marker.lower() not in text.lower(), f"{name} contains {marker!r}"
