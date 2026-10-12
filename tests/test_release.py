"""The release metadata agrees with itself, so a tagged release can't ship half-updated."""

import re
from pathlib import Path

import areafluforecast as af

ROOT = Path(__file__).resolve().parents[1]


def test_citation_matches_the_version():
    citation = (ROOT / "CITATION.cff").read_text()
    assert re.search(r"^version: (\S+)$", citation, re.M).group(1) == af.__version__


def test_changelog_has_an_entry_for_the_version():
    assert f"## [{af.__version__}]" in (ROOT / "CHANGELOG.md").read_text()
