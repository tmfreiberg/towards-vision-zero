"""Checks on the explorer page itself.

The page is a static file with no build step, so nothing else would notice a
string referenced in one language and defined in only the other. Two bugs of
exactly that kind shipped: four copy keys added to neither language block, and
field labels that fell back to the variable identifier in both. Both rendered as
`undefined` or as `PUB_PRIV_RD` in front of a reader.

These parse the file and compare the keys each language defines against the keys
the code actually reads.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parents[2] / "app" / "index.html"


@pytest.fixture(scope="module")
def source() -> str:
    return PAGE.read_text(encoding="utf-8")


def _copy_table(source: str) -> tuple[str, str]:
    """Split the page into the COPY table and everything after it.

    Located by matching braces rather than by whatever identifier happens to
    follow. An earlier version anchored on the next `const` declaration and
    broke the moment that constant was renamed, which is not a fault worth
    reporting as a test failure.
    """
    start = source.index("const COPY = {")
    depth = 0
    for position in range(start, len(source)):
        character = source[position]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return source[start : position + 1], source[position + 1 :]
    raise AssertionError("the COPY table is not brace-balanced")


@pytest.fixture(scope="module")
def blocks(source: str) -> dict[str, str]:
    """The two language blocks of the COPY table, as raw text."""
    table, _ = _copy_table(source)
    # Slice past each block's own label, or `en:` and `fr:` are read as keys.
    english = table[table.index("en: {") + len("en: {") : table.index("fr: {")]
    french = table[table.index("fr: {") + len("fr: {") :]
    return {"en": english, "fr": french}


def _defined(block: str) -> set[str]:
    """Keys defined in one language block.

    Keys sit at the start of a line or after a comma; matching only line starts
    would miss those sharing a line with a neighbour, which is how the first
    version of this test passed while two keys were missing.
    """
    return set(re.findall(r"(?:^|[{,])\s*([a-zA-Z][a-zA-Z0-9]*)\s*:", block, flags=re.MULTILINE))


def _referenced(body: str) -> set[str]:
    """Keys the page reads, by either route into the table.

    The renderer usually binds `const copy = COPY[lang]` first, but not always,
    and a check that saw only one form reported a key as unused while it was
    being read through the other.
    """
    return set(re.findall(r"\bcopy\.([a-zA-Z][a-zA-Z0-9]*)", body)) | set(
        re.findall(r"\bCOPY\[[a-zA-Z]+\]\.([a-zA-Z][a-zA-Z0-9]*)", body)
    )


def test_the_page_exists_and_is_self_contained(source: str) -> None:
    assert source.count("<script") == 1
    assert source.count("<style") == 1


def test_both_languages_define_the_same_keys(blocks: dict[str, str]) -> None:
    """A key present in one language renders as `undefined` in the other."""
    english, french = _defined(blocks["en"]), _defined(blocks["fr"])
    assert english == french, (
        f"only en: {sorted(english - french)}, only fr: {sorted(french - english)}"
    )


def test_every_key_the_code_reads_is_defined(source: str, blocks: dict[str, str]) -> None:
    """The bug that shipped: four keys referenced by the renderer and never added."""
    _, body = _copy_table(source)
    used = _referenced(body)
    defined = _defined(blocks["en"])
    assert used <= defined, f"referenced but never defined: {sorted(used - defined)}"


def test_no_key_is_defined_and_never_used(source: str, blocks: dict[str, str]) -> None:
    """Dead copy drifts out of step with the page and misleads the next editor."""
    _, body = _copy_table(source)
    unused = _defined(blocks["en"]) - _referenced(body)
    assert not unused, f"defined but never used: {sorted(unused)}"


def test_the_citation_is_present_in_both_languages(blocks: dict[str, str]) -> None:
    for block in blocks.values():
        assert "donneesquebec" in block
        assert "SAAQ" in block


def test_field_labels_come_from_the_payload_not_the_identifier(source: str) -> None:
    """`feature.label[lang]` must be read, so a translated name can appear."""
    assert "feature.label[lang]" in source


def test_the_page_makes_no_claim_about_safety(source: str) -> None:
    """Above the marker means opposite things for the mildest and severest class."""
    for phrase in ("safer than average", "safer than", "is worse"):
        assert phrase not in source


def test_accessibility_floor_is_met(source: str) -> None:
    assert "prefers-reduced-motion" in source
    assert "focus-visible" in source
    assert 'aria-live="polite"' in source
    assert "aria-pressed" in source


def test_the_page_shows_official_definitions(source: str) -> None:
    """A reader asking what a public road is should not have to leave the page."""
    assert "v.definition[lang]" in source
    assert "feature.summary" in source
    assert "feature.notes" in source


def test_outcome_definitions_disclose_a_merged_class(source: str) -> None:
    """The three-level scale merges two categories; that must be visible."""
    assert "meta.target.merged" in source
    assert "copy.combines" in source


def test_the_severity_ramp_has_a_step_for_every_published_level(source: str) -> None:
    assert source.count("--sev-") >= 4


def test_the_page_can_fold_the_published_scale(source: str) -> None:
    """Counts and the base rate must be folded together, or shares mislead."""
    assert "activeScale" in source
    assert "fold(raw.counts" in source
    assert "fold(meta.target.base_rate" in source


def test_a_merged_scale_discloses_that_it_merges(source: str) -> None:
    assert "copy.mergedNote" in source


def test_the_provenance_line_describes_scope_not_a_filter_state(blocks: dict[str, str]) -> None:
    """It reports what was loaded. A phrase naming a filter contradicts itself
    as soon as the reader applies one."""
    for block in blocks.values():
        assert "all regions" not in block
        assert "toutes les régions" not in block


def test_the_payload_is_revalidated_rather_than_cached(source: str) -> None:
    """The payload is regenerated whenever the data or schema changes.

    A cached `meta.json` describes a field list the records no longer match,
    and the page shows the old fields with no indication anything is wrong.
    `no-cache` revalidates, so an unchanged payload still costs one round trip
    rather than five megabytes.
    """
    assert 'cache: "no-cache"' in source


def test_a_missing_payload_is_reported_rather_than_guessed_at(source: str) -> None:
    assert "payload not found" in source


def test_the_masthead_names_the_page_not_the_publisher(source: str, blocks: dict[str, str]) -> None:
    """The publisher's name at the top made this look like the publisher's tool."""
    masthead = source[source.index('class="eyebrow"') : source.index("</header>")]
    assert "assurance automobile" not in masthead
    for block in blocks.values():
        assert "independent" in block or "indépendante" in block


def test_the_page_disclaims_affiliation(blocks: dict[str, str]) -> None:
    """Showing someone's data is not the same as speaking for them."""
    assert "not affiliated" in blocks["en"]
    assert "pas affiliée" in blocks["fr"]
