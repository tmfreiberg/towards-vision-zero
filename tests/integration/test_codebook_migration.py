"""Verify the YAML codebook against the Python modules it was generated from.

This is a migration guard. It runs only while the source modules are still on
disk, and skips once they are gone — at which point the YAML is the sole
authority and this file can be deleted with it.

Its purpose is narrow and specific: to establish, before anything is deleted,
that the transformation of roughly 2,300 lines of nested Python literals into
YAML preserved every variable, value, label, gloss, and translation exactly,
with no deviation other than the ones declared in ``DECLARED_DEVIATIONS``.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

from vision_zero.data import load_codebook

REPO_ROOT = Path(__file__).resolve().parents[2]

# The source modules are split across two directories: the dictionary and the
# variable maps in one, its duplicated copy in the other. Both go on the path.
SOURCE_DIRS = [REPO_ROOT / "legacy_src", REPO_ROOT / "scripts", REPO_ROOT / "streamlit"]
REQUIRED_MODULES = ("saaq_data_dictionary.py", "dictionaries.py", "variables.py")

DECLARED_DEVIATIONS = """
1. The empty-string value is emitted as a `missing` block rather than as an
   entry in `values`, on the thirteen variables the published documentation
   shows a "Non précisé" row for. ASPECT and ACCDN_TYPE gain one (their gloss
   was present but had no value to attach to, so it was never displayed);
   HOUR gains one (it had neither).
2. Labels are coerced to strings. Code-based variables previously carried
   integer labels; the readable name lives in `short`.
3. `shorthand` is merged across the two source copies, which disagree on eight
   variables where one holds display labels and the other holds None.
4. RDWX documents one value where `class_codes` encodes two. The published
   documentation lists only the positive value, so the negative one is a
   modelling inference and belongs in the encoding layer, not the codebook.
"""

# The sole variable whose encoding exceeds its published values. See item 4.
UNDOCUMENTED_NEGATIVE = "RDWX"


def _load_sources() -> Any:
    """Import the source modules, or skip if they are no longer present."""
    present = [d for d in SOURCE_DIRS if d.is_dir()]
    found = {
        module: directory
        for module in REQUIRED_MODULES
        for directory in present
        if (directory / module).exists()
    }
    if len(found) < len(REQUIRED_MODULES):
        pytest.skip("source modules removed; the YAML codebook is now the sole authority")

    for directory in {str(d) for d in found.values()}:
        if directory not in sys.path:
            sys.path.insert(0, directory)

    return (
        importlib.import_module("saaq_data_dictionary"),
        importlib.import_module("dictionaries"),
    )


@pytest.fixture(scope="module")
def sources() -> Any:
    return _load_sources()


def test_variable_set_is_preserved(sources: Any) -> None:
    legacy, _ = sources
    book = load_codebook()
    assert set(book) == set(legacy.FR_EN.values())


def test_source_columns_are_preserved(sources: Any) -> None:
    legacy, _ = sources
    book = load_codebook()
    for fr_term, en_term in legacy.FR_EN.items():
        assert book[en_term].source_column == fr_term


def test_every_value_and_gloss_is_preserved(sources: Any) -> None:
    """The load-bearing assertion: no value or gloss lost in transformation."""
    legacy, _ = sources
    book = load_codebook()

    for fr_term, en_term in legacy.FR_EN.items():
        explain_en = legacy.explain_en.get(en_term, {})
        explain_fr = legacy.explain_fr.get(fr_term, {})

        source_en = [v for v in list(explain_en.get("values") or []) if v != ""]
        source_fr = [v for v in list(explain_fr.get("values") or []) if v != ""]
        gloss_en = list(explain_en.get("elaborate") or [])
        gloss_fr = list(explain_fr.get("elaborate") or [])

        assert list(book[en_term].values) == [str(v) for v in source_en], en_term

        for index, value in enumerate(source_en):
            entry = book[en_term].values[str(value)]
            assert entry.label.en == str(value)
            if index < len(source_fr):
                assert entry.label.fr == str(source_fr[index]), f"{en_term}[{value}] FR label"
            if index < len(gloss_en):
                assert entry.gloss is not None
                assert entry.gloss.en == gloss_en[index], f"{en_term}[{value}] EN gloss"
            if index < len(gloss_fr):
                assert entry.gloss is not None
                assert entry.gloss.fr == gloss_fr[index], f"{en_term}[{value}] FR gloss"


def test_summaries_are_preserved(sources: Any) -> None:
    legacy, _ = sources
    book = load_codebook()
    for fr_term, en_term in legacy.FR_EN.items():
        source_en = legacy.explain_en.get(en_term, {}).get("summary")
        source_fr = legacy.explain_fr.get(fr_term, {}).get("summary")
        if source_en or source_fr:
            summary = book[en_term].summary
            assert summary is not None
            assert summary.en == source_en
            assert summary.fr == source_fr


def test_short_labels_take_the_richer_of_the_two_copies(sources: Any) -> None:
    """One copy holds display labels for eight variables where the other holds None."""
    legacy, extra = sources
    book = load_codebook()
    undocumented: set[tuple[str, str]] = set()

    for en_term in book:
        merged = dict(legacy.shorthand.get(en_term) or {})
        merged.update(extra.shorthand.get(en_term) or {})

        for value, label in merged.items():
            if value == "":
                continue
            entry = book[en_term].values.get(str(value))
            if entry is None:
                undocumented.add((en_term, str(value)))
                continue
            assert entry.short is not None
            assert entry.short.en == str(label), f"{en_term}[{value}]"

    assert undocumented == {(UNDOCUMENTED_NEGATIVE, "N")}, (
        "a short label exists for a value the documentation does not define"
    )


def test_french_short_labels_are_preserved(sources: Any) -> None:
    legacy, extra = sources
    book = load_codebook()

    for fr_term, en_term in legacy.FR_EN.items():
        raw_en = list(legacy.explain_en.get(en_term, {}).get("values") or [])
        raw_fr = list(legacy.explain_fr.get(fr_term, {}).get("values") or [])
        source_en = [v for v in raw_en if v != ""]
        source_fr = [v for v in raw_fr if v != ""]
        short_fr = extra.shorthand_fr.get(fr_term) or {}

        for index, value in enumerate(source_en):
            if index >= len(source_fr):
                continue
            expected = short_fr.get(source_fr[index])
            if expected is None:
                continue
            entry = book[en_term].values[str(value)]
            assert entry.short is not None
            assert entry.short.fr == str(expected), f"{en_term}[{value}] FR short"


def test_the_only_new_content_is_the_declared_missing_blocks(sources: Any) -> None:
    """Nothing was invented beyond the thirteen documented missing states."""
    legacy, _ = sources
    book = load_codebook()

    had_empty_value = {
        en_term
        for fr_term, en_term in legacy.FR_EN.items()
        if "" in list(legacy.explain_en.get(en_term, {}).get("values") or [])
    }
    gained = set(book.nullable_variables) - had_empty_value
    assert gained == {"ASPECT", "ACCDN_TYPE", "HOUR"}, DECLARED_DEVIATIONS


def test_class_codes_are_preserved(sources: Any) -> None:
    """Encoding must not shift, or every fitted model changes meaning."""
    assert sources is not None  # ensures the source path is configured
    variables = importlib.import_module("variables")
    book = load_codebook()

    divergent: set[str] = set()
    for name in book:
        source_codes = variables.class_codes.get(name) or {}
        expected = {str(k): v for k, v in source_codes.items() if k != ""}
        if book[name].codes != expected:
            missing_from_book = set(expected) - set(book[name].codes)
            # Codes for documented values must match exactly; only a code for an
            # undocumented value is permitted to be absent.
            for key in set(book[name].codes):
                assert book[name].codes[key] == expected[key], f"{name}[{key}] code shifted"
            assert missing_from_book, name
            divergent.add(name)

    assert divergent == {UNDOCUMENTED_NEGATIVE}, (
        "an encoding exists for a value the documentation does not define"
    )


def test_the_documentation_gap_is_recorded(sources: Any) -> None:
    """Where the published documentation is incomplete, the codebook says so."""
    assert sources is not None
    book = load_codebook()
    notes = book[UNDOCUMENTED_NEGATIVE].notes
    assert notes is not None
    assert notes.en
    assert notes.fr
