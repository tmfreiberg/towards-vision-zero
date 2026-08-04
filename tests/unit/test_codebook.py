"""Tests for the bilingual codebook."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from vision_zero.data import load_codebook
from vision_zero.data.codebook import Codebook, Language

# Variables whose published documentation shows a "Non précisé" row.
DOCUMENTED_MISSING = {
    "ACCDN_TYPE",
    "ASPECT",
    "HOUR",
    "LIGHT",
    "LONG_LOC",
    "NUM_VEH",
    "PUB_PRIV_RD",
    "RD_COND",
    "RD_CONFG",
    "REGION",
    "SPD_LIM",
    "WEATHER",
    "ZONE",
}


@pytest.fixture(scope="module")
def book() -> Codebook:
    return load_codebook()


def test_loads_and_is_non_trivial(book: Codebook) -> None:
    assert len(book) == 25
    assert book.dataset == "saaq"
    assert book.schema_version == 1


def test_every_variable_declares_its_source_column(book: Codebook) -> None:
    for name in book:
        assert book[name].source_column, f"{name} has no source column"


@pytest.mark.parametrize("lang", ["en", "fr"])
def test_every_value_carries_a_label_in_both_languages(book: Codebook, lang: Language) -> None:
    """The bilingual guarantee is enforced, not merely intended."""
    for name in book:
        for key, value in book[name].values.items():
            assert value.label.get(lang), f"{name}[{key}] has no {lang} label"


@pytest.mark.parametrize("lang", ["en", "fr"])
def test_every_missing_state_is_labelled_in_both_languages(book: Codebook, lang: Language) -> None:
    for name in book.nullable_variables:
        missing = book[name].missing
        assert missing is not None
        assert missing.label.get(lang), f"{name} missing state has no {lang} label"


def test_missing_states_match_the_published_documentation(book: Codebook) -> None:
    assert set(book.nullable_variables) == DOCUMENTED_MISSING


def test_variables_that_cannot_be_unspecified(book: Codebook) -> None:
    """Every report carries a date, and severity is what a report exists to record."""
    assert not book["MONTH"].is_nullable
    assert not book["SEVERITY"].is_nullable
    assert not book["YEAR"].is_nullable


def test_missing_state_is_never_an_encodable_value(book: Codebook) -> None:
    """A blank field is the absence of a value, so it receives no integer code."""
    for name in book:
        assert "" not in book[name].values
        for key, value in book[name].values.items():
            if value.code is not None:
                assert key != "", f"{name} encodes the empty string"


def test_codes_are_unique_within_a_variable(book: Codebook) -> None:
    for name in book:
        codes = list(book[name].codes.values())
        assert len(codes) == len(set(codes)), f"{name} has duplicate class codes"


def test_lookup_by_integer_and_by_string_agree(book: Codebook) -> None:
    assert book["WEATHER"].label_for(19) == book["WEATHER"].label_for("19")


def test_none_resolves_to_the_missing_label(book: Codebook) -> None:
    assert book["WEATHER"].label_for(None, lang="fr") == "Non précisé"
    assert book["WEATHER"].label_for(None, lang="en") == "Unspecified"


def test_none_on_a_non_nullable_variable_has_no_label(book: Codebook) -> None:
    assert book["MONTH"].label_for(None) is None


def test_undocumented_values_return_none(book: Codebook) -> None:
    assert book["SEVERITY"].label_for("Catastrophic") is None


def test_short_falls_back_to_the_full_label(book: Codebook) -> None:
    """Code-based variables carry the readable name in `short`, not `label`."""
    assert book["WEATHER"].short_for(19, lang="en") == "Black ice"
    assert book["WEATHER"].label_for(19, lang="en") == "19"
    assert book["SEVERITY"].short_for("Minor") == "Minor"


def test_translation_is_preserved(book: Codebook) -> None:
    assert book.label("SEVERITY", "Minor", lang="fr") == "Léger"
    assert book.label("ASPECT", "Straight", lang="fr") == "Droit"
    assert book["WEATHER"].short_for(19, lang="fr") == "Verglas"


def test_source_columns_are_the_published_french_names(book: Codebook) -> None:
    assert book.source_columns["SEVERITY"] == "GRAVITE"
    assert book.source_columns["ACCDN_TYPE"] == "CD_GENRE_ACCDN"


def test_summary_is_available_in_both_languages(book: Codebook) -> None:
    assert book.summary("SEVERITY", lang="en")
    assert book.summary("SEVERITY", lang="fr")


def test_membership_and_iteration(book: Codebook) -> None:
    assert "SEVERITY" in book
    assert "NOT_A_VARIABLE" not in book
    assert set(book) == set(book.variables)


def test_unknown_variable_raises(book: Codebook) -> None:
    with pytest.raises(KeyError):
        book["NOT_A_VARIABLE"]


def test_loading_is_cached() -> None:
    assert load_codebook() is load_codebook()


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_codebook(tmp_path / "absent.yaml")


def test_malformed_codebook_is_rejected_with_a_precise_path(tmp_path: Path) -> None:
    """A bad codebook must fail at load, naming the offending key."""
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        yaml.dump(
            {
                "schema_version": 1,
                "dataset": "saaq",
                "source": {"publisher": "x", "url": "y", "accessed": "z"},
                "languages": ["en", "fr"],
                "variables": {"X": {"source_column": "Y", "type": "not-a-type"}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError) as excinfo:
        load_codebook(bad)
    assert "type" in str(excinfo.value)


def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "extra.yaml"
    bad.write_text(
        yaml.dump(
            {
                "schema_version": 1,
                "dataset": "saaq",
                "source": {"publisher": "x", "url": "y", "accessed": "z"},
                "languages": ["en", "fr"],
                "variables": {"X": {"source_column": "Y", "surprise": 1}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_codebook(bad)


def test_the_codebook_is_immutable(book: Codebook) -> None:
    with pytest.raises(ValidationError):
        book.variables["SEVERITY"].source_column = "OTHER"  # type: ignore[misc]


def test_the_variable_with_no_documented_negative_records_the_gap(book: Codebook) -> None:
    """RDWX documents only its positive value; the codebook says so in both languages."""
    notes = book["RDWX"].notes
    assert notes is not None
    assert notes.en
    assert notes.fr
    assert set(book["RDWX"].values) == {"Y"}


def test_notes_are_the_exception_not_the_rule(book: Codebook) -> None:
    annotated = {name for name in book if book[name].notes is not None}
    assert annotated == {"RDWX"}


def test_every_variable_has_a_readable_name_in_both_languages(book: Codebook) -> None:
    """The identifier is not a label. `PUB_PRIV_RD` is not English or French."""
    for name in book:
        for lang in ("en", "fr"):
            readable = book[name].name_in(lang)
            assert readable, f"{name} has no {lang} name"
            assert readable != name, f"{name} falls back to its identifier in {lang}"


def test_readable_names_differ_between_the_languages(book: Codebook) -> None:
    """A French name identical to the English one is usually an untranslated stub."""
    identical = [name for name in book if book[name].name_in("en") == book[name].name_in("fr")]
    assert identical == ["Year"] or not identical, f"untranslated: {identical}"


def test_documented_blankness_is_not_the_same_as_observed_blankness(book: Codebook) -> None:
    """`is_nullable` describes the documentation, and the two can disagree.

    RDWX is the case that makes the distinction matter: the SAAQ documents a
    value for "in a work zone" and none for "not in one", so no blank is
    permitted, while the data is empty on almost every record. Anything
    presenting this to a reader has to say which it means.
    """
    assert book["RDWX"].is_nullable is False
    assert book["RDWX"].notes is not None
    assert set(book["RDWX"].values) == {"Y"}
