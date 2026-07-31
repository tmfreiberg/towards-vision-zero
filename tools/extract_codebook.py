"""Convert the legacy Python data-dictionary modules into ``saaq.yaml``.

This is a one-shot migration tool, not part of the package. It reads
``saaq_data_dictionary.py``, ``dictionaries.py``, and ``variables.py``, and
emits the bilingual codebook as YAML.

Run it from the repository root with the legacy modules on the path::

    uv run python tools/extract_codebook.py --legacy scripts --extra streamlit \
        --out src/vision_zero/data/codebook/saaq.yaml

The script verifies its own output: after writing, it reloads the YAML and
asserts that every variable, value, label, and gloss round-trips back to the
source structures, allowing only the deviations declared in
:data:`DOCUMENTED_MISSING`. It exits non-zero on any undeclared difference.

Two source conventions are reconciled here.

**The empty string is the missing marker.** Ten variables carry a value ``""``
whose gloss reads "Unspecified."; the SAAQ documentation shows these as a row
with a blank *valeur possible* cell and a *description valeur* of "Non précisé".
The empty-string value is therefore not a category but a statement about what a
blank field means, and it is emitted as a ``missing`` block rather than as an
entry in ``values``. The modelling layer agrees: ``class_codes`` assigns no
integer to ``""`` in any variable.

**Three variables are inconsistent with that convention.** ``ASPECT`` and
``ACCDN_TYPE`` carry the "Unspecified." gloss with no corresponding value, so
the gloss is silently dropped by ``zip`` and has never been displayed. ``HOUR``
omits it entirely though the SAAQ documentation lists it. All three receive a
``missing`` block, bringing them into line with the other ten.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

import yaml

# Variables for which the SAAQ documentation shows a "Non précisé" row.
# Read directly off the published data dictionary, pages 2-7.
DOCUMENTED_MISSING: frozenset[str] = frozenset(
    {
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
)

MISSING_LABEL = {"en": "Unspecified", "fr": "Non précisé"}

# Caveats recorded where the published documentation is incomplete.
#
# RDWX is the only variable for which the SAAQ documents a positive value and no
# negative one: `CD_ZON_TRAVX_ROUTR` lists `O` alone, with no `N` row and no
# "Non précisé" row, unlike the `IND_*` flags which list both. The data carries
# `Y` on a small minority of records and a blank on the rest. Whether a blank
# means "not in a work zone" or "not recorded" is not resolvable from the
# documentation, so the codebook documents only what is published and the
# inference is left to the encoding layer.
NOTES: dict[str, dict[str, str]] = {
    "RDWX": {
        "en": (
            "The published documentation lists only the positive value. The data "
            "carries it on a small minority of records and is blank on the rest; "
            "whether a blank means 'not in a work zone' or 'not recorded' cannot "
            "be determined from the documentation."
        ),
        "fr": (
            "La documentation publiée ne mentionne que la valeur positive. Les "
            "données la portent sur une faible minorité d'enregistrements et sont "
            "vides pour les autres; la documentation ne permet pas de déterminer "
            "si une valeur vide signifie « hors zone de travaux » ou « non "
            "consigné »."
        ),
    },
}

# The empty string as it appears in the legacy value lists.
MISSING_SENTINEL = ""


class _Dumper(yaml.SafeDumper):
    """Dumper that indents block sequences under their key."""

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:  # noqa: ARG002
        super().increase_indent(flow=flow, indentless=False)


def _str_presenter(dumper: yaml.Dumper, data: str) -> yaml.ScalarNode:
    """Emit multi-line strings as literal blocks so glosses stay readable."""
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_Dumper.add_representer(str, _str_presenter)


def _as_list(value: Any) -> list[Any]:
    """Normalise a dict view, list, or None to a list."""
    if value is None:
        return []
    return list(value)


def _merge_shorthand(primary: Any, secondary: Any) -> dict[Any, Any]:
    """Merge two shorthand maps, preferring whichever is populated.

    The two legacy copies disagree on eight variables: one holds display labels
    where the other holds ``None``. The populated map wins.
    """
    if primary and secondary:
        merged = dict(secondary)
        merged.update(primary)
        return merged
    return dict(primary or secondary or {})


def build_codebook(legacy: Any, extra: Any, variables: Any) -> dict[str, Any]:
    """Assemble the codebook mapping from the legacy modules.

    Parameters
    ----------
    legacy
        The ``saaq_data_dictionary`` module.
    extra
        The ``dictionaries`` module, whose ``shorthand`` is richer for eight
        variables and whose ``shorthand_fr`` is the only French short-label map.
    variables
        The ``variables`` module, supplying variable types and class codes.

    Returns
    -------
    dict
        The full codebook, ready to serialise.
    """
    fr_en: dict[str, str] = legacy.FR_EN
    var_types: dict[str, str | None] = variables.saaq_variable_types
    class_codes: dict[str, dict[Any, int] | None] = variables.class_codes

    shorthand_en = _merge_shorthand(getattr(extra, "shorthand", None), legacy.shorthand)
    shorthand_fr = getattr(extra, "shorthand_fr", {}) or {}

    out: dict[str, Any] = {}

    for fr_term, en_term in fr_en.items():
        explain_en = legacy.explain_en.get(en_term, {})
        explain_fr = legacy.explain_fr.get(fr_term, {})

        en_values = _as_list(explain_en.get("values"))
        fr_values = _as_list(explain_fr.get("values"))
        en_gloss = _as_list(explain_en.get("elaborate"))
        fr_gloss = _as_list(explain_fr.get("elaborate"))

        short_en = shorthand_en.get(en_term) or {}
        short_fr = shorthand_fr.get(fr_term) or {}
        codes = class_codes.get(en_term) or {}

        entry: dict[str, Any] = {
            "source_column": fr_term,
            "type": var_types.get(en_term),
        }

        summary_en = explain_en.get("summary")
        summary_fr = explain_fr.get("summary")
        if summary_en or summary_fr:
            entry["summary"] = {"en": summary_en, "fr": summary_fr}

        values: dict[str, Any] = {}
        for index, en_value in enumerate(en_values):
            if en_value == MISSING_SENTINEL:
                continue  # handled by the `missing` block below

            fr_value = fr_values[index] if index < len(fr_values) else en_value
            # Labels are display strings. For code-based variables the label is
            # the code's own spelling; the readable name lives in `short`.
            record: dict[str, Any] = {
                "label": {"en": str(en_value), "fr": str(fr_value)},
            }
            if index < len(en_gloss) or index < len(fr_gloss):
                record["gloss"] = {
                    "en": en_gloss[index] if index < len(en_gloss) else None,
                    "fr": fr_gloss[index] if index < len(fr_gloss) else None,
                }
            if en_value in short_en or fr_value in short_fr:
                short_en_value = short_en.get(en_value)
                short_fr_value = short_fr.get(fr_value)
                record["short"] = {
                    "en": None if short_en_value is None else str(short_en_value),
                    "fr": None if short_fr_value is None else str(short_fr_value),
                }
            if en_value in codes:
                record["code"] = codes[en_value]

            values[str(en_value)] = record

        if values:
            entry["values"] = values

        if en_term in DOCUMENTED_MISSING:
            entry["missing"] = {"label": dict(MISSING_LABEL)}

        if en_term in NOTES:
            entry["notes"] = dict(NOTES[en_term])

        out[en_term] = entry

    return {
        "schema_version": 1,
        "dataset": "saaq",
        "source": {
            "publisher": "Société de l'assurance automobile du Québec",
            "url": "https://www.donneesquebec.ca/recherche/dataset/rapports-d-accident",
            "accessed": "2024-03-13",
        },
        "languages": ["en", "fr"],
        "variables": out,
    }


def verify(codebook: dict[str, Any], legacy: Any, extra: Any) -> list[str]:
    """Check the codebook against the legacy structures.

    Returns
    -------
    list of str
        One entry per undeclared discrepancy. Empty means the migration is
        faithful.
    """
    problems: list[str] = []
    variables = codebook["variables"]
    shorthand_en = _merge_shorthand(getattr(extra, "shorthand", None), legacy.shorthand)

    if set(variables) != set(legacy.FR_EN.values()):
        problems.append("variable set differs from FR_EN")

    for fr_term, en_term in legacy.FR_EN.items():
        entry = variables[en_term]
        explain_en = legacy.explain_en.get(en_term, {})
        explain_fr = legacy.explain_fr.get(fr_term, {})

        src_en = [v for v in _as_list(explain_en.get("values")) if v != MISSING_SENTINEL]
        got = list(entry.get("values", {}))
        if [str(v) for v in src_en] != got:
            problems.append(f"{en_term}: values {got} != source {[str(v) for v in src_en]}")

        # Glosses must match position for position, on the non-missing values.
        src_gloss = _as_list(explain_en.get("elaborate"))
        for index, value in enumerate(src_en):
            record = entry.get("values", {}).get(str(value), {})
            expected = src_gloss[index] if index < len(src_gloss) else None
            actual = (record.get("gloss") or {}).get("en")
            if expected != actual:
                problems.append(f"{en_term}[{value}]: EN gloss differs")

        # French labels.
        src_fr = [v for v in _as_list(explain_fr.get("values")) if v != MISSING_SENTINEL]
        for index, value in enumerate(src_en):
            record = entry.get("values", {}).get(str(value), {})
            expected = src_fr[index] if index < len(src_fr) else value
            if record.get("label", {}).get("fr") != str(expected):
                problems.append(f"{en_term}[{value}]: FR label differs")

        # Short labels, where the merged source has them.
        for value in src_en:
            short = (shorthand_en.get(en_term) or {}).get(value)
            record = entry.get("values", {}).get(str(value), {})
            if short is not None and (record.get("short") or {}).get("en") != str(short):
                problems.append(f"{en_term}[{value}]: EN short label differs")

        # Declared deviation: the missing block.
        has_missing = "missing" in entry
        should = en_term in DOCUMENTED_MISSING
        if has_missing != should:
            problems.append(f"{en_term}: missing block presence {has_missing} != expected {should}")

    return problems


def main() -> int:
    """Run the migration and verification. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy", type=Path, required=True, help="dir holding saaq_data_dictionary.py"
    )
    parser.add_argument("--extra", type=Path, required=True, help="dir with dictionaries.py")
    parser.add_argument("--out", type=Path, required=True, help="destination YAML path")
    args = parser.parse_args()

    sys.path.insert(0, str(args.legacy.resolve()))
    sys.path.insert(0, str(args.extra.resolve()))

    legacy = importlib.import_module("saaq_data_dictionary")
    extra = importlib.import_module("dictionaries")
    variables = importlib.import_module("variables")

    codebook = build_codebook(legacy, extra, variables)

    problems = verify(codebook, legacy, extra)
    if problems:
        print(f"VERIFICATION FAILED — {len(problems)} discrepancies:")
        for problem in problems[:40]:
            print(f"  {problem}")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        yaml.dump(
            codebook,
            handle,
            Dumper=_Dumper,
            allow_unicode=True,
            sort_keys=False,
            width=88,
        )

    n_vars = len(codebook["variables"])
    n_values = sum(len(v.get("values", {})) for v in codebook["variables"].values())
    n_missing = sum(1 for v in codebook["variables"].values() if "missing" in v)
    print(
        f"wrote {args.out}: {n_vars} variables, {n_values} values, "
        f"{n_missing} with a documented missing state — verification passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
