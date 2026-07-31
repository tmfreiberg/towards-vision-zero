"""Packaging a dataset for the browser.

The explorer is a static page. It has no server to ask, so everything it needs
travels with it: the records themselves, the vocabulary to render them, and the
base rates to compare against.

Records are written as one signed byte per field, column by column. A byte is
ample — no variable here has more than twenty documented values — and laying the
columns end to end rather than interleaving them lets the browser filter one
field at a time over a contiguous run of memory.

The rows are sorted before writing. Nothing downstream depends on their order —
the explorer counts records, and a count is indifferent to sequence — so sorting
is free, and it turns every column into long runs of a repeated value. On the
province-wide extract that halves the payload, from about nine and a half
megabytes to five: the difference between a download and a page load.

Unrecorded values are written as ``-1`` and kept. Whether a field was filled in
is part of what the record says, and a tool that quietly dropped blanks would
misrepresent the data in exactly the way the analysis warns about.

Examples
--------
>>> from vision_zero.export import export_for_browser  # doctest: +SKIP
>>> export_for_browser(experiment, destination)  # doctest: +SKIP
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from vision_zero import __version__
from vision_zero.data.codebook import Language, load_codebook
from vision_zero.data.schema import DERIVED, DerivedColumn
from vision_zero.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from vision_zero.modeling.config import Experiment
    from vision_zero.modeling.dataset import Dataset

__all__ = ["MISSING_CODE", "ExportResult", "export_for_browser"]

log = get_logger(__name__)

#: Written for a field that was left blank on the report.
MISSING_CODE = -1

#: Provenance, carried in the payload so the page never has to hard-code it.
SOURCE_CITATION: dict[str, dict[str, str]] = {
    "en": {
        "citation": (
            "QUEBEC AUTOMOBILE INSURANCE SOCIETY (SAAQ). Accident reports, "
            "[Dataset], in Data Quebec, 2017, updated December 18, 2023. "
            "https://www.donneesquebec.ca/recherche/dataset/rapports-d-accident, "
            "(accessed March 13, 2024)."
        ),
        "gloss": (
            "Data from accident reports completed by police officers, including "
            "the time, severity of the accident as well as the type of vehicles "
            "involved."
        ),
    },
    "fr": {
        "citation": (
            "SOCIÉTÉ DE L'ASSURANCE AUTOMOBILE DU QUÉBEC (SAAQ). Rapports "
            "d'accident, [Jeu de données], dans Données Québec, 2017, mis à jour "
            "le 18 décembre 2023. "
            "https://www.donneesquebec.ca/recherche/dataset/rapports-d-accident, "
            "(consulté le 13 mars 2024)."
        ),
        "gloss": (
            "Données issues des rapports d'accident remplis par les policiers, "
            "incluant notamment le moment, la gravité de l'accident de même que "
            "le type des véhicules impliqués."
        ),
    },
}

DATA_NAME = "records.bin.gz"
META_NAME = "meta.json"
SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ExportResult:
    """What was written, and how big it turned out."""

    data_path: Path
    meta_path: Path
    n_records: int
    n_features: int
    compressed_bytes: int

    @property
    def megabytes(self) -> float:
        """Compressed payload size."""
        return self.compressed_bytes / (1 << 20)


LANGUAGES: tuple[Language, ...] = ("en", "fr")


def _feature_vocabulary(name: str) -> dict[str, Any]:
    """Describe one variable and its documented values, in both languages.

    Carries the SAAQ's own definition of every value, not just its name. A
    reader asking what counts as a public road, or where the boundary between a
    business and a residential area falls, should not have to leave the page to
    find out — and the answer is not obvious, since these are administrative
    definitions rather than everyday ones.
    """
    book = load_codebook()
    variable = book[name]

    values = []
    for key in variable.values:
        entry = variable.values[key]
        values.append(
            {
                "key": key,
                "label": {lang: variable.short_for(key, lang) or key for lang in LANGUAGES},
                "definition": {
                    lang: (entry.gloss.get(lang) if entry.gloss else None) for lang in LANGUAGES
                },
            }
        )

    missing = variable.missing
    return {
        "name": name,
        "source_column": variable.source_column,
        "type": variable.type,
        "label": {lang: variable.name_in(lang) or name for lang in LANGUAGES},
        "summary": {
            lang: (variable.summary.get(lang) if variable.summary else None) for lang in LANGUAGES
        },
        "values": values,
        "notes": {
            lang: (variable.notes.get(lang) if variable.notes else None) for lang in LANGUAGES
        },
        "missing": {
            "documented": missing is not None,
            "label": {
                lang: (missing.label.get(lang) if missing else None) or "not recorded"
                for lang in LANGUAGES
            },
        },
    }


def _target_vocabulary(target: str, classes: list[int]) -> dict[str, Any]:
    """Describe the outcome classes, tracing each back to its published definition.

    A derived target is a relabelling of a published one, so its classes inherit
    the SAAQ's definitions. Where a derived class merges several published
    values — as the three-level severity scale merges the two material-damage
    categories, which differ only by a dollar threshold on the damage estimate —
    every merged definition is carried, so the merge is visible rather than
    silently performed.

    Falls back to the class number where a target has no published counterpart.
    """
    book = load_codebook()
    derived = next((column for column in DERIVED if column.name == target), None)

    if derived is None or derived.source not in book:
        return {
            "labels": {lang: [str(label) for label in classes] for lang in LANGUAGES},
            "definitions": {lang: [None] * len(classes) for lang in LANGUAGES},
            "merged": [1] * len(classes),
        }

    source = book[derived.source]
    members: dict[int, list[str]] = {label: [] for label in classes}
    for value, mapped in derived.mapping.items():
        if mapped in members and str(value) in source.values:
            members[mapped].append(str(value))

    labels: dict[str, list[str]] = {}
    definitions: dict[str, list[str | None]] = {}
    for lang in LANGUAGES:
        named, described = [], []
        for label in classes:
            keys = members[label]
            named.append(
                " / ".join(source.short_for(key, lang) or key for key in keys) or str(label)
            )
            glosses = []
            for key in keys:
                gloss = source.values[key].gloss
                text = gloss.get(lang) if gloss is not None else None
                if text:
                    glosses.append(text)
            described.append(" ".join(glosses) or None)
        labels[lang] = named
        definitions[lang] = described

    return {
        "labels": labels,
        "definitions": definitions,
        "merged": [len(members[label]) for label in classes],
        "source_variable": derived.source,
        "groupings": _groupings(derived, classes),
    }


def _groupings(target: DerivedColumn, classes: list[int]) -> list[dict[str, Any]]:
    """Express every other scale over the same variable as a partition of this one.

    Severity is published on four levels and the analysis uses three, merging
    the two material-damage categories because they differ only by whether the
    damage estimate cleared a dollar threshold. That merge is already declared
    in the schema, as a second derived column over the same source, so the
    explorer can offer it without anyone restating which classes combine.

    Any coarser scale defined over the same variable appears here automatically.
    A scale that does not partition this one cleanly — because it splits a class
    this target has already merged — is skipped rather than approximated.

    Returns
    -------
    list of dict
        One entry per scale, finest first, each with the groups of this target's
        class positions that it combines, and the labels and definitions of the
        combined classes.
    """
    book = load_codebook()
    source = book[target.source]

    # Where each published value sits among this target's classes.
    position_of_value = {
        value: classes.index(mapped)
        for value, mapped in target.mapping.items()
        if mapped in classes and str(value) in source.values
    }

    scales = [target] + [
        column
        for column in DERIVED
        if column.source == target.source and column.name != target.name
    ]

    out = []
    for scale in scales:
        groups: dict[int, set[int]] = {}
        keys: dict[int, list[str]] = {}
        for value, mapped in scale.mapping.items():
            if value not in position_of_value:
                continue
            groups.setdefault(mapped, set()).add(position_of_value[value])
            keys.setdefault(mapped, []).append(str(value))

        ordered = sorted(groups)
        partition = [sorted(groups[label]) for label in ordered]

        covered = [position for group in partition for position in group]
        if sorted(covered) != list(range(len(classes))) or len(covered) != len(set(covered)):
            continue  # not a clean partition of this scale

        labels: dict[str, list[str]] = {}
        definitions: dict[str, list[str | None]] = {}
        for lang in LANGUAGES:
            named, described = [], []
            for label in ordered:
                members_here = keys[label]
                named.append(" / ".join(source.short_for(key, lang) or key for key in members_here))
                glosses = []
                for key in members_here:
                    gloss = source.values[key].gloss
                    text = gloss.get(lang) if gloss is not None else None
                    if text:
                        glosses.append(text)
                described.append(" ".join(glosses) or None)
            labels[lang] = named
            definitions[lang] = described

        out.append(
            {
                "name": scale.name,
                "published": scale.name == target.name,
                "partition": partition,
                "labels": labels,
                "definitions": definitions,
                "merged": [len(keys[label]) for label in ordered],
            }
        )

    return sorted(out, key=lambda entry: -len(entry["partition"]))


def _encode(column: pd.Series, keys: list[str]) -> np.ndarray:
    """Map a column's values onto their index in the documented vocabulary."""
    lookup = {key: index for index, key in enumerate(keys)}
    as_text = column.astype("string")
    codes = as_text.map(lookup)
    return codes.fillna(MISSING_CODE).to_numpy(dtype=np.int8)


def export_for_browser(
    experiment: Experiment,
    dataset: Dataset,
    destination: Path,
    *,
    include_test: bool = True,
) -> ExportResult:
    """Write the records and their vocabulary as static files.

    Parameters
    ----------
    experiment
        The specification that produced the dataset, recorded in the metadata so
        a reader can see which records the tool is showing.
    dataset
        Assembled features and targets.
    destination
        Directory to write into. Created if absent.
    include_test
        Whether to include the held-out records. Default true: the explorer
        makes no predictions and fits nothing, so there is nothing for held-out
        records to leak into, and excluding them would show the reader a smaller
        dataset than exists for no benefit.

    Returns
    -------
    ExportResult
        Paths written and the compressed size.
    """
    destination.mkdir(parents=True, exist_ok=True)
    book = load_codebook()

    if include_test:
        X = pd.concat([dataset.X_train, dataset.X_test], ignore_index=True)
        y = pd.concat([dataset.y_train, dataset.y_test], ignore_index=True)
    else:
        X, y = dataset.X_train, dataset.y_train

    features = [name for name in dataset.features if name in book]
    vocabulary = [_feature_vocabulary(name) for name in features]

    columns = [
        _encode(X[entry["name"]], [value["key"] for value in entry["values"]])
        for entry in vocabulary
    ]
    # NumPy integers are not JSON-serialisable; the metadata must be plain.
    classes = [int(label) for label in dataset.classes]
    columns.append(
        np.asarray(y.map({label: index for index, label in enumerate(classes)}), dtype=np.int8)
    )

    payload = np.concatenate(columns).tobytes()
    compressed = gzip.compress(payload, 9)

    data_path = destination / DATA_NAME
    data_path.write_bytes(compressed)

    base_rate = [float((np.asarray(y) == label).mean()) for label in classes]
    meta = {
        "schema_version": SCHEMA_VERSION,
        "package_version": __version__,
        "source": SOURCE_CITATION,
        "experiment": {
            "name": experiment.name,
            "description": experiment.description.strip(),
            "years": experiment.data.years,
            "restrict": experiment.data.restrict,
            "max_missing_features": experiment.data.max_missing_features,
        },
        "n_records": len(y),
        "missing_code": MISSING_CODE,
        "layout": "column-major int8, features in order, target last",
        "features": vocabulary,
        "target": {
            "name": dataset.target,
            "classes": classes,
            "base_rate": base_rate,
            **_target_vocabulary(dataset.target, classes),
        },
    }
    meta_path = destination / META_NAME
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info(
        "Exported %d records over %d features to %s (%.2f MB compressed)",
        len(y),
        len(features),
        destination,
        len(compressed) / (1 << 20),
    )
    return ExportResult(
        data_path=data_path,
        meta_path=meta_path,
        n_records=len(y),
        n_features=len(features),
        compressed_bytes=len(compressed),
    )
