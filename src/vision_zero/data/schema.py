"""The tabular schema of the SAAQ collision extract.

The schema is *derived* from the codebook rather than restated alongside it.
Dtypes, permitted values, and nullability all follow from what the published
documentation says about each variable, so there is one place to change when the
documentation changes and no possibility of the two drifting apart.

The dtype rule is simple: a variable whose documented values are all integers is
read as a nullable integer; anything else is read as a category. ``SPD_LIM`` is
the interesting case — it is ordinal, but one of its values is ``"<50"``, so it
cannot be an integer column and is ordered explicitly instead.

Beyond the published columns, three targets are derived. They exist because the
questions worth asking are coarser than the recorded severity: whether anyone was
hurt, how badly, and whether more than one vehicle was involved.

Examples
--------
>>> from vision_zero.data.schema import dtypes, TERNARY_SEVERITY
>>> dtypes()["WEATHER"]
'Int64'
>>> dtypes()["SEVERITY"]
'category'
>>> TERNARY_SEVERITY.source
'SEVERITY'
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Final

from vision_zero.data.codebook import load_codebook

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

__all__ = [
    "DERIVED",
    "MULTI_VEHICLE",
    "SEVERITY_FOUR",
    "TERNARY_SEVERITY",
    "UNDOCUMENTED_VALUES",
    "VICTIMS",
    "DerivedColumn",
    "blocks_for",
    "columns",
    "dtypes",
    "feature_candidates",
    "leaks",
    "ordered_categories",
]

#: Columns that identify a record rather than describe one. Never features.
IDENTIFIERS: Final[frozenset[str]] = frozenset({"ID"})

#: Values that occur in the extracts but that the published documentation does
#: not admit. They are coerced to null on load, and the count is logged, so the
#: coercion is a recorded decision rather than something that happens silently
#: inside an encoder.
#:
#: ``LIGHT`` takes the value ``0`` on four records across 2011-2022 — two in
#: 2012, one in 2014, one in 2016 — out of 1,717,407. The documentation defines
#: 1 through 4 and nothing else. Four records confined to no particular year is
#: data entry, not a change of coding, and no meaning can be assigned to the
#: value, so it is treated as unrecorded.
UNDOCUMENTED_VALUES: Final[dict[str, frozenset[str]]] = {
    "LIGHT": frozenset({"0"}),
}


@dataclass(frozen=True, slots=True)
class DerivedColumn:
    """A target computed from a published column.

    Attributes
    ----------
    name
        Name of the derived column.
    source
        The published column it is computed from.
    mapping
        Source value to derived value. Any source value absent from the mapping
        yields a null, which is then dropped: a record whose target cannot be
        determined does not belong in a supervised dataset.
    description
        What the derived column means, for documentation and for the analysis.
    """

    name: str
    source: str
    mapping: dict[Any, int]
    description: str

    def apply(self, values: Any) -> Any:
        """Map a pandas Series of source values to the derived values."""
        return values.map(self.mapping)


#: Severity collapsed to three levels. The two material-damage categories differ
#: only by a dollar threshold on the damage estimate, which is a reporting
#: distinction rather than an outcome one; merging them makes the target
#: "how badly was anyone hurt" rather than "how expensive was the paperwork".
TERNARY_SEVERITY: Final = DerivedColumn(
    name="TNRY_SEV",
    source="SEVERITY",
    mapping={
        "Material damage below the reporting threshold": 0,
        "Material damage only": 0,
        "Minor": 1,
        "Fatal or serious": 2,
    },
    description=(
        "Severity in three levels: 0 material damage only, 1 at least one minor "
        "injury, 2 at least one serious injury or fatality."
    ),
)

#: Severity as published, on its own four-level scale.
#:
#: The analysis collapses this to three levels, because the two material-damage
#: categories differ only by whether the damage estimate cleared a $2,000
#: threshold — a distinction that matters to an insurer and not to anyone asking
#: who got hurt. The explorer keeps all four, because it shows the record as
#: published rather than as modelled.
SEVERITY_FOUR: Final = DerivedColumn(
    name="SEV4",
    source="SEVERITY",
    mapping={
        "Material damage below the reporting threshold": 0,
        "Material damage only": 1,
        "Minor": 2,
        "Fatal or serious": 3,
    },
    description=(
        "Severity on the published four-level scale: material damage below the "
        "reporting threshold, material damage above it, at least one minor "
        "injury, at least one serious injury or fatality."
    ),
)

#: Whether anyone was hurt at all.
VICTIMS: Final = DerivedColumn(
    name="VICTIMS",
    source="NUM_VICTIMS",
    mapping={0: 0, 1: 1, 2: 1, 9: 1},
    description="Whether the collision produced at least one victim.",
)

#: Whether more than one vehicle was involved.
MULTI_VEHICLE: Final = DerivedColumn(
    name="MULT_VEH",
    source="NUM_VEH",
    mapping={1: 0, 2: 1, 9: 1},
    description="Whether more than one vehicle was involved.",
)

DERIVED: Final[tuple[DerivedColumn, ...]] = (
    TERNARY_SEVERITY,
    SEVERITY_FOUR,
    VICTIMS,
    MULTI_VEHICLE,
)

#: Variables that determine, or are determined by, a target — and so cannot be
#: used to predict it. ``NUM_VICTIMS`` is the clearest case: it is positive
#: exactly when severity exceeds material damage, so a model given both would
#: score well and have learned nothing.
#:
#: ``PED`` belongs in the same category, which is easy to miss. The published
#: definitions of the road-user indicators are not parallel:
#:
#: - ``LT_TRK``, ``HVY_VEH``, ``MTRCYC``, ``BICYC`` each record that a vehicle
#:   of some kind was *involved* in the collision.
#: - ``PED`` records that there was at least one pedestrian *victim*, injured
#:   or killed.
#:
#: So ``PED = Y`` entails at least one victim, and therefore a severity above
#: material damage, by definition rather than by any empirical relationship. It
#: is a partial disclosure of the target — the pedestrian subset of
#: ``NUM_VICTIMS`` — and a model given it is rewarded for restating the
#: question. It is also useless for the purpose the data would actually serve:
#: nobody planning an intersection knows in advance whether a pedestrian will
#: be hurt there.
#:
#: ``ACCDN_TYPE`` is not affected. Its ``pedestrian`` value means a collision
#: *with* a pedestrian, which is a description of the event rather than of its
#: outcome, and it remains available.
_SEVERITY_FAMILY = frozenset({"SEVERITY", "SEV4", "TNRY_SEV", "VICTIMS", "NUM_VICTIMS", "PED"})


def _leaks_severity(target: str) -> frozenset[str]:
    """Everything in the severity family except the target itself."""
    return _SEVERITY_FAMILY - {target}


LEAKAGE: Final[dict[str, frozenset[str]]] = {
    "SEVERITY": _leaks_severity("SEVERITY"),
    "SEV4": _leaks_severity("SEV4"),
    "TNRY_SEV": _leaks_severity("TNRY_SEV"),
    "VICTIMS": _leaks_severity("VICTIMS"),
    "NUM_VICTIMS": _leaks_severity("NUM_VICTIMS"),
    "ACCDN_TYPE": frozenset({"PED", "BICYC", "MTRCYC", "NUM_VEH", "HVY_VEH", "LT_TRK"}),
    "PED": _leaks_severity("PED") | {"ACCDN_TYPE"},
    "BICYC": frozenset({"ACCDN_TYPE"}),
    "MTRCYC": frozenset({"ACCDN_TYPE"}),
    "NUM_VEH": frozenset({"MULT_VEH"}),
    "MULT_VEH": frozenset({"NUM_VEH"}),
}

#: Features grouped by what kind of thing they describe, and — the reason the
#: grouping exists — by whether anyone could act on them.
#:
#: The question a road-safety analysis is asked to answer is not "can severity be
#: predicted" but "does anything we control affect it". Those are different, and
#: separating the blocks is what lets them be answered separately. Knowing a
#: pedestrian was struck predicts injury well and suggests nothing to do about
#: it; knowing the road configuration predicts poorly and would suggest a great
#: deal if it predicted at all.
FEATURE_BLOCKS: Final[dict[str, frozenset[str]]] = {
    # Who and what took part. Predictive, and outside anyone's control at the
    # moment of the collision.
    "participants": frozenset({"ACCDN_TYPE", "BICYC", "MTRCYC", "HVY_VEH", "LT_TRK", "NUM_VEH"}),
    # The road and the conditions on it. This is the block an engineer could
    # change, and therefore the one whose predictive value is the finding.
    "environment": frozenset(
        {
            "WEATHER",
            "LIGHT",
            "RD_COND",
            "SPD_LIM",
            "ASPECT",
            "RD_CONFG",
            "LONG_LOC",
            "ZONE",
            "PUB_PRIV_RD",
            "RDWX",
        }
    ),
    # When it happened. Partly actionable through enforcement and scheduling.
    "temporal": frozenset({"MONTH", "HOUR", "WKDY_WKND"}),
    # Administrative geography. Not a property of the site and not actionable,
    # but it stands for everything unmeasured that varies between regions:
    # urban density, enforcement, road standards, reporting practice. It is
    # constant, and so dropped, in any experiment restricted to one region.
    "geography": frozenset({"REGION"}),
}


def blocks_for(features: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """Split a feature list into blocks, keeping only what is present.

    Parameters
    ----------
    features
        Feature columns actually available.

    Returns
    -------
    dict
        Block name to its members, in the order given. Empty blocks are omitted.

    Examples
    --------
    >>> sorted(blocks_for(["WEATHER", "ACCDN_TYPE", "MONTH"]))
    ['environment', 'participants', 'temporal']
    """
    out = {}
    for name, members in FEATURE_BLOCKS.items():
        present = tuple(feature for feature in features if feature in members)
        if present:
            out[name] = present
    return out


#: Variables recording an outcome rather than a circumstance. Any of these used
#: to predict severity is answering the question with the answer.
OUTCOME_VARIABLES: Final[frozenset[str]] = _SEVERITY_FAMILY


def _all_integer_keys(keys: list[str]) -> bool:
    """Whether every documented value of a variable spells an integer."""
    if not keys:
        return False
    for key in keys:
        try:
            int(key)
        except ValueError:
            return False
    return True


@lru_cache(maxsize=4)
def dtypes(path: Path | None = None) -> dict[str, str]:
    """Return the pandas dtype for each published column.

    Integer-coded variables become ``Int64`` — pandas' nullable integer — so that
    an unspecified value stays null instead of forcing the column to float.
    Everything else becomes ``category``.

    Parameters
    ----------
    path
        Codebook to derive from. Defaults to the packaged SAAQ codebook.

    Returns
    -------
    dict
        Column name to dtype string, in codebook order.

    Examples
    --------
    >>> dtypes()["MONTH"], dtypes()["SPD_LIM"], dtypes()["ID"]
    ('Int64', 'category', 'string')
    """
    book = load_codebook(path)
    out: dict[str, str] = {}
    for name in book:
        if name in IDENTIFIERS:
            out[name] = "string"
        # YEAR enumerates no values in the codebook but is plainly an integer.
        elif name == "YEAR" or _all_integer_keys(list(book[name].values)):
            out[name] = "Int64"
        else:
            out[name] = "category"
    return out


@lru_cache(maxsize=4)
def columns(path: Path | None = None) -> tuple[str, ...]:
    """Return the published column names, in codebook order."""
    return tuple(load_codebook(path))


@lru_cache(maxsize=4)
def ordered_categories(path: Path | None = None) -> dict[str, tuple[str, ...]]:
    """Return the value order for ordinal variables read as categories.

    Ordinal variables that cannot be integers still have an order, and losing it
    turns a meaningful axis into an alphabetical one. ``SPD_LIM`` runs from
    ``"<50"`` upward; ``SEVERITY`` runs from material damage to fatal.
    """
    book = load_codebook(path)
    out: dict[str, tuple[str, ...]] = {}
    for name in book:
        variable = book[name]
        if variable.type != "ordinal":
            continue
        if _all_integer_keys(list(variable.values)):
            continue
        ordered = sorted(variable.values.items(), key=lambda item: item[1].code or 0)
        out[name] = tuple(key for key, _ in ordered)
    return out


def leaks(target: str) -> frozenset[str]:
    """Return the columns that cannot be used to predict ``target``.

    Parameters
    ----------
    target
        Name of the target column, published or derived.

    Returns
    -------
    frozenset of str
        Columns excluded from the feature set for this target. Empty if the
        target has no known leakage.

    Examples
    --------
    >>> "NUM_VICTIMS" in leaks("TNRY_SEV")
    True
    >>> leaks("WEATHER")
    frozenset()
    """
    return LEAKAGE.get(target, frozenset())


def feature_candidates(
    target: str,
    *,
    available: tuple[str, ...] | None = None,
    exclude: frozenset[str] = frozenset(),
    path: Path | None = None,
) -> tuple[str, ...]:
    """Return the columns usable as features for ``target``.

    Removes the target itself, anything that leaks it, identifiers, and any
    caller-supplied exclusions.

    Parameters
    ----------
    target
        The column to be predicted.
    available
        Columns actually present. Defaults to the published columns plus the
        derived ones.
    exclude
        Further columns to drop, for instance ``{"YEAR"}`` when the year of a
        collision is not plausibly explanatory.
    path
        Codebook to derive from.

    Returns
    -------
    tuple of str
        Candidate feature columns, in order.

    Examples
    --------
    >>> features = feature_candidates("TNRY_SEV", exclude=frozenset({"YEAR"}))
    >>> "NUM_VICTIMS" in features or "YEAR" in features or "SEVERITY" in features
    False
    >>> "WEATHER" in features
    True
    """
    if available is None:
        available = columns(path) + tuple(d.name for d in DERIVED)

    blocked = set(exclude) | set(IDENTIFIERS) | set(leaks(target)) | {target}

    # A derived column is a deterministic function of its source. Offering both
    # to a model contributes no information and introduces exact collinearity,
    # which for a linear estimator is not merely useless but ill-posed.
    for derived in DERIVED:
        if derived.source in available and derived.source not in blocked:
            blocked.add(derived.name)

    return tuple(name for name in available if name not in blocked)
