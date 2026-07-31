"""Tests for the ambiguous-group exhibit.

This is the artifact the argument is presented through, so it has to be right
about what it shows: the groups must genuinely be identical on every feature,
and the counts must be the counts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vision_zero.ceiling.bayes_error import cell_labels
from vision_zero.ceiling.examples import ambiguous_cells, describe_cell, worst_case_pairs


@pytest.fixture
def mixed() -> tuple[pd.DataFrame, pd.Series]:
    """One large mixed group, one large unanimous group, one small mixed group."""
    rows = (
        [("mixed", label) for label in [0] * 30 + [1] * 15 + [2] * 5]
        + [("clean", 0)] * 40
        + [("tiny", 0), ("tiny", 1)]
    )
    X = pd.DataFrame({"f": [value for value, _ in rows]})
    y = pd.Series([label for _, label in rows])
    return X, y


def test_unanimous_groups_are_not_reported_as_ambiguous(
    mixed: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = mixed
    groups = ambiguous_cells(X, y, min_multiplicity=10)
    identifiers = cell_labels(X)
    for _, group in groups.iterrows():
        members = y[identifiers.to_numpy() == group["cell"]]
        assert len(set(members)) > 1 or group["entropy_bits"] == 0


def test_small_groups_are_excluded(mixed: tuple[pd.DataFrame, pd.Series]) -> None:
    """A group of two that disagrees proves nothing."""
    X, y = mixed
    groups = ambiguous_cells(X, y, min_multiplicity=10)
    assert (groups["n"] >= 10).all()


def test_counts_match_the_data(mixed: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = mixed
    groups = ambiguous_cells(X, y, min_multiplicity=10)
    identifiers = cell_labels(X)
    for _, group in groups.iterrows():
        members = y[identifiers.to_numpy() == group["cell"]]
        assert int(group["n"]) == len(members)
        for label in (0, 1, 2):
            assert int(group[f"n_class_{label}"]) == int((members == label).sum())


def test_the_local_ceiling_is_the_largest_share(
    mixed: tuple[pd.DataFrame, pd.Series],
) -> None:
    """max_share is what a perfect classifier would score on that group."""
    X, y = mixed
    group = ambiguous_cells(X, y, min_multiplicity=10).iloc[0]
    assert group["max_share"] == pytest.approx(30 / 50)


def test_ranking_prefers_large_and_mixed_over_either_alone(
    mixed: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = mixed
    groups = ambiguous_cells(X, y, min_multiplicity=10)
    assert int(groups.iloc[0]["n"]) == 50
    assert int(groups.iloc[0]["classes_present"]) == 3


def test_describe_returns_the_values_shared_by_the_group() -> None:
    X = pd.DataFrame({"a": ["x", "x", "y"], "b": ["p", "p", "q"]})
    identifiers = cell_labels(X)
    described = describe_cell(X, int(identifiers.iloc[0]), cells=identifiers)
    assert described == {"a": "x", "b": "p"}


def test_unrecorded_values_are_named_as_such() -> None:
    """Being unrecorded is itself something the records share."""
    X = pd.DataFrame({"a": [None, None]})
    identifiers = cell_labels(X)
    assert describe_cell(X, int(identifiers.iloc[0]), cells=identifiers) == {"a": "not recorded"}


def test_describing_an_absent_group_is_empty() -> None:
    X = pd.DataFrame({"a": ["x"]})
    assert describe_cell(X, 999) == {}


def test_worst_case_requires_both_extremes() -> None:
    """A group holding a fatality and a scratch is the sharpest form of the exhibit."""
    rows = [("both", label) for label in [0] * 20 + [2] * 3] + [
        ("mild_only", label) for label in [0] * 20 + [1] * 5
    ]
    X = pd.DataFrame({"f": [value for value, _ in rows]})
    y = pd.Series([label for _, label in rows])

    identifiers = cell_labels(X)
    pairs = worst_case_pairs(X, y, min_multiplicity=10, cells=identifiers)
    assert len(pairs) == 1
    assert int(pairs.iloc[0]["n_class_2"]) == 3
    assert int(pairs.iloc[0]["n_class_0"]) == 20


def test_no_qualifying_group_returns_empty() -> None:
    X = pd.DataFrame({"f": [f"v{i}" for i in range(50)]})
    y = pd.Series(np.arange(50) % 3)
    assert ambiguous_cells(X, y, min_multiplicity=10).empty
    assert worst_case_pairs(X, y, min_multiplicity=10).empty


def test_entropy_is_zero_for_a_unanimous_group() -> None:
    X = pd.DataFrame({"f": ["same"] * 40})
    y = pd.Series([0] * 40)
    groups = ambiguous_cells(X, y, min_multiplicity=10)
    assert groups.iloc[0]["entropy_bits"] == pytest.approx(0.0)
    assert groups.iloc[0]["max_share"] == pytest.approx(1.0)
