"""Tests for minority-class upsampling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from vision_zero.modeling.resampling import class_counts, upsample


def _make(counts: dict[int, int]) -> tuple[pd.DataFrame, pd.Series]:
    labels = np.concatenate([np.full(n, label) for label, n in counts.items()])
    return pd.DataFrame({"a": np.arange(len(labels))}), pd.Series(labels)


def test_no_resampling_returns_the_input_unchanged() -> None:
    X, y = _make({0: 10, 1: 3})
    X_out, y_out = upsample(X, y, max_repeats=1)
    pd.testing.assert_frame_equal(X_out, X)
    pd.testing.assert_series_equal(y_out, y)


def test_minority_classes_grow_toward_the_largest() -> None:
    X, y = _make({0: 800, 1: 190, 2: 10})
    _, y_out = upsample(X, y, max_repeats=25, seed=0)
    counts = class_counts(y_out)
    assert counts[0] == 800
    assert counts[1] == 800
    assert counts[2] == 250


def test_the_repeat_bound_is_respected() -> None:
    """Without a bound a one-percent class would be repeated a hundredfold."""
    X, y = _make({0: 1000, 1: 10})
    _, y_out = upsample(X, y, max_repeats=5, seed=0)
    assert class_counts(y_out)[1] == 50


def test_no_class_exceeds_the_largest() -> None:
    X, y = _make({0: 100, 1: 90})
    _, y_out = upsample(X, y, max_repeats=100, seed=0)
    assert max(class_counts(y_out).values()) == 100


def test_features_and_labels_stay_aligned() -> None:
    """A misalignment here would silently scramble every label."""
    labels = np.array([0] * 20 + [1] * 5)
    X = pd.DataFrame({"label_copy": labels})
    y = pd.Series(labels)
    X_out, y_out = upsample(X, y, max_repeats=4, seed=0)
    assert (X_out["label_copy"].to_numpy() == y_out.to_numpy()).all()


def test_only_existing_records_are_used() -> None:
    """Upsampling repeats records; it does not invent them."""
    X, y = _make({0: 50, 1: 5})
    X_out, _ = upsample(X, y, max_repeats=10, seed=0)
    assert set(X_out["a"]) <= set(X["a"])


def test_resampling_is_reproducible() -> None:
    X, y = _make({0: 100, 1: 7})
    first = upsample(X, y, max_repeats=8, seed=11)[1]
    second = upsample(X, y, max_repeats=8, seed=11)[1]
    pd.testing.assert_series_equal(first, second)


def test_mismatched_lengths_are_rejected() -> None:
    X, y = _make({0: 10, 1: 3})
    with pytest.raises(ValueError, match="rows"):
        upsample(X, y.iloc[:5], max_repeats=2)


@given(
    counts=st.lists(st.integers(min_value=1, max_value=100), min_size=2, max_size=4),
    max_repeats=st.integers(min_value=1, max_value=20),
)
@settings(max_examples=40, deadline=None)
def test_no_class_ever_shrinks(counts: list[int], max_repeats: int) -> None:
    X, y = _make(dict(enumerate(counts)))
    _, y_out = upsample(X, y, max_repeats=max_repeats, seed=0)
    before, after = class_counts(y), class_counts(y_out)
    for label, count in before.items():
        assert after[label] >= count
        assert after[label] <= count * max_repeats
