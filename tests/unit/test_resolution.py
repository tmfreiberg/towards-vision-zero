"""Tests for the resolution-versus-coverage analysis.

These estimators are easy to get wrong in a way that looks fine. An earlier
version used the plug-in mutual information and confidently ranked eight pure
noise variables above the two that carried the signal, because more variables
make smaller groups and a group of one shows no uncertainty. So the tests here
are built around data with known structure, and they check the ranking rather
than the arithmetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vision_zero.ceiling.resolution import (
    drop_one_profile,
    resolution_profile,
    single_feature_profile,
    slice_dispersion,
)

NOISE_FEATURES = 8


@pytest.fixture(scope="module")
def planted() -> tuple[pd.DataFrame, pd.Series]:
    """Two variables that matter, eight that do not."""
    rng = np.random.default_rng(0)
    n = 40_000
    signal_a = rng.integers(0, 3, n)
    signal_b = rng.integers(0, 2, n)
    noise = {f"noise{i}": rng.integers(0, 4, n) for i in range(NOISE_FEATURES)}
    risk = 0.05 + 0.25 * (signal_a == 2) + 0.15 * (signal_b == 1)
    y = pd.Series((rng.random(n) < risk).astype(int))
    X = pd.DataFrame({"signal_a": signal_a, "signal_b": signal_b, **noise})
    return X, y


def test_matching_on_nothing_carries_no_information(
    planted: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = planted
    assert slice_dispersion(X, y, ())["information_bits"] == pytest.approx(0.0, abs=0.02)


def test_matching_on_nothing_gives_one_group(
    planted: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = planted
    assert int(slice_dispersion(X, y, ())["n_cells"]) == 1


def test_the_honest_estimate_separates_signal_from_noise(
    planted: tuple[pd.DataFrame, pd.Series],
) -> None:
    """The test an earlier implementation failed outright."""
    X, y = planted
    ranked = single_feature_profile(X, y)
    top_two = set(ranked.head(2)["feature"])
    assert top_two == {"signal_a", "signal_b"}


def test_the_stronger_signal_ranks_first(
    planted: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = planted
    ranked = single_feature_profile(X, y).set_index("feature")["information_bits"]
    assert float(ranked["signal_a"]) > float(ranked["signal_b"])


def test_noise_variables_are_mutually_indistinguishable(
    planted: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = planted
    ranked = single_feature_profile(X, y).set_index("feature")["information_bits"]
    noise = ranked[[f"noise{i}" for i in range(NOISE_FEATURES)]]
    assert float(noise.max() - noise.min()) < 0.005


def test_signal_clears_the_noise_floor_by_an_order_of_magnitude(
    planted: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = planted
    ranked = single_feature_profile(X, y).set_index("feature")["information_bits"]
    noise_floor = float(ranked[[f"noise{i}" for i in range(NOISE_FEATURES)]].max())
    assert float(ranked["signal_a"]) > 5 * noise_floor


def test_the_plug_in_estimate_rises_with_meaningless_variables(
    planted: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Documenting the failure mode that motivated the half-split."""
    X, y = planted
    few = slice_dispersion(X, y, ("signal_a", "signal_b"))
    many = slice_dispersion(X, y, list(X.columns))
    assert many["information_bits_plugin"] > few["information_bits_plugin"]
    assert many["information_bits"] < few["information_bits"]


def test_groups_shrink_as_more_variables_are_matched(
    planted: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = planted
    profile = resolution_profile(X, y, sizes=(0, 2, 6, 10), n_samples=3, seed=1)
    assert profile["mean_cell_size"].is_monotonic_decreasing
    assert profile["n_cells"].is_monotonic_increasing


def test_singletons_appear_only_once_the_match_is_strict(
    planted: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = planted
    profile = resolution_profile(X, y, sizes=(2, 10), n_samples=3, seed=1)
    singletons = profile.set_index("n_features")["singleton_share"].astype(float)
    assert singletons[2] == pytest.approx(0.0, abs=1e-9)
    assert singletons[10] > 0.5


def test_information_peaks_before_the_full_variable_set(
    planted: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Past a point, extra variables destroy information rather than adding it."""
    X, y = planted
    profile = resolution_profile(X, y, sizes=(0, 1, 2, 4, 10), n_samples=4, seed=1)
    best = int(np.argmax(profile["information_bits"].to_numpy()))
    peak = int(profile["n_features"].to_numpy()[best])
    assert 0 < peak < 10


def test_the_profile_covers_every_requested_size(
    planted: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = planted
    profile = resolution_profile(X, y, sizes=(0, 3, 10), n_samples=2, seed=1)
    assert list(profile["n_features"]) == [0, 3, 10]


def test_endpoints_are_evaluated_once_only(
    planted: tuple[pd.DataFrame, pd.Series],
) -> None:
    """There is one subset of size zero and one of full size; sampling them is waste."""
    X, y = planted
    profile = resolution_profile(X, y, sizes=(0, 3, 10), n_samples=5, seed=1)
    subsets = profile.set_index("n_features")["n_subsets"].astype(int)
    assert subsets[0] == 1
    assert subsets[10] == 1
    assert subsets[3] == 5


def test_drop_one_warns_when_the_full_set_overfits(
    planted: tuple[pd.DataFrame, pd.Series],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Differencing negative numbers gives an ordering that means nothing."""
    X, y = planted
    with caplog.at_level("WARNING", logger="vision_zero"):
        drop_one_profile(X, y)
    assert any("overfits" in record.message for record in caplog.records)


def test_drop_one_is_reliable_on_a_small_variable_set(
    planted: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = planted
    profile = drop_one_profile(X, y, features=["signal_a", "signal_b", "noise0"])
    assert profile.iloc[0]["dropped"] == "signal_a"
