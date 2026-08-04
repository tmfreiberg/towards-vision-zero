"""Tests for the Hamming locality analysis.

The measure has to distinguish features that carry local structure from features
that carry none, so the tests are built around two constructed datasets where
the answer is known by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vision_zero.ceiling.locality import (
    confined_difference_agreement,
    difference_set_profile,
    locality_profile,
    pair_agreement,
)

FEATURES = 10


def _build(*, structured: bool, seed: int = 0, n: int = 20_000) -> tuple[pd.DataFrame, pd.Series]:
    """Ten categorical features; the rare class is either clustered or scattered."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.integers(0, 4, size=(n, FEATURES)).astype(np.int8),
        columns=[f"c{i}" for i in range(FEATURES)],
    )
    risk = (
        0.02 + 0.40 * ((X["c0"] == 0) & (X["c1"] == 0))
        if structured
        else pd.Series(np.full(n, 0.05))
    )
    y = pd.Series(np.where(rng.random(n) < risk, 2, 0).astype(np.int8))
    return X, y


@pytest.fixture(scope="module")
def structured() -> tuple[pd.DataFrame, pd.Series]:
    return _build(structured=True)


@pytest.fixture(scope="module")
def unstructured() -> tuple[pd.DataFrame, pd.Series]:
    return _build(structured=False)


def _supported(profile: pd.DataFrame, minimum: float = 100.0) -> pd.DataFrame:
    """Rows with enough expected neighbours for the lift to mean anything.

    A hundred expected gives a standard error near ten percent of the base rate,
    so a lift of one should land inside a fairly narrow band. Below that,
    ordinary fluctuation produces lifts of 0.6 or 1.4 on data with no structure
    whatever, and reading them would be reading noise.
    """
    return profile[profile["expected"] >= minimum]


def test_a_clustered_class_shows_lift_near_by(
    structured: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = structured
    profile = _supported(locality_profile(X, y, target_class=2, n_samples=200_000))
    near = profile[profile["distance"] <= 2]
    assert float(near["lift"].min()) > 2.0


def test_a_scattered_class_shows_no_lift_anywhere(
    unstructured: tuple[pd.DataFrame, pd.Series],
) -> None:
    """The result that would tell you a model has nothing to generalise along."""
    X, y = unstructured
    profile = _supported(locality_profile(X, y, target_class=2, n_samples=200_000))
    assert not profile.empty
    assert profile["lift"].between(0.8, 1.25).all()


def _near_over_far(profile: pd.DataFrame) -> float:
    """Ratio of the lift close by to the lift far away.

    The direct signature of local structure, and the quantity the whole measure
    exists to report: above one means near neighbours resemble each other more
    than distant records do.
    """
    supported = _supported(profile)
    near = float(supported.loc[supported["distance"] <= 2, "lift"].mean())
    far = float(supported.loc[supported["distance"] >= 6, "lift"].mean())
    return near / far


def test_lift_decays_with_distance_when_structure_exists(
    structured: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = structured
    assert _near_over_far(locality_profile(X, y, target_class=2, n_samples=200_000)) > 2.0


def test_lift_does_not_decay_when_there_is_no_structure(
    unstructured: tuple[pd.DataFrame, pd.Series],
) -> None:
    """The comparison that distinguishes the two cases, stated as one number."""
    X, y = unstructured
    ratio = _near_over_far(locality_profile(X, y, target_class=2, n_samples=200_000))
    assert 0.75 < ratio < 1.35


def test_the_near_distances_are_computed_exactly(
    structured: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Sampling never lands close in ten dimensions, so near rows must be exhaustive."""
    X, y = structured
    profile = locality_profile(X, y, target_class=2, exact_to=2, n_samples=200_000)
    assert profile.loc[profile["distance"] <= 2, "exact"].all()
    assert not profile.loc[profile["distance"] > 2, "exact"].any()


def test_distance_zero_agrees_with_the_cell_counts(
    structured: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Distance zero is the cell, so the two must give the same pair count."""
    from vision_zero.ceiling.bayes_error import cell_labels

    X, y = structured
    sizes = cell_labels(X).value_counts()
    expected = int((sizes * (sizes - 1)).sum())

    agreement = pair_agreement(X, y, exact_to=1, n_samples=0)
    assert int(agreement.loc[agreement["distance"] == 0, "pairs"].iloc[0]) == expected


def test_distance_one_counts_the_right_pairs() -> None:
    """Checked by hand: three records, two of which differ in one column."""
    X = pd.DataFrame({"a": ["x", "x", "y"], "b": ["p", "q", "q"]})
    y = pd.Series([1, 1, 0])

    agreement = pair_agreement(X, y, exact_to=2, n_samples=0)
    at_one = agreement[agreement["distance"] == 1]
    # (0,1) differ in b only; (1,2) differ in a only. Two ordered pairs each way.
    assert int(at_one["pairs"].iloc[0]) == 4
    # Of those, (0,1) agree on the outcome and (1,2) do not.
    assert float(at_one["agreement"].iloc[0]) == pytest.approx(0.5)


def test_agreement_reports_the_chance_level(
    unstructured: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Agreement means nothing without the rate two random records would achieve."""
    X, y = unstructured
    agreement = pair_agreement(X, y, n_samples=200_000)
    shares = y.value_counts(normalize=True)
    assert agreement.attrs["chance_agreement"] == pytest.approx(float((shares**2).sum()), abs=1e-9)
    far = agreement[agreement["distance"] >= 6]
    assert far["agreement"].sub(agreement.attrs["chance_agreement"]).abs().max() < 0.02


def test_a_lift_from_too_few_pairs_is_flagged_by_its_expectation(
    unstructured: tuple[pd.DataFrame, pd.Series],
) -> None:
    """The closest distances can hold a handful of pairs, where lift is noise."""
    X, y = unstructured
    profile = locality_profile(X, y, target_class=2, n_samples=0)
    thin = profile[profile["expected"] < 5]
    assert not thin.empty, "the constructed case should have a thinly populated row"
    assert (thin["pairs"] * profile.attrs["base_rate"] < 5).all()


def test_an_absent_class_returns_nothing(unstructured: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = unstructured
    assert locality_profile(X, y, target_class=99, n_samples=0).empty


def test_blank_values_are_a_value_rather_than_a_hole() -> None:
    """Two records both unrecorded in a field agree on it, as elsewhere in this package."""
    X = pd.DataFrame({"a": [None, None, "z"], "b": ["p", "p", "p"]})
    y = pd.Series([1, 1, 0])
    agreement = pair_agreement(X, y, exact_to=1, n_samples=0)
    assert int(agreement.loc[agreement["distance"] == 0, "pairs"].iloc[0]) == 2


def test_the_grouped_counts_match_brute_force() -> None:
    """The correction is arithmetic, so it can be checked against the slow way.

    Counting pairs by grouping is fast because it never enumerates them, and the
    price is that a pair at distance `t` is counted once for every choice of
    ignored features that covers its differences. Getting that multiplicity
    wrong is invisible in the output and shifts every figure that follows, so it
    is checked here against an exhaustive comparison on a dataset small enough
    to permit one.
    """
    rng = np.random.default_rng(0)
    n, k = 400, 6
    codes = rng.integers(0, 3, size=(n, k))
    X = pd.DataFrame(codes, columns=[f"f{i}" for i in range(k)])
    y = pd.Series(rng.integers(0, 2, n))

    distances = (codes[:, None, :] != codes[None, :, :]).sum(axis=2)
    agrees = y.to_numpy()[:, None] == y.to_numpy()[None, :]
    np.fill_diagonal(distances, -1)

    grouped = pair_agreement(X, y, exact_to=3, n_samples=0).set_index("distance")
    pairs = grouped["pairs"].astype(int).to_dict()
    agreement = grouped["agreement"].astype(float).to_dict()
    for distance in range(4):
        mask = distances == distance
        assert pairs[distance] == int(mask.sum()), distance
        assert agreement[distance] == pytest.approx(float(agrees[mask].mean())), distance


def test_the_class_counts_match_brute_force() -> None:
    """The same check for the per-class lift, which uses the same inversion."""
    rng = np.random.default_rng(1)
    n, k = 400, 6
    codes = rng.integers(0, 3, size=(n, k))
    X = pd.DataFrame(codes, columns=[f"f{i}" for i in range(k)])
    y = pd.Series((rng.random(n) < 0.2).astype(int) * 2)

    distances = (codes[:, None, :] != codes[None, :, :]).sum(axis=2)
    np.fill_diagonal(distances, -1)
    is_target = y.to_numpy() == 2

    profile = locality_profile(X, y, target_class=2, exact_to=3, n_samples=0).set_index("distance")
    pairs = profile["pairs"].astype(int).to_dict()
    shares = profile["share"].astype(float).to_dict()
    neighbour_is_target = is_target[None, :].repeat(n, 0)
    for distance in range(4):
        mask = (distances == distance) & is_target[:, None]
        assert pairs[distance] == int(mask.sum()), distance
        assert shares[distance] == pytest.approx(float(neighbour_is_target[mask].mean())), distance


# --------------------------------------------------------------------------
# Which variables differ, not just how many
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def two_of_eight() -> tuple[pd.DataFrame, pd.Series]:
    """Eight variables, of which exactly two carry the rare outcome."""
    rng = np.random.default_rng(0)
    n, k = 40_000, 8
    X = pd.DataFrame(
        rng.integers(0, 3, size=(n, k)).astype(np.int8), columns=[f"v{i}" for i in range(k)]
    )
    risk = 0.03 + 0.35 * ((X["v0"] == 0) & (X["v1"] == 0))
    y = pd.Series(np.where(rng.random(n) < risk, 2, 0).astype(np.int8))
    return X, y


def test_single_variable_differences_find_the_informative_ones(
    two_of_eight: tuple[pd.DataFrame, pd.Series],
) -> None:
    """The measure `locality_profile` averages away, recovered one variable at a time."""
    X, y = two_of_eight
    profile = difference_set_profile(X, y, target_class=2, max_size=1, min_expected=30)
    assert set(profile.head(2)["variables"]) == {"v0", "v1"}


def test_an_inert_variable_leaves_the_enrichment_intact(
    two_of_eight: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Changing a variable that carries nothing should not move the outcome.

    So its lift sits at the identical-record lift rather than at one, which is
    why that figure and not unity is the reference.
    """
    X, y = two_of_eight
    profile = difference_set_profile(X, y, target_class=2, max_size=1, min_expected=30)
    reference = profile.attrs["identical_lift"]

    inert = profile[profile["variables"].isin([f"v{i}" for i in range(2, 8)])]
    assert len(inert) == 6
    assert (inert["lift"] / reference).between(0.9, 1.1).all()


def test_an_informative_variable_destroys_the_enrichment(
    two_of_eight: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = two_of_eight
    profile = difference_set_profile(X, y, target_class=2, max_size=1, min_expected=30)
    reference = profile.attrs["identical_lift"]

    informative = profile[profile["variables"].isin(["v0", "v1"])]
    assert len(informative) == 2
    assert (informative["lift"] < 0.4 * reference).all()


def test_pairs_of_variables_are_reported_too(
    two_of_eight: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = two_of_eight
    profile = difference_set_profile(X, y, target_class=2, max_size=2, min_expected=30)
    assert set(profile["size"]) == {1, 2}
    assert profile["lift"].is_monotonic_increasing


def test_an_absent_target_gives_an_empty_frame(
    two_of_eight: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = two_of_eight
    assert difference_set_profile(X, y, target_class=99).empty


def test_confining_differences_to_a_block_separates_signal_from_noise(
    two_of_eight: tuple[pd.DataFrame, pd.Series],
) -> None:
    """The model-free counterpart to the ablation, on data with a known answer."""
    X, y = two_of_eight
    blocks = {"signal": ["v0", "v1"], "noise": [f"v{i}" for i in range(2, 8)]}
    result = confined_difference_agreement(X, y, blocks, n_controls=10).set_index("block")
    difference = result["difference"].astype(float).to_dict()

    assert difference["signal"] < 0, "differing within the signal must cost agreement"
    assert difference["noise"] > 0, "differing within the noise must not"


def test_the_control_is_matched_for_size(
    two_of_eight: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Comparing a block against records identical throughout would be wrong.

    Agreement weights groups by the pairs they contain, so refining a partition
    destroys its large groups and can *lower* agreement. A finer partition is
    therefore not an upper bound on a coarser one, and blocks of different sizes
    impose partitions of different granularity. The control has to hold size
    fixed, which is what a random same-sized block does.
    """
    X, y = two_of_eight
    blocks = {"two": ["v0", "v1"], "six": [f"v{i}" for i in range(2, 8)]}
    result = confined_difference_agreement(X, y, blocks, n_controls=10).set_index("block")
    variables = result["variables"].astype(int).to_dict()
    agreement = result["agreement"].astype(float).to_dict()
    difference = result["difference"].astype(float).to_dict()

    assert variables == {"two": 2, "six": 6}
    # The raw agreements are not comparable across sizes; the differences are.
    assert agreement["six"] > agreement["two"]
    assert difference["six"] > difference["two"]


def test_refining_a_partition_can_lower_agreement() -> None:
    """The fact that rules out the obvious reference, pinned as a test.

    One large homogeneous group and many small mixed ones. Splitting the large
    group destroys the pairs that were carrying the agreement, so the finer
    partition scores lower despite conditioning on more.
    """
    rng = np.random.default_rng(0)
    labels = np.concatenate([np.zeros(9000, dtype=int), rng.integers(0, 2, 1000)])
    coarse = np.concatenate([np.zeros(9000, dtype=int), np.arange(1, 1001)])
    fine = np.arange(len(labels)) // 2

    def agreement(group: np.ndarray) -> float:
        frame = pd.DataFrame({"g": group, "y": labels})
        # `unstack` on an already-aggregated size, not a pivot of raw values.
        counts = frame.groupby(["g", "y"]).size().unstack(fill_value=0).to_numpy()  # noqa: PD010
        sizes = counts.sum(axis=1)
        return float((counts * (counts - 1)).sum() / (sizes * (sizes - 1)).sum())

    assert agreement(fine) < agreement(coarse)


def test_a_block_naming_absent_columns_is_skipped(
    two_of_eight: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = two_of_eight
    result = confined_difference_agreement(X, y, {"nowhere": ["absent"], "signal": ["v0"]})
    assert list(result["block"]) == ["signal"]
