r"""Whether collisions that resemble each other turn out alike.

Every classifier rests on an assumption about local structure. Records close
together in feature space are supposed to have similar outcomes, because that is
what lets a model say anything about a combination of circumstances it has never
seen: it borrows from the combinations it has seen nearby.

That assumption is testable, and on categorical features closeness has an exact
meaning.

Notation
--------
Let :math:`R = \{1, \dots, n\}` index the records, let :math:`x_i(r)` be the
value of variable :math:`i \in \{1, \dots, k\}` for record :math:`r`, and let
:math:`y(r) \in \{0, \dots, M-1\}` be its recorded outcome. Records are
indexed rather than identified with their vectors, so :math:`r \neq s` always
means distinct indices and never distinct vectors. Write

.. math::

    \Delta(r,s) = \{\, i : x_i(r) \neq x_i(s) \,\}

for the set of variables on which two records disagree, so the Hamming distance
is :math:`|\Delta(r,s)|`. All counts below are of ordered pairs off the
diagonal, which double-counts each unordered pair and cancels in every ratio
reported.

.. math::

    P_t &= \#\{\, (r,s) : r \neq s,\ |\Delta(r,s)| = t \,\} \\
    Q_t &= \#\{\, (r,s) : r \neq s,\ |\Delta(r,s)| = t,\ y(r) = y(s) \,\}

:math:`P_0` is not zero: it counts pairs of distinct records carrying identical
vectors, which is :math:`\sum_c n_c(n_c - 1)` over cells of size :math:`n_c`,
the same quantity :mod:`~vision_zero.ceiling.bayes_error` works with.

What is reported
----------------
:func:`pair_agreement` gives :math:`Q_t / P_t`, the chance that two collisions
differing on exactly :math:`t` recorded variables shared an outcome. Two
reference points fix the scale. At :math:`t = k` the records share nothing and
behave as independent draws, so the ratio approaches :math:`\sum_c \pi_c^2`
where :math:`\pi_c` is the share of class :math:`c`. At :math:`t = 0` the pair
is co-cellular and the ratio is what @sec-limits already reports.

A curve sloping from :math:`t = 0` down toward the chance rate means similar
collisions really do resemble each other in outcome, and a model has something
to generalise along. A flat curve means they do not, and a model's answers about
unseen combinations rest on an assumption the data does not support.

For a rare class the agreement ratio is dominated by the majority class agreeing
with itself, so :func:`locality_profile` reports a class-conditional form
instead. Fixing a class :math:`c`,

.. math::

    P_t^{(c)} &= \#\{\, (r,s) : r \neq s,\ |\Delta(r,s)| = t,\ y(r) = c
        \,\} \\
    Q_t^{(c)} &= \#\{\, (r,s) : r \neq s,\ |\Delta(r,s)| = t,\ y(r) =
        y(s) = c \,\}

and the reported **lift** is

.. math::

    \mathrm{lift}_c(t) = \frac{Q_t^{(c)} / P_t^{(c)}}{\pi_c} .

A lift of one says a member of class :math:`c` has neighbours at that distance
no more likely to be in :math:`c` than a stranger is, which is the direct
statement that the class does not cluster in feature space. Note that
:math:`P_t = \sum_c P_t^{(c)}` and :math:`Q_t = \sum_c Q_t^{(c)}`, so both
views come from one tally.

How it is computed
------------------
Comparing every pair is quadratic and out of the question at this size, so the
counts are obtained by grouping instead; :func:`_grouped_totals` explains the
identity that makes this work and :func:`_invert_multiplicities` inverts it.
Every one of :math:`P_t`, :math:`Q_t`, :math:`P_t^{(c)}` and :math:`Q_t^{(c)}`
is a pair count filtered by a condition on :math:`(y(r), y(s))` alone, which is
independent of which variables are ignored, so each satisfies the same identity
and each is inverted separately.

Distances up to :data:`MAX_EXACT_DISTANCE` are computed exactly. Beyond that the
pairs are sampled, which suffices because the question concerns near
neighbours, and because in eighteen dimensions two records drawn at random are
almost never close.

What this does not distinguish
------------------------------
The distance counts how many variables differ and ignores which. Two pairs both
at distance three, one differing in month, weekday and hour and the other in
speed limit, road configuration and collision type, are indistinguishable here
and are averaged together. If some variables carry the outcome and most do not,
the pairs differing only in the inert ones will dominate the tally at every
distance and hold the curve up.

The measure is therefore conservative in one direction and not the other. A
curve that decays establishes that local structure exists. A flat curve does not
establish that none does, only that none survives averaging over which variables
happen to differ. @sec-matters addresses the same question from the other side,
by asking what each group of variables contributes when the others are known.

Examples
--------
>>> import pandas as pd
>>> X = pd.DataFrame({"a": ["x", "x", "y"], "b": ["p", "q", "q"]})
>>> y = pd.Series([1, 1, 0])
>>> profile = locality_profile(X, y, target_class=1, exact_to=1, n_samples=0)
>>> int(profile.loc[profile["distance"] == 1, "pairs"].iloc[0])
2
"""

from __future__ import annotations

from itertools import combinations
from math import comb
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from vision_zero.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "MAX_EXACT_DISTANCE",
    "confined_difference_agreement",
    "difference_set_profile",
    "locality_profile",
    "pair_agreement",
]

log = get_logger(__name__)

#: Distances computed by exhaustive grouping rather than sampling. Three would
#: mean 816 groupings on eighteen variables, which is slow for little gain.
MAX_EXACT_DISTANCE = 2


def _codes(X: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    """Encode the frame as small integers, with blanks as their own value."""
    columns = []
    for name in features:
        codes, _ = pd.factorize(X[name], use_na_sentinel=False)
        columns.append(codes.astype(np.int32))
    return np.column_stack(columns)


def _totals_for_subset(
    codes: np.ndarray, labels: np.ndarray, ignored: tuple[int, ...], n_classes: int
) -> tuple[int, int, np.ndarray, np.ndarray]:
    r"""Tally ordered pairs whose differences lie inside one subset.

    Partitions the records by their values outside ``ignored``, so that two
    records fall in the same class exactly when :math:`\Delta(r,s) \subseteq A`
    for :math:`A` the ignored set. Counting within-class pairs then gives
    :math:`G_A`, and the same pass gives the agreeing count and the per-class
    breakdowns.

    Returns
    -------
    tuple
        Ordered pairs, pairs sharing an outcome, pairs by the class of their
        first member, and pairs by the classes of both.
    """
    n_features = codes.shape[1]
    kept = [index for index in range(n_features) if index not in ignored]
    block = codes[:, kept] if kept else np.zeros((len(codes), 1), dtype=np.int32)
    frame = pd.DataFrame(block)
    group = frame.groupby(list(frame.columns), sort=False).ngroup().to_numpy()

    counts = np.zeros((int(group.max()) + 1, n_classes), dtype=np.int64)
    np.add.at(counts, (group, labels), 1)
    sizes = counts.sum(axis=1)

    pairs = int((sizes * (sizes - 1)).sum())
    agree = int((counts * (counts - 1)).sum())
    joint = counts.T @ counts - np.diag(counts.sum(axis=0))
    first = (counts * (sizes[:, None] - 1)).sum(axis=0)
    return pairs, agree, first, joint


def _grouped_totals(
    codes: np.ndarray, labels: np.ndarray, distance: int, n_classes: int
) -> tuple[int, int, np.ndarray, np.ndarray]:
    r"""Sum :func:`_totals_for_subset` over every subset of a given size.

    Notation. Let :math:`R = \{1, \dots, n\}` index the records and let
    :math:`x_i(r)` be the value of variable :math:`i` for record :math:`r`. Two
    records may share every value while remaining distinct records, so
    :math:`r \neq s` below means distinct *indices* and never distinct vectors.
    With :math:`\Delta(r,s) = \{\, i : x_i(r) \neq x_i(s) \,\}`, write

    .. math::

        P_t = \#\{\, (r,s) : r \neq s,\ |\Delta(r,s)| = t \,\}, \qquad
        G_A = \#\{\, (r,s) : r \neq s,\ \Delta(r,s) \subseteq A \,\}.

    This function returns :math:`S_d = \sum_{|A| = d} G_A`.

    Why that is not :math:`P_d`, or even :math:`\sum_{t \le d} P_t`. A pair at
    distance :math:`t` satisfies :math:`\Delta(r,s) \subseteq A` for every
    :math:`A` of size :math:`d` containing its fixed :math:`t`-element
    difference set, and there are :math:`\binom{k-t}{d-t}` such sets. Hence

    .. math::

        S_d = \sum_{t \le d} \binom{k-t}{d-t} P_t ,

    which :func:`_invert_multiplicities` solves for :math:`P_d`. Taking
    :math:`S_d - S_{d-1}` instead, which is the obvious move and wrong, leaves
    :math:`k-1` spurious copies of every identical pair in the distance-one
    count.

    Cost. One subset requires hashing :math:`n` records on :math:`k - d`
    coordinates and one pass over the resulting classes, so :math:`S_d` costs
    :math:`O\!\left(\binom{k}{d} n k\right)`: linear in the number of
    records, and binomial in :math:`k`. That factor is why
    :data:`MAX_EXACT_DISTANCE` is two; at eighteen variables the three distances
    need 1, 18 and 153 groupings, and the next would need 816.
    """
    n_features = codes.shape[1]
    at_most_pairs = at_most_agree = 0
    at_most_first = np.zeros(n_classes, dtype=np.int64)
    at_most_joint = np.zeros((n_classes, n_classes), dtype=np.int64)

    for ignored in combinations(range(n_features), distance):
        pairs, agree, first, joint = _totals_for_subset(codes, labels, ignored, n_classes)
        at_most_pairs += pairs
        at_most_agree += agree
        at_most_first += first
        at_most_joint += joint

    return at_most_pairs, at_most_agree, at_most_first, at_most_joint


def _invert_multiplicities(totals: list[float], n_features: int) -> list[float]:
    r"""Recover per-distance counts from the grouped totals.

    Solves :math:`S_d = \sum_{t \le d} inom{k-t}{d-t} P_t` for :math:`P_d`,
    working outward from :math:`P_0 = S_0`.
    """
    exact: list[float] = []
    for distance, total in enumerate(totals):
        overlap = sum(
            comb(n_features - closer, distance - closer) * exact[closer]
            for closer in range(distance)
        )
        exact.append(total - overlap)
    return exact


def pair_agreement(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    features: Sequence[str] | None = None,
    exact_to: int = MAX_EXACT_DISTANCE,
    n_samples: int = 2_000_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Tabulate outcome agreement against Hamming distance.

    Parameters
    ----------
    X, y
        Features and labels, aligned.
    features
        Columns forming the distance. All by default.
    exact_to
        Compute distances up to this value by exhaustive grouping. Beyond it,
        sample.
    n_samples
        Random pairs to draw for the sampled range. Zero skips it.
    seed
        Seed for the sampling.

    Returns
    -------
    pandas.DataFrame
        One row per distance, with the number of ordered pairs, the share
        sharing an outcome, and whether the row was computed exactly.
    """
    columns = list(features) if features is not None else list(X.columns)
    codes = _codes(X, columns)
    labels = np.asarray(y)
    classes, labels = np.unique(labels, return_inverse=True)
    n_classes = len(classes)

    grouped_pairs, grouped_agree = [], []
    for distance in range(exact_to + 1):
        pairs, agree, _, _ = _grouped_totals(codes, labels, distance, n_classes)
        grouped_pairs.append(float(pairs))
        grouped_agree.append(float(agree))

    n_features = codes.shape[1]
    exact_pairs = _invert_multiplicities(grouped_pairs, n_features)
    exact_agree = _invert_multiplicities(grouped_agree, n_features)

    rows: list[dict[str, object]] = []
    for distance, (at_distance, sharing) in enumerate(zip(exact_pairs, exact_agree, strict=True)):
        if at_distance > 0:
            rows.append(
                {
                    "distance": distance,
                    "pairs": round(at_distance),
                    "agreement": sharing / at_distance,
                    "exact": True,
                }
            )

    if n_samples:
        rng = np.random.default_rng(seed)
        first = rng.integers(0, len(codes), n_samples)
        second = rng.integers(0, len(codes), n_samples)
        keep = first != second
        first, second = first[keep], second[keep]
        distances = (codes[first] != codes[second]).sum(axis=1)
        agrees = labels[first] == labels[second]

        for distance in range(exact_to + 1, codes.shape[1] + 1):
            mask = distances == distance
            if int(mask.sum()) < 400:
                continue
            rows.append(
                {
                    "distance": distance,
                    "pairs": int(mask.sum()),
                    "agreement": float(agrees[mask].mean()),
                    "exact": False,
                }
            )

    frame = pd.DataFrame(rows)
    shares = np.bincount(labels, minlength=n_classes) / len(labels)
    frame.attrs["chance_agreement"] = float((shares**2).sum())
    return frame


def locality_profile(
    X: pd.DataFrame,
    y: pd.Series,
    target_class: int,
    *,
    features: Sequence[str] | None = None,
    exact_to: int = MAX_EXACT_DISTANCE,
    n_samples: int = 2_000_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Measure whether one class clusters in feature space.

    For each distance, reports how often a neighbour of a record in the target
    class is itself in that class, against how often any record is. The ratio is
    the lift, and a lift near one at every distance says the class is scattered
    rather than concentrated.

    Parameters
    ----------
    X, y
        Features and labels, aligned.
    target_class
        The class to trace.
    features
        Columns forming the distance.
    exact_to
        Compute distances up to this value exactly.
    n_samples
        Random pairs for the sampled range.
    seed
        Seed for the sampling.

    Returns
    -------
    pandas.DataFrame
        One row per distance, with the pair count, how many neighbours in the
        target class the base rate alone would produce, the share observed, the
        lift over the base rate, and whether the row was computed exactly.

        Read the lift only where ``expected`` is comfortably above zero. At the
        closest distances there may be very few pairs, and a lift computed from
        a handful of them says nothing.
    """
    columns = list(features) if features is not None else list(X.columns)
    codes = _codes(X, columns)
    raw = np.asarray(y)
    classes, labels = np.unique(raw, return_inverse=True)
    if target_class not in classes:
        return pd.DataFrame(columns=["distance", "pairs", "expected", "share", "lift", "exact"])

    position = int(np.flatnonzero(classes == target_class)[0])
    n_classes = len(classes)
    base = float((labels == position).mean())

    grouped_first, grouped_joint = [], []
    for distance in range(exact_to + 1):
        _, _, first, joint = _grouped_totals(codes, labels, distance, n_classes)
        grouped_first.append(float(first[position]))
        grouped_joint.append(float(joint[position, position]))

    n_features = codes.shape[1]
    exact_first = _invert_multiplicities(grouped_first, n_features)
    exact_joint = _invert_multiplicities(grouped_joint, n_features)

    rows: list[dict[str, object]] = []
    for distance, (total, both) in enumerate(zip(exact_first, exact_joint, strict=True)):
        if total > 0:
            share = both / total
            rows.append(
                {
                    "distance": distance,
                    "pairs": round(total),
                    # Neighbours in the target class one would see at the base
                    # rate. A lift computed where this is near zero is noise.
                    "expected": total * base,
                    "share": share,
                    "lift": share / base if base else float("nan"),
                    "exact": True,
                }
            )

    if n_samples:
        rng = np.random.default_rng(seed)
        anchors = np.flatnonzero(labels == position)
        if len(anchors):
            first = rng.choice(anchors, n_samples)
            second = rng.integers(0, len(codes), n_samples)
            keep = first != second
            first, second = first[keep], second[keep]
            distances = (codes[first] != codes[second]).sum(axis=1)
            neighbour_is_target = labels[second] == position

            for distance in range(exact_to + 1, codes.shape[1] + 1):
                mask = distances == distance
                if int(mask.sum()) < 400:
                    continue
                share = float(neighbour_is_target[mask].mean())
                rows.append(
                    {
                        "distance": distance,
                        "pairs": int(mask.sum()),
                        "expected": float(mask.sum()) * base,
                        "share": share,
                        "lift": share / base if base else float("nan"),
                        "exact": False,
                    }
                )

    frame = pd.DataFrame(rows)
    frame.attrs["base_rate"] = base
    return frame


def _exact_difference_set(
    codes: np.ndarray,
    labels: np.ndarray,
    difference: tuple[int, ...],
    n_classes: int,
    cache: dict[tuple[int, ...], tuple[int, int, np.ndarray, np.ndarray]],
) -> tuple[int, int, np.ndarray, np.ndarray]:
    r"""Tally pairs whose differences are exactly one given set.

    :math:`G_A` counts pairs with :math:`\Delta(r,s) \subseteq A`, so recovering
    :math:`\Delta(r,s) = D` needs Möbius inversion over the subsets of
    :math:`D`:

    .. math::

        \#\{\, \Delta(r,s) = D \,\} = \sum_{E \subseteq D} (-1)^{|D| - |E|}
        \, G_E .

    The subsets are shared between neighbouring :math:`D`, so they are cached.
    """
    pairs = agree = 0
    first = np.zeros(n_classes, dtype=np.int64)
    joint = np.zeros((n_classes, n_classes), dtype=np.int64)

    for size in range(len(difference) + 1):
        sign = (-1) ** (len(difference) - size)
        for subset in combinations(difference, size):
            if subset not in cache:
                cache[subset] = _totals_for_subset(codes, labels, subset, n_classes)
            subset_pairs, subset_agree, subset_first, subset_joint = cache[subset]
            pairs += sign * subset_pairs
            agree += sign * subset_agree
            first += sign * subset_first
            joint += sign * subset_joint

    return pairs, agree, first, joint


def difference_set_profile(
    X: pd.DataFrame,
    y: pd.Series,
    target_class: int,
    *,
    features: Sequence[str] | None = None,
    max_size: int = 2,
    min_expected: float = 50.0,
) -> pd.DataFrame:
    r"""Measure local influence one difference set at a time.

    :func:`locality_profile` counts how many variables differ and ignores which,
    so pairs differing in the month, the weekday and the hour are averaged
    together with pairs differing in the speed limit, the road configuration and
    the type of collision. Where most variables carry nothing, the pairs
    differing only in the inert ones are the common case and hold the curve up.

    This conditions on the difference set itself. For :math:`D \subseteq
    \{1,\dots,k\}` it reports

    .. math::

        \mathrm{lift}_c(D) = \frac{\Pr[\, y(s) = c \mid y(r) = c,\
        \Delta(r,s) = D \,]}{\pi_c},

    over pairs agreeing on every variable outside :math:`D` and differing on
    every variable inside it.

    The number to compare against is **not one**. It is the lift among records
    identical throughout, :math:`\mathrm{lift}_c(\emptyset)`, which is reported
    in the frame's ``attrs`` and as the ``identical`` column. Holding every
    variable fixed preserves whatever enrichment the features can produce, so
    that figure is the reference; a difference set whose lift falls well below
    it contains variables that move the outcome, and one whose lift stays near
    it contains variables that do not.

    On constructed data where a rare class depends on two variables of eight,
    difference sets containing either of the two return lifts around 0.7 while
    sets containing neither return lifts around 3.5, against an identical-record
    lift of about 3.5. Changing an inert variable leaves the enrichment intact,
    which is exactly what makes it inert.

    Parameters
    ----------
    X, y
        Features and labels, aligned.
    target_class
        The class to trace.
    features
        Columns forming the space.
    max_size
        Largest difference set to report. Two means :math:`k + \binom{k}{2}`
        rows before filtering, and :math:`1 + k + \binom{k}{2}` groupings.
    min_expected
        Suppress rows where the base rate alone would produce fewer than this
        many neighbours in the target class, since a lift from a handful of
        pairs is noise.

    Returns
    -------
    pandas.DataFrame
        One row per difference set, most influential first: the variables, the
        pair count, the expected count under the base rate, the observed share,
        the lift, and the identical-record lift it should be read against.
    """
    columns = list(features) if features is not None else list(X.columns)
    codes = _codes(X, columns)
    raw = np.asarray(y)
    classes, labels = np.unique(raw, return_inverse=True)
    empty = pd.DataFrame(
        columns=["variables", "size", "pairs", "expected", "share", "lift", "identical"]
    )
    if target_class not in classes:
        return empty

    position = int(np.flatnonzero(classes == target_class)[0])
    n_classes = len(classes)
    base = float((labels == position).mean())
    cache: dict[tuple[int, ...], tuple[int, int, np.ndarray, np.ndarray]] = {}

    # The reference: records identical on every variable. Whatever enrichment
    # the features can produce, they produce it here.
    _, _, same_first, same_joint = _totals_for_subset(codes, labels, (), n_classes)
    identical = (
        (float(same_joint[position, position]) / int(same_first[position])) / base
        if int(same_first[position]) and base
        else float("nan")
    )

    rows = []
    for size in range(1, max_size + 1):
        for difference in combinations(range(len(columns)), size):
            _, _, first, joint = _exact_difference_set(codes, labels, difference, n_classes, cache)
            total = int(first[position])
            expected = total * base
            if expected < min_expected:
                continue
            share = float(joint[position, position]) / total
            rows.append(
                {
                    "variables": ", ".join(columns[index] for index in difference),
                    "size": size,
                    "pairs": total,
                    "expected": expected,
                    "share": share,
                    "lift": share / base if base else float("nan"),
                    "identical": identical,
                }
            )

    if not rows:
        return empty
    frame = pd.DataFrame(rows).sort_values("lift").reset_index(drop=True)
    frame.attrs["identical_lift"] = identical
    frame.attrs["base_rate"] = base
    return frame


def confined_difference_agreement(
    X: pd.DataFrame,
    y: pd.Series,
    blocks: Mapping[str, Sequence[str]],
    *,
    features: Sequence[str] | None = None,
    n_controls: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    r"""Ask what a group of variables carries, holding every other one fixed.

    A model-free counterpart to the ablation in @sec-matters. Take a block
    :math:`B` and keep only pairs of records agreeing on every variable outside
    it, so that whatever separates them lies inside :math:`B`. How often do such
    pairs share an outcome?

    If :math:`B` carries nothing, agreeing outside it means agreeing on
    everything that matters, and such pairs should share an outcome often. If
    :math:`B` carries a great deal, differing within it should pull them apart.

    .. warning::

       The obvious reference, agreement among records identical throughout, is
       **not** an upper bound and must not be used. Agreement over a partition
       is :math:`\sum_g w_g \sum_c p_{gc}^2` with :math:`w_g \propto n_g(n_g -
       1)`, a per-pair weighting. Refining a partition destroys its large groups
       and their weight does not redistribute evenly, so a finer partition can
       show *lower* agreement than a coarser one. Comparing the two therefore
       compares differently weighted populations rather than measuring anything
       about the variables.

       The same objection applies to comparing blocks with each other, since
       "agrees outside :math:`B`" is a stronger condition when :math:`B` is
       small, and a stronger condition means smaller groups.

    The control used instead is a block of the same size drawn at random. For a
    block of :math:`m` variables, the same quantity is computed for
    ``n_controls`` random subsets of :math:`m` variables and averaged. Both then
    face identical granularity, and the difference between them is what the
    block's membership is worth.

    Parameters
    ----------
    X, y
        Features and labels, aligned.
    blocks
        Block name to its member columns.
    features
        Columns forming the space. All by default.
    n_controls
        Random same-sized subsets to average over. Zero skips the control, in
        which case only the raw figures are returned.
    seed
        Seed for the control draws.

    Returns
    -------
    pandas.DataFrame
        One row per block, most informative first: the variables it holds, the
        pairs differing only within it, how often those share an outcome, the
        same figure for random blocks of that size, and the difference. A
        difference near zero says the block's membership does not matter, which
        is to say it carries nothing a same-sized handful of variables would
        not.
    """
    columns = list(features) if features is not None else list(X.columns)
    position_of = {name: index for index, name in enumerate(columns)}
    codes = _codes(X, columns)
    labels = np.unique(np.asarray(y), return_inverse=True)[1]
    n_classes = int(labels.max()) + 1
    rng = np.random.default_rng(seed)

    identical_pairs, identical_agree, _, _ = _totals_for_subset(codes, labels, (), n_classes)

    def confined(ignored: tuple[int, ...]) -> float:
        """Agreement among pairs differing only inside ``ignored``."""
        pairs, agree, _, _ = _totals_for_subset(codes, labels, ignored, n_classes)
        remaining = pairs - identical_pairs
        return (agree - identical_agree) / remaining if remaining > 0 else float("nan")

    sizes = np.bincount(labels) / len(labels)
    chance = float((sizes**2).sum())

    control_cache: dict[int, float] = {}
    rows = []
    for name, members in blocks.items():
        present = tuple(sorted(position_of[m] for m in members if m in position_of))
        if not present:
            continue
        observed = confined(present)
        if np.isnan(observed):
            continue

        size = len(present)
        if n_controls and size not in control_cache:
            draws = [
                confined(tuple(sorted(rng.choice(len(columns), size, replace=False))))
                for _ in range(n_controls)
            ]
            finite = [value for value in draws if not np.isnan(value)]
            control_cache[size] = float(np.mean(finite)) if finite else float("nan")
        control = control_cache.get(size, float("nan"))

        rows.append(
            {
                "block": name,
                "variables": size,
                "pairs": _totals_for_subset(codes, labels, present, n_classes)[0] - identical_pairs,
                "agreement": observed,
                "same_size_control": control,
                "difference": observed - control,
                "chance": chance,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["block", "variables", "pairs", "agreement", "same_size_control", "difference"]
        )
    frame = pd.DataFrame(rows).sort_values("difference").reset_index(drop=True)
    frame.attrs["chance_agreement"] = chance
    return frame
