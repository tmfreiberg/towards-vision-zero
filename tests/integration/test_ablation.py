"""Block ablation and the blank-field cap, against the fixture."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from vision_zero.ceiling.ablation import block_ablation, cumulative_ablation
from vision_zero.data.prepare import prepare
from vision_zero.data.schema import FEATURE_BLOCKS, blocks_for
from vision_zero.modeling.config import Experiment
from vision_zero.modeling.dataset import Dataset, build_dataset

FIXTURE_RAW = Path(__file__).resolve().parents[1] / "fixtures" / "raw"
YEARS = [2011, 2012, 2013]


def _spec(**data_overrides: object) -> Experiment:
    data: dict[str, object] = {
        "years": YEARS,
        "target": "TNRY_SEV",
        "exclude_features": ["YEAR"],
    }
    data.update(data_overrides)
    return Experiment.model_validate(
        {
            "name": "fixture",
            "data": data,
            "split": {"test_size": 0.2, "folds": 2, "seed": 7},
            "model": {"kind": "decision_tree", "params": {"max_depth": 4}},
        }
    )


@pytest.fixture(scope="module")
def frame(tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    root = tmp_path_factory.mktemp("ablation")
    shutil.copytree(FIXTURE_RAW, root / "raw")
    return pd.read_parquet(prepare(YEARS, root / "raw", root / "proc").parquet_path)


@pytest.fixture(scope="module")
def dataset(frame: pd.DataFrame) -> Dataset:
    return build_dataset(_spec(), frame)


def test_blocks_partition_the_features_without_overlap() -> None:
    members = [name for block in FEATURE_BLOCKS.values() for name in block]
    assert len(members) == len(set(members)), "a feature belongs to at most one block"


def test_every_feature_belongs_to_a_block(dataset: Dataset) -> None:
    """An unassigned feature would silently vanish from the ablation."""
    assigned = {name for block in blocks_for(dataset.features).values() for name in block}
    assert set(dataset.features) == assigned


def test_ablation_covers_every_block_plus_the_reference_rows(dataset: Dataset) -> None:
    result = block_ablation(_spec(), dataset, folds=2, subsample=None)
    expected = set(blocks_for(dataset.features)) | {"majority baseline", "all features"}
    assert set(result["block"]) == expected


def test_ablation_reports_both_alone_and_added(dataset: Dataset) -> None:
    result = block_ablation(_spec(), dataset, folds=2, subsample=None)
    blocks = result[~result["block"].isin(["majority baseline", "all features"])]
    assert blocks["mcc_alone"].notna().all()
    assert blocks["mcc_added"].notna().all()


def test_the_full_feature_set_is_scored_for_reference(dataset: Dataset) -> None:
    result = block_ablation(_spec(), dataset, folds=2, subsample=None).set_index("block")
    counts = result["n_features"].astype(int).to_dict()
    assert counts["all features"] == len(dataset.features)
    assert counts["majority baseline"] == 0


def test_cumulative_ablation_grows_monotonically(dataset: Dataset) -> None:
    steps = cumulative_ablation(_spec(), dataset, folds=2, subsample=None)
    assert steps["n_features"].is_monotonic_increasing
    assert steps["n_features"].iloc[0] == 0
    assert steps["n_features"].iloc[-1] == len(dataset.features)


def test_cumulative_ablation_reports_each_step_change(dataset: Dataset) -> None:
    steps = cumulative_ablation(_spec(), dataset, folds=2, subsample=None)
    assert pd.isna(steps["mcc_delta"].iloc[0])
    assert steps["mcc_delta"].iloc[1:].notna().all()


def test_an_unknown_block_in_the_order_is_skipped(dataset: Dataset) -> None:
    steps = cumulative_ablation(
        _spec(), dataset, order=("participants", "nonexistent"), folds=2, subsample=None
    )
    assert list(steps["step"]) == ["majority baseline", "+ participants"]


# --------------------------------------------------------------------------
# The blank-field cap
# --------------------------------------------------------------------------


def test_capping_blanks_reduces_the_record_count(frame: pd.DataFrame) -> None:
    unrestricted = build_dataset(_spec(), frame)
    capped = build_dataset(_spec(max_missing_features=0), frame)
    assert capped.n_train < unrestricted.n_train


def test_a_generous_cap_keeps_everything(frame: pd.DataFrame) -> None:
    unrestricted = build_dataset(_spec(), frame)
    capped = build_dataset(_spec(max_missing_features=99), frame)
    assert capped.n_train == unrestricted.n_train


def test_a_zero_cap_leaves_no_blanks_at_all(frame: pd.DataFrame) -> None:
    capped = build_dataset(_spec(max_missing_features=0), frame)
    assert not capped.X_train.isna().any().any()


def test_the_cap_is_validated() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _spec(max_missing_features=-1)


def test_a_stricter_filter_holds_out_the_same_records(frame: pd.DataFrame) -> None:
    """Two experiments differing only in how strictly they filter must share a vault.

    The split is taken before any quality filter, so the looser experiment's
    training set contains the stricter one's, and neither one's training data
    can overlap the other's held-out set. Filtering first would give them
    unrelated splits and quietly break that.
    """
    loose = build_dataset(_spec(), frame)
    strict = build_dataset(_spec(max_missing_features=0), frame)

    loose_train = set(map(tuple, loose.X_train[list(strict.features)].to_numpy()))
    assert strict.n_train <= loose.n_train
    assert strict.n_test <= loose.n_test
    assert loose_train, "sanity: the loose training set is not empty"


def test_the_held_out_set_shrinks_with_the_filter_too(frame: pd.DataFrame) -> None:
    """The held-out records must come from the same population as the training
    ones, so a filter applied to one is applied to both."""
    unfiltered = build_dataset(_spec(), frame)
    filtered = build_dataset(_spec(max_missing_features=0), frame)

    assert filtered.n_test < unfiltered.n_test
    assert not filtered.X_test.isna().any().any()
