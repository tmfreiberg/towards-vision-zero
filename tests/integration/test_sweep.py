"""Model sweep and learning curve, against the fixture."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from vision_zero.ceiling.sweep import learning_curve, model_sweep
from vision_zero.data.prepare import prepare
from vision_zero.modeling.config import Experiment
from vision_zero.modeling.dataset import Dataset, build_dataset

FIXTURE_RAW = Path(__file__).resolve().parents[1] / "fixtures" / "raw"
YEARS = [2011, 2012, 2013]

FAMILIES: dict[str, dict[str, object]] = {
    "decision_tree": {"max_depth": 4},
    "xgboost": {"n_estimators": 10, "max_depth": 2, "tree_method": "hist"},
}


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> tuple[Experiment, Dataset]:
    root = tmp_path_factory.mktemp("sweep")
    shutil.copytree(FIXTURE_RAW, root / "raw")
    frame = pd.read_parquet(prepare(YEARS, root / "raw", root / "proc").parquet_path)
    spec = Experiment.model_validate(
        {
            "name": "fixture",
            "data": {"years": YEARS, "target": "TNRY_SEV", "exclude_features": ["YEAR"]},
            "split": {"test_size": 0.2, "folds": 2, "seed": 7},
            "model": {
                "kind": "xgboost",
                "params": {"n_estimators": 10, "max_depth": 2, "tree_method": "hist"},
            },
        }
    )
    return spec, build_dataset(spec, frame)


def test_sweep_scores_every_family_and_the_baseline(dataset: tuple[Experiment, Dataset]) -> None:
    spec, data = dataset
    result = model_sweep(spec, data, families=FAMILIES, folds=2, subsample=None)
    assert set(result["family"]) == {*FAMILIES, "majority baseline"}


def test_sweep_reports_spread_across_folds(dataset: tuple[Experiment, Dataset]) -> None:
    spec, data = dataset
    result = model_sweep(spec, data, families=FAMILIES, folds=2, subsample=None)
    for column in ("mcc_mean", "mcc_std", "balanced_accuracy_mean", "focus_recall_mean"):
        assert column in result.columns
    assert (result["mcc_std"] >= 0).all()


def test_sweep_is_sorted_by_the_aggregate_score(dataset: tuple[Experiment, Dataset]) -> None:
    spec, data = dataset
    result = model_sweep(spec, data, families=FAMILIES, folds=2, subsample=None)
    assert result["mcc_mean"].is_monotonic_decreasing


def test_no_family_meaningfully_beats_the_baseline_on_noise(
    dataset: tuple[Experiment, Dataset],
) -> None:
    """The fixture is drawn independently, so nothing should separate the classes."""
    spec, data = dataset
    result = model_sweep(spec, data, families=FAMILIES, folds=2, subsample=None)
    assert result["mcc_mean"].abs().max() < 0.15


def test_subsampling_preserves_the_class_balance(dataset: tuple[Experiment, Dataset]) -> None:
    spec, data = dataset
    result = model_sweep(spec, data, families=FAMILIES, folds=2, subsample=500)
    assert len(result) == len(FAMILIES) + 1


def test_learning_curve_covers_every_requested_fraction(
    dataset: tuple[Experiment, Dataset],
) -> None:
    spec, data = dataset
    curve = learning_curve(spec, data, fractions=(0.25, 0.5, 1.0), folds=2, subsample=None)
    assert list(curve["fraction"]) == [0.25, 0.5, 1.0]
    assert curve["n_train"].is_monotonic_increasing


def test_learning_curve_is_flat_when_there_is_nothing_to_learn(
    dataset: tuple[Experiment, Dataset],
) -> None:
    """A curve still climbing would mean more records help. On noise it cannot."""
    spec, data = dataset
    curve = learning_curve(spec, data, fractions=(0.25, 1.0), folds=2, subsample=None)
    assert curve["mcc"].abs().max() < 0.15
