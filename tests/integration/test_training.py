"""Dataset assembly, pipeline construction, and fitting, against the fixture."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vision_zero.data.prepare import prepare
from vision_zero.modeling.config import Experiment
from vision_zero.modeling.dataset import build_dataset, load_compiled
from vision_zero.modeling.pipeline import build_pipeline, supports_missing
from vision_zero.modeling.train import train

FIXTURE_RAW = Path(__file__).resolve().parents[1] / "fixtures" / "raw"
YEARS = [2011, 2012, 2013]


def _spec(**overrides: object) -> Experiment:
    base: dict[str, object] = {
        "name": "fixture",
        "data": {"years": YEARS, "target": "TNRY_SEV", "exclude_features": ["YEAR"]},
        "split": {"test_size": 0.2, "folds": 3, "seed": 7},
        "model": {
            "kind": "xgboost",
            "params": {"n_estimators": 10, "max_depth": 2, "tree_method": "hist"},
        },
    }
    base.update(overrides)
    return Experiment.model_validate(base)


@pytest.fixture(scope="module")
def frame(tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    root = tmp_path_factory.mktemp("compiled")
    shutil.copytree(FIXTURE_RAW, root / "raw")
    return pd.read_parquet(prepare(YEARS, root / "raw", root / "proc").parquet_path)


def test_dataset_splits_without_overlap(frame: pd.DataFrame) -> None:
    data = build_dataset(_spec(), frame)
    assert data.n_train + data.n_test == len(frame)
    assert data.n_test == pytest.approx(0.2 * len(frame), abs=2)


def test_stratification_preserves_class_shares(frame: pd.DataFrame) -> None:
    """With a class near one percent, an unstratified split can lose it entirely."""
    data = build_dataset(_spec(), frame)
    train_share = data.y_train.value_counts(normalize=True).sort_index()
    test_share = data.y_test.value_counts(normalize=True).sort_index()
    np.testing.assert_allclose(train_share, test_share, atol=0.01)


def test_constant_features_are_dropped(frame: pd.DataFrame) -> None:
    """Restricting to one region leaves REGION constant and useless."""
    data = build_dataset(
        _spec(
            data={"years": YEARS, "target": "TNRY_SEV", "restrict": {"REGION": ["Montréal (06)"]}}
        ),
        frame,
    )
    assert "REGION" not in data.features
    assert "REGION" in data.dropped_constant


def test_target_is_never_a_feature(frame: pd.DataFrame) -> None:
    data = build_dataset(_spec(), frame)
    for blocked in ("TNRY_SEV", "SEVERITY", "NUM_VICTIMS"):
        assert blocked not in data.features


def test_a_specification_selecting_nothing_is_an_error(frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="no records"):
        build_dataset(_spec(data={"years": [1999], "target": "TNRY_SEV"}), frame)


def test_missing_compiled_extract_says_what_to_run(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="tvz data prepare"):
        load_compiled(tmp_path / "absent.parquet")


# The fixture is drawn independently, so a linear model has nothing to converge
# toward; the iteration cap here is set low for speed. Convergence is not what
# this test is about, and warnings are errors everywhere else.
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.parametrize("kind", ["xgboost", "logistic", "decision_tree"])
def test_every_estimator_family_fits(frame: pd.DataFrame, kind: str) -> None:
    params = {"max_iter": 50} if kind == "logistic" else {"max_depth": 2}
    data = build_dataset(_spec(), frame)
    pipeline = build_pipeline(_spec(model={"kind": kind, "params": params}), data.features)
    pipeline.fit(data.X_train, data.y_train)
    assert pipeline.predict_proba(data.X_test).shape == (data.n_test, len(data.classes))


def test_estimators_that_cannot_take_nulls_get_an_imputer(frame: pd.DataFrame) -> None:
    data = build_dataset(_spec(), frame)
    assert supports_missing("xgboost")
    assert not supports_missing("logistic")
    pipeline = build_pipeline(
        _spec(model={"kind": "logistic", "params": {"max_iter": 20}}), data.features
    )
    transformed = pipeline.named_steps["preprocess"].fit_transform(data.X_train)
    assert not np.isnan(np.asarray(transformed, dtype=float)).any()


def test_encoding_width_is_the_same_for_every_fold(frame: pd.DataFrame) -> None:
    """Categories come from the codebook, so a fold cannot shift column positions."""
    spec = _spec()
    data = build_dataset(spec, frame)
    widths = set()
    for train_positions, _ in data.folds(spec):
        preprocessor = build_pipeline(spec, data.features).named_steps["preprocess"]
        widths.add(preprocessor.fit_transform(data.X_train.iloc[train_positions]).shape[1])
    assert len(widths) == 1


def test_training_predicts_every_record_exactly_once(frame: pd.DataFrame) -> None:
    spec = _spec()
    data = build_dataset(spec, frame)
    run = train(spec, data)
    y_true, y_pred, probabilities = run.out_of_fold()

    assert len(run.folds) == spec.split.folds
    assert len(y_true) == data.n_train
    assert len(y_pred) == data.n_train
    assert probabilities.shape == (data.n_train, len(data.classes))


def test_out_of_fold_labels_match_the_training_labels(frame: pd.DataFrame) -> None:
    """Out-of-fold reassembly must restore the original record order."""
    spec = _spec()
    data = build_dataset(spec, frame)
    y_true, _, _ = train(spec, data).out_of_fold()
    np.testing.assert_array_equal(y_true, data.y_train.to_numpy())


def test_probabilities_are_distributions(frame: pd.DataFrame) -> None:
    spec = _spec()
    run = train(spec, build_dataset(spec, frame))
    _, _, probabilities = run.out_of_fold()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


def test_baselines_accompany_every_fold(frame: pd.DataFrame) -> None:
    """A fold score without its baseline invites a wrong intuition about it."""
    spec = _spec()
    run = train(spec, build_dataset(spec, frame))
    for fold in run.folds:
        assert set(fold.baselines) == {"majority", "minority", "prior", "uniform"}
        assert len(fold.baselines["majority"].labels) == len(fold.y_true)


def test_importances_cover_every_feature(frame: pd.DataFrame) -> None:
    """One-hot columns are summed back to their variable, or wide ones look small."""
    spec = _spec()
    data = build_dataset(spec, frame)
    importances = train(spec, data).mean_importances()
    assert set(importances.index) == set(data.features)


def test_resampling_changes_the_fitted_class_balance(frame: pd.DataFrame) -> None:
    spec = _spec(resampling={"max_repeats": 10})
    run = train(spec, build_dataset(spec, frame))
    counts = run.folds[0].train_counts
    assert counts[2] > 0
    plain = train(_spec(), build_dataset(_spec(), frame))
    assert counts[2] > plain.folds[0].train_counts[2]


def test_training_is_reproducible(frame: pd.DataFrame) -> None:
    """Same specification, same seed, same predictions."""
    spec = _spec()
    data = build_dataset(spec, frame)
    first = train(spec, data).out_of_fold()
    second = train(spec, data).out_of_fold()
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_allclose(first[2], second[2])


def test_a_missing_column_names_itself_and_the_remedy(frame: pd.DataFrame) -> None:
    """An extract older than the schema must say so, not raise a bare KeyError."""
    from vision_zero.modeling.dataset import MissingColumnsError

    with pytest.raises(MissingColumnsError, match="tvz data prepare --force"):
        build_dataset(_spec(data={"years": YEARS, "target": "SEV4"}), frame.drop(columns=["SEV4"]))


def test_a_missing_restriction_column_is_caught_too(frame: pd.DataFrame) -> None:
    from vision_zero.modeling.dataset import MissingColumnsError

    spec = _spec(
        data={"years": YEARS, "target": "TNRY_SEV", "restrict": {"REGION": ["Laval (13)"]}}
    )
    with pytest.raises(MissingColumnsError, match="REGION"):
        build_dataset(spec, frame.drop(columns=["REGION"]))
