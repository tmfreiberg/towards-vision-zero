"""Tests for experiment specification loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from vision_zero.modeling.config import Experiment, load_experiment

MINIMAL = {
    "name": "test",
    "data": {"years": [2011], "target": "TNRY_SEV"},
    "model": {"kind": "xgboost"},
}


def test_shipped_experiments_are_valid() -> None:
    """Every specification in conf/ must load, or CI is lying about the analysis."""
    root = Path(__file__).resolve().parents[2] / "conf" / "experiment"
    specs = sorted(root.glob("*.yaml"))
    assert specs, "no experiment specifications found"
    for path in specs:
        experiment = load_experiment(path)
        assert experiment.name
        assert experiment.features


def test_defaults_are_supplied() -> None:
    experiment = Experiment.model_validate(MINIMAL)
    assert experiment.split.folds == 5
    assert experiment.split.stratify is True
    assert experiment.resampling.max_repeats == 1
    assert experiment.evaluation.focus_class == 2


def test_unknown_target_is_rejected() -> None:
    with pytest.raises(ValidationError, match="Unknown column"):
        Experiment.model_validate({**MINIMAL, "data": {"years": [2011], "target": "NOPE"}})


def test_unknown_excluded_feature_is_rejected() -> None:
    with pytest.raises(ValidationError, match="Unknown column"):
        Experiment.model_validate(
            {
                **MINIMAL,
                "data": {"years": [2011], "target": "TNRY_SEV", "exclude_features": ["NOPE"]},
            }
        )


def test_undocumented_restriction_value_is_rejected() -> None:
    """A typo in a region name would otherwise silently select nothing."""
    with pytest.raises(ValidationError, match="undocumented value"):
        Experiment.model_validate(
            {
                **MINIMAL,
                "data": {
                    "years": [2011],
                    "target": "TNRY_SEV",
                    "restrict": {"REGION": ["Montreal (06)"]},
                },
            }
        )


def test_valid_restriction_value_is_accepted() -> None:
    experiment = Experiment.model_validate(
        {
            **MINIMAL,
            "data": {
                "years": [2011],
                "target": "TNRY_SEV",
                "restrict": {"REGION": ["Montréal (06)"]},
            },
        }
    )
    assert experiment.data.restrict["REGION"] == ["Montréal (06)"]


def test_unknown_estimator_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Experiment.model_validate({**MINIMAL, "model": {"kind": "crystal_ball"}})


def test_test_size_must_be_a_proper_fraction() -> None:
    for bad in (0.0, 1.0, 1.5, -0.1):
        with pytest.raises(ValidationError):
            Experiment.model_validate({**MINIMAL, "split": {"test_size": bad}})


def test_the_focus_class_is_configurable() -> None:
    """Which class a summary singles out is a property of the question asked."""
    experiment = Experiment.model_validate({**MINIMAL, "evaluation": {"focus_class": 1}})
    assert experiment.evaluation.focus_class == 1


def test_no_aggregate_weighting_can_be_configured() -> None:
    """Scores are reported per class and ranked by a metric with no parameters.

    A weighting a reader would have to accept on trust is not something this
    project offers to configure.
    """
    with pytest.raises(ValidationError):
        Experiment.model_validate({**MINIMAL, "evaluation": {"beta": [1.0, 1.0, 1.0]}})
    with pytest.raises(ValidationError):
        Experiment.model_validate({**MINIMAL, "evaluation": {"weights": [1.0, 1.0, 1.0]}})


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Experiment.model_validate({**MINIMAL, "surprise": 1})


def test_features_exclude_leakage_and_the_target() -> None:
    experiment = Experiment.model_validate(MINIMAL)
    for blocked in ("TNRY_SEV", "SEVERITY", "NUM_VICTIMS", "VICTIMS", "ID"):
        assert blocked not in experiment.features
    assert "WEATHER" in experiment.features


def test_derived_columns_are_excluded_when_their_source_is_present() -> None:
    """MULT_VEH is a function of NUM_VEH; offering both is exact collinearity."""
    experiment = Experiment.model_validate(MINIMAL)
    assert "NUM_VEH" in experiment.features
    assert "MULT_VEH" not in experiment.features


def test_specification_is_immutable() -> None:
    experiment = Experiment.model_validate(MINIMAL)
    with pytest.raises(ValidationError):
        experiment.name = "other"  # type: ignore[misc]


def test_missing_specification_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_experiment("no_such_experiment", conf_dir=tmp_path)


def test_a_specification_can_be_loaded_by_path(tmp_path: Path) -> None:
    path = tmp_path / "custom.yaml"
    path.write_text(yaml.dump(MINIMAL), encoding="utf-8")
    assert load_experiment(path).name == "test"
