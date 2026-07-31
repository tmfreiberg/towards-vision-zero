"""Experiment configuration.

An experiment is a file. Which records, which target, which features, which
estimator, which split — all of it lives in YAML under ``conf/`` and is
validated into these models on load. A result can then be traced to the exact
file that produced it, and reproducing it is a matter of passing that file
rather than reconstructing a set of values from context.

Validation happens at load, so a misspelled column name or an out-of-range test
fraction fails immediately with the path to the offending key, rather than
several minutes into a fit with a ``KeyError``.

Examples
--------
>>> from vision_zero.modeling.config import load_experiment  # doctest: +SKIP
>>> experiment = load_experiment("montreal_ternary")  # doctest: +SKIP
>>> experiment.data.target  # doctest: +SKIP
'TNRY_SEV'
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from vision_zero.config import get_settings
from vision_zero.data.codebook import load_codebook
from vision_zero.data.schema import DERIVED, columns, leaks

__all__ = [
    "DataConfig",
    "EvaluationConfig",
    "Experiment",
    "ModelConfig",
    "ResamplingConfig",
    "SplitConfig",
    "load_experiment",
]

EstimatorKind = Literal[
    "xgboost", "logistic", "random_forest", "gradient_boosting", "decision_tree", "mlp"
]

Fraction = Annotated[float, Field(gt=0.0, lt=1.0)]


class _Frozen(BaseModel):
    """Base for the immutable configuration models."""

    model_config = ConfigDict(frozen=True, extra="forbid")


def _known_columns() -> set[str]:
    """Every column that can exist after loading and deriving."""
    return set(columns()) | {derived.name for derived in DERIVED}


class DataConfig(_Frozen):
    """Which records and which columns the experiment uses.

    Attributes
    ----------
    years
        Calendar years to include.
    restrict
        Column to permitted values. A record is kept only if every listed
        column holds one of its listed values.
    target
        The column to be predicted.
    exclude_features
        Columns to withhold beyond those the schema already excludes for
        leakage. ``YEAR`` is the usual entry: the calendar year of a collision
        is not plausibly a cause of its severity, and including it invites the
        model to fit drift in reporting practice.
    drop_rows_missing_target
        Whether to drop records whose target cannot be determined. Default true;
        a record with no label does not belong in a supervised dataset.
    max_missing_features
        Discard records with more than this many blank feature fields. ``None``
        keeps everything.

        Report completeness tracks severity on this data — a collision with
        several fields left blank is very unlikely to have hurt anyone — so a
        model with access to sparse records is partly reading the thoroughness
        of the paperwork. Capping the blanks removes that channel at the cost of
        some records, and the cost is reported when the dataset is built.
    """

    years: list[int] = Field(min_length=1)
    restrict: dict[str, list[str]] = Field(default_factory=dict)
    target: str
    exclude_features: list[str] = Field(default_factory=list)
    drop_rows_missing_target: bool = True
    max_missing_features: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _columns_exist(self) -> DataConfig:
        """Reject column names the schema does not know."""
        known = _known_columns()
        unknown = {
            name
            for name in [self.target, *self.restrict, *self.exclude_features]
            if name not in known
        }
        if unknown:
            raise ValueError(f"Unknown column(s): {sorted(unknown)}")
        return self

    @model_validator(mode="after")
    def _restriction_values_are_documented(self) -> DataConfig:
        """Reject filter values the codebook does not list.

        A typo in a region name would otherwise select nothing and produce an
        empty dataset, which is a confusing way to learn about a typo.
        """
        book = load_codebook()
        for name, values in self.restrict.items():
            if name not in book:
                continue
            documented = set(book[name].values)
            if not documented:
                continue
            unknown = set(values) - documented
            if unknown:
                raise ValueError(f"{name}: undocumented value(s) {sorted(unknown)}")
        return self


class SplitConfig(_Frozen):
    """How records are divided for training, validation, and testing.

    Attributes
    ----------
    test_size
        Fraction held out entirely until final evaluation.
    folds
        Cross-validation folds over the remaining records.
    seed
        Seed for both the split and the fold assignment.
    stratify
        Whether to preserve class proportions. Default true, and it matters
        here: the severe class is under one percent, so an unstratified fold
        can differ materially from another in how many severe records it holds.
    """

    test_size: Fraction = 0.15
    folds: int = Field(default=5, ge=2)
    seed: int = 42
    stratify: bool = True


class ModelConfig(_Frozen):
    """Which estimator, with which hyperparameters.

    Attributes
    ----------
    kind
        Estimator family. Named rather than imported so that a configuration
        file cannot execute arbitrary code.
    params
        Keyword arguments passed through to the estimator.
    """

    kind: EstimatorKind
    params: dict[str, Any] = Field(default_factory=dict)


class ResamplingConfig(_Frozen):
    """Whether and how far to upsample minority classes.

    Attributes
    ----------
    max_repeats
        Upper bound on how many times a minority record may be repeated. One
        means no resampling. Upsampling is applied inside the cross-validation
        loop, to the training fold only — resampling before splitting would
        place copies of the same record on both sides and inflate every score.
    """

    max_repeats: int = Field(default=1, ge=1)


class EvaluationConfig(_Frozen):
    """The scoring rule, class by class.

    Attributes
    ----------
    beta
        Per-class beta for the F-beta score, in class order. Above one weights
        recall, below one weights precision.
    weights
        Per-class weight in the aggregate score.

    Notes
    -----
    The defaults encode an asymmetry in what errors cost. Missing a serious
    collision matters more than raising a false alarm about one, so the severe
    class gets ``beta = 2``. Calling a material-damage collision severe wastes
    attention rather than risking a life, so that class gets ``beta = 0.5``.
    The weights counteract the class imbalance, so that a model cannot score
    well merely by handling the majority class competently.
    """

    beta: list[float] = Field(default_factory=lambda: [0.5, 1.0, 2.0])
    weights: list[float] = Field(default_factory=lambda: [1.0, 4.0, 16.0])

    @model_validator(mode="after")
    def _same_length(self) -> EvaluationConfig:
        """Both lists index the classes, so both must have the same length."""
        if len(self.beta) != len(self.weights):
            raise ValueError(
                f"beta has {len(self.beta)} entries but weights has {len(self.weights)}"
            )
        return self


class Experiment(_Frozen):
    """A complete, reproducible experiment specification."""

    name: str
    description: str = ""
    data: DataConfig
    split: SplitConfig = Field(default_factory=SplitConfig)
    model: ModelConfig
    resampling: ResamplingConfig = Field(default_factory=ResamplingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @property
    def features(self) -> tuple[str, ...]:
        """Columns usable as features, after leakage and explicit exclusions.

        Examples
        --------
        >>> experiment = load_experiment("montreal_ternary")  # doctest: +SKIP
        >>> "NUM_VICTIMS" in experiment.features  # doctest: +SKIP
        False
        """
        from vision_zero.data.schema import feature_candidates

        return feature_candidates(self.data.target, exclude=frozenset(self.data.exclude_features))

    @property
    def excluded_for_leakage(self) -> frozenset[str]:
        """Columns withheld because they determine or are determined by the target."""
        return leaks(self.data.target)


def load_experiment(name_or_path: str | Path, conf_dir: Path | None = None) -> Experiment:
    """Load and validate an experiment specification.

    Parameters
    ----------
    name_or_path
        Either a bare name, resolved as ``conf/experiment/<name>.yaml``, or a
        path to a specific file.
    conf_dir
        Configuration root. Defaults to the configured ``conf`` directory.

    Returns
    -------
    Experiment
        The validated specification.

    Raises
    ------
    FileNotFoundError
        If no such file exists.
    pydantic.ValidationError
        If the file does not conform to the schema.
    """
    candidate = Path(name_or_path)
    if candidate.suffix in {".yaml", ".yml"} and candidate.exists():
        path = candidate
    else:
        root = conf_dir or get_settings().conf_dir
        path = root / "experiment" / f"{candidate.stem}.yaml"

    if not path.is_file():
        raise FileNotFoundError(f"No experiment specification at {path}")

    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return Experiment.model_validate(raw)
