"""Building the estimator pipeline.

Encoding is driven by the schema: integer-coded and ordinal variables get an
ordinal encoding with categories taken from the codebook, and unordered
categoricals get one-hot. Taking the category lists from the documentation
rather than from the data matters — a category absent from one fold but present
in another would otherwise shift column positions between folds, and a model
fitted on one would be meaningless applied to the other.

Estimators are named, not imported by path, so a configuration file can select
one without being able to execute arbitrary code.

Examples
--------
>>> from vision_zero.modeling.pipeline import build_estimator
>>> estimator = build_estimator("logistic", {"max_iter": 200})
>>> type(estimator).__name__
'LogisticRegression'
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from vision_zero.data.codebook import load_codebook
from vision_zero.data.schema import dtypes
from vision_zero.logging import get_logger

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

    from vision_zero.modeling.config import EstimatorKind, Experiment

__all__ = [
    "MISSING_SENTINEL",
    "build_estimator",
    "build_pipeline",
    "build_preprocessor",
    "supports_missing",
]

log = get_logger(__name__)

#: Stands in for an unrecorded value once encoded. Distinct from every real
#: category code, so an estimator that can split on it is free to treat
#: "not recorded" as its own condition — which, for variables blank on a fifth
#: of records, is a distinction worth preserving rather than imputing away.
MISSING_SENTINEL = -1

#: Estimators that handle missing values natively and need no imputation.
_NATIVELY_HANDLES_MISSING: frozenset[str] = frozenset({"xgboost", "decision_tree"})


def supports_missing(kind: EstimatorKind | str) -> bool:
    """Whether an estimator family accepts nulls without imputation."""
    return kind in _NATIVELY_HANDLES_MISSING


def _categories_for(feature: str) -> list[Any]:
    """Return a feature's documented values, typed to match the column.

    Taken from the documentation rather than from the data, so that encoded
    column positions are a property of the schema and not of whichever records
    a fold happened to contain. A category present in one fold and absent from
    another would otherwise shift every column after it.

    Codebook keys are strings; a column read as a nullable integer holds
    integers. The values are converted to the column's own type, and sorted —
    scikit-learn's encoders reject unsorted numeric categories.

    Sorting an ordinal integer variable ascending may reverse the direction the
    codes were originally assigned in, which is immaterial: for a tree the
    ordering only has to be consistent, and for a linear model it flips the sign
    of a coefficient and nothing else.
    """
    book = load_codebook()
    keys = list(book[feature].values)
    if dtypes()[feature] == "Int64":
        return sorted(int(key) for key in keys)
    return keys


def build_preprocessor(features: tuple[str, ...], *, impute: bool) -> ColumnTransformer:
    """Assemble the column transformer for a feature set.

    Parameters
    ----------
    features
        Feature column names.
    impute
        Whether to fill nulls before encoding. Required for estimators that
        cannot accept them.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Ordinal encoding for ordered variables, one-hot for the rest.
    """
    book = load_codebook()
    ordinal = [name for name in features if book[name].type == "ordinal"]
    nominal = [name for name in features if book[name].type != "ordinal"]

    ordinal_encoder = OrdinalEncoder(
        categories=[_categories_for(name) for name in ordinal],
        handle_unknown="use_encoded_value",
        unknown_value=np.nan,
        encoded_missing_value=np.nan,
    )
    nominal_encoder = OneHotEncoder(
        categories=[_categories_for(name) for name in nominal],
        handle_unknown="ignore",
        sparse_output=False,
    )

    ordinal_steps: list[tuple[str, Any]] = [("encode", ordinal_encoder)]
    nominal_steps: list[tuple[str, Any]] = [("encode", nominal_encoder)]
    if impute:
        ordinal_steps.append(
            ("impute", SimpleImputer(strategy="constant", fill_value=MISSING_SENTINEL))
        )

    return ColumnTransformer(
        transformers=[
            ("ordinal", Pipeline(ordinal_steps), ordinal),
            ("nominal", Pipeline(nominal_steps), nominal),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_estimator(kind: EstimatorKind | str, params: dict[str, Any]) -> BaseEstimator:
    """Instantiate an estimator by name.

    Parameters
    ----------
    kind
        Estimator family.
    params
        Keyword arguments passed through.

    Returns
    -------
    sklearn.base.BaseEstimator
        The configured estimator.

    Raises
    ------
    ValueError
        If the family is not one this project supports.
    """
    builders: dict[str, Any] = {
        "xgboost": XGBClassifier,
        "logistic": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
        "decision_tree": DecisionTreeClassifier,
        "mlp": MLPClassifier,
        "dummy": DummyClassifier,
    }
    if kind not in builders:
        raise ValueError(f"Unsupported estimator {kind!r}; expected one of {sorted(builders)}")
    return builders[kind](**params)


def build_pipeline(experiment: Experiment, features: tuple[str, ...]) -> Pipeline:
    """Assemble the full preprocessing and estimation pipeline.

    Parameters
    ----------
    experiment
        The validated specification.
    features
        Feature columns actually present, after constant columns were dropped.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Preprocessor followed by estimator.
    """
    kind = experiment.model.kind
    params = dict(experiment.model.params)

    # Seed anything stochastic from the experiment, so that a rerun of the same
    # specification produces the same artifact.
    if kind != "logistic" or params.get("solver") in {"saga", "liblinear"}:
        params.setdefault("random_state", experiment.split.seed)

    impute = not supports_missing(kind)
    if impute:
        log.debug("%s cannot accept nulls; imputing with %d", kind, MISSING_SENTINEL)

    return Pipeline(
        [
            ("preprocess", build_preprocessor(features, impute=impute)),
            ("estimate", build_estimator(kind, params)),
        ]
    )
