"""Dataset assembly, pipeline construction, baselines, and fitting."""

from __future__ import annotations

from vision_zero.modeling.baselines import Baseline, all_baselines, baseline_predictions
from vision_zero.modeling.config import Experiment, load_experiment
from vision_zero.modeling.dataset import Dataset, build_dataset
from vision_zero.modeling.pipeline import build_estimator, build_pipeline
from vision_zero.modeling.resampling import upsample
from vision_zero.modeling.train import FoldResult, TrainingRun, train

__all__ = [
    "Baseline",
    "Dataset",
    "Experiment",
    "FoldResult",
    "TrainingRun",
    "all_baselines",
    "baseline_predictions",
    "build_dataset",
    "build_estimator",
    "build_pipeline",
    "load_experiment",
    "train",
    "upsample",
]
