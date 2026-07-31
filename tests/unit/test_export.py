"""Tests for the browser payload.

The explorer has no server to correct it, so whatever is written here is what a
reader sees. These tests check that the bytes decode back to the records they
came from and that the metadata describes them truthfully — in particular that
the stated base rate is the rate an unfiltered slice actually produces, since
every comparison the page draws is against that number.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vision_zero.export import MISSING_CODE, export_for_browser
from vision_zero.modeling.config import Experiment
from vision_zero.modeling.dataset import Dataset


@pytest.fixture
def dataset() -> tuple[Experiment, Dataset]:
    """A small dataset with a blank, built directly rather than from the fixture."""
    rng = np.random.default_rng(0)
    n = 400
    frame = pd.DataFrame(
        {
            "WEATHER": pd.Series(rng.choice([11, 12, 13], n)).astype("Int64"),
            "ASPECT": pd.Series(rng.choice(["Straight", "Curve"], n)).astype("category"),
        }
    )
    frame.loc[:19, "WEATHER"] = pd.NA
    y = pd.Series(rng.choice([0, 1, 2], n, p=[0.8, 0.19, 0.01]))

    experiment = Experiment.model_validate(
        {
            "name": "tiny",
            "data": {"years": [2011], "target": "TNRY_SEV"},
            "model": {"kind": "decision_tree", "params": {}},
        }
    )
    data = Dataset(
        X_train=frame.iloc[:300].reset_index(drop=True),
        y_train=y.iloc[:300].reset_index(drop=True),
        X_test=frame.iloc[300:].reset_index(drop=True),
        y_test=y.iloc[300:].reset_index(drop=True),
        features=("WEATHER", "ASPECT"),
        target="TNRY_SEV",
        classes=(0, 1, 2),
        dropped_constant=(),
    )
    return experiment, data


def _decode(path: Path, n: int, n_features: int) -> tuple[list[np.ndarray], np.ndarray]:
    raw = np.frombuffer(gzip.decompress(path.read_bytes()), dtype=np.int8)
    columns = [raw[i * n : (i + 1) * n] for i in range(n_features)]
    return columns, raw[n_features * n : (n_features + 1) * n]


def test_payload_decodes_to_the_declared_shape(
    dataset: tuple[Experiment, Dataset], tmp_path: Path
) -> None:
    experiment, data = dataset
    result = export_for_browser(experiment, data, tmp_path)
    raw = np.frombuffer(gzip.decompress(result.data_path.read_bytes()), dtype=np.int8)
    assert len(raw) == (result.n_features + 1) * result.n_records


def test_both_files_are_written(dataset: tuple[Experiment, Dataset], tmp_path: Path) -> None:
    experiment, data = dataset
    result = export_for_browser(experiment, data, tmp_path)
    assert result.data_path.is_file()
    assert result.meta_path.is_file()


def test_the_stated_base_rate_is_what_an_unfiltered_slice_gives(
    dataset: tuple[Experiment, Dataset], tmp_path: Path
) -> None:
    """Every comparison on the page is against this number."""
    experiment, data = dataset
    result = export_for_browser(experiment, data, tmp_path)
    meta = json.loads(result.meta_path.read_text(encoding="utf-8"))

    _, target = _decode(result.data_path, result.n_records, result.n_features)
    observed = np.bincount(target, minlength=3) / len(target)
    np.testing.assert_allclose(observed, meta["target"]["base_rate"], atol=1e-9)


def test_values_decode_back_to_their_labels(
    dataset: tuple[Experiment, Dataset], tmp_path: Path
) -> None:
    experiment, data = dataset
    result = export_for_browser(experiment, data, tmp_path)
    meta = json.loads(result.meta_path.read_text(encoding="utf-8"))
    columns, _ = _decode(result.data_path, result.n_records, result.n_features)

    index = [feature["name"] for feature in meta["features"]].index("ASPECT")
    keys = [value["key"] for value in meta["features"][index]["values"]]
    decoded = [keys[code] for code in columns[index]]

    combined = pd.concat([data.X_train["ASPECT"], data.X_test["ASPECT"]], ignore_index=True)
    assert decoded == list(combined.astype("string"))


def test_blanks_are_preserved_rather_than_dropped(
    dataset: tuple[Experiment, Dataset], tmp_path: Path
) -> None:
    """Whether a field was filled in is part of what the record says."""
    experiment, data = dataset
    result = export_for_browser(experiment, data, tmp_path)
    meta = json.loads(result.meta_path.read_text(encoding="utf-8"))
    columns, _ = _decode(result.data_path, result.n_records, result.n_features)

    index = [feature["name"] for feature in meta["features"]].index("WEATHER")
    assert int((columns[index] == MISSING_CODE).sum()) == 20
    assert meta["missing_code"] == MISSING_CODE


def test_held_out_records_are_included_by_default(
    dataset: tuple[Experiment, Dataset], tmp_path: Path
) -> None:
    """The explorer fits nothing, so there is nothing for them to leak into."""
    experiment, data = dataset
    result = export_for_browser(experiment, data, tmp_path)
    assert result.n_records == data.n_train + data.n_test


def test_held_out_records_can_be_excluded(
    dataset: tuple[Experiment, Dataset], tmp_path: Path
) -> None:
    experiment, data = dataset
    result = export_for_browser(experiment, data, tmp_path, include_test=False)
    assert result.n_records == data.n_train


def test_every_feature_is_documented_in_both_languages(
    dataset: tuple[Experiment, Dataset], tmp_path: Path
) -> None:
    experiment, data = dataset
    result = export_for_browser(experiment, data, tmp_path)
    meta = json.loads(result.meta_path.read_text(encoding="utf-8"))

    for feature in meta["features"]:
        assert feature["source_column"]
        for value in feature["values"]:
            assert value["label"]["en"]
            assert value["label"]["fr"]
        assert feature["missing"]["label"]["en"]
        assert feature["missing"]["label"]["fr"]


def test_metadata_records_which_experiment_produced_it(
    dataset: tuple[Experiment, Dataset], tmp_path: Path
) -> None:
    experiment, data = dataset
    result = export_for_browser(experiment, data, tmp_path)
    meta = json.loads(result.meta_path.read_text(encoding="utf-8"))
    assert meta["experiment"]["name"] == "tiny"
    assert meta["package_version"]


def test_metadata_is_plain_json(dataset: tuple[Experiment, Dataset], tmp_path: Path) -> None:
    """NumPy scalars would serialise fine here and fail on a different dataset."""
    experiment, data = dataset
    result = export_for_browser(experiment, data, tmp_path)
    meta = json.loads(result.meta_path.read_text(encoding="utf-8"))
    for label in meta["target"]["classes"]:
        assert type(label) is int


def test_the_payload_carries_its_own_provenance(
    dataset: tuple[Experiment, Dataset], tmp_path: Path
) -> None:
    """A static page has no server to ask where the data came from."""
    experiment, data = dataset
    result = export_for_browser(experiment, data, tmp_path)
    meta = json.loads(result.meta_path.read_text(encoding="utf-8"))

    for lang in ("en", "fr"):
        assert "donneesquebec.ca" in meta["source"][lang]["citation"]
        assert "SAAQ" in meta["source"][lang]["citation"]
        assert meta["source"][lang]["gloss"]


def test_sorting_preserves_every_record(
    dataset: tuple[Experiment, Dataset], tmp_path: Path
) -> None:
    """Rows are reordered to compress; nothing may be lost or altered by it."""
    experiment, data = dataset
    result = export_for_browser(experiment, data, tmp_path)
    meta = json.loads(result.meta_path.read_text(encoding="utf-8"))
    columns, target = _decode(result.data_path, result.n_records, result.n_features)

    index = [feature["name"] for feature in meta["features"]].index("WEATHER")
    combined = pd.concat([data.X_train["WEATHER"], data.X_test["WEATHER"]], ignore_index=True)
    keys = [value["key"] for value in meta["features"][index]["values"]]

    written = sorted(
        MISSING_CODE if code == MISSING_CODE else int(keys[code]) for code in columns[index]
    )
    original = sorted(MISSING_CODE if pd.isna(value) else int(value) for value in combined)
    assert written == original

    expected = pd.concat([data.y_train, data.y_test], ignore_index=True)
    assert sorted(target.tolist()) == sorted(expected.tolist())


def test_sorting_actually_shrinks_the_payload(
    dataset: tuple[Experiment, Dataset], tmp_path: Path
) -> None:
    """The reason the rows are reordered at all."""
    experiment, data = dataset
    result = export_for_browser(experiment, data, tmp_path)

    raw = gzip.decompress(result.data_path.read_bytes())
    shuffled = np.frombuffer(raw, dtype=np.int8).copy()
    rng = np.random.default_rng(0)
    n = result.n_records
    permutation = rng.permutation(n)
    for column in range(result.n_features + 1):
        block = shuffled[column * n : (column + 1) * n]
        block[:] = block[permutation]

    assert len(gzip.compress(shuffled.tobytes(), 9)) > result.compressed_bytes


def test_a_coarser_scale_is_offered_as_a_partition(
    dataset: tuple[Experiment, Dataset], tmp_path: Path
) -> None:
    """The three-level scale is already declared in the schema, so it is derived.

    Nobody restates which classes combine: `TNRY_SEV` and `SEV4` are both
    defined over `SEVERITY`, and the coarser one is expressible as groups of the
    finer one's classes.
    """
    experiment, data = dataset
    severity = data.__class__(
        X_train=data.X_train,
        y_train=pd.Series([0, 1, 2, 3] * (len(data.y_train) // 4 + 1))[: len(data.y_train)],
        X_test=data.X_test,
        y_test=pd.Series([0, 1, 2, 3] * (len(data.y_test) // 4 + 1))[: len(data.y_test)],
        features=data.features,
        target="SEV4",
        classes=(0, 1, 2, 3),
        dropped_constant=(),
    )
    result = export_for_browser(experiment, severity, tmp_path)
    meta = json.loads(result.meta_path.read_text(encoding="utf-8"))

    groupings = meta["target"]["groupings"]
    assert [len(entry["partition"]) for entry in groupings] == [4, 3]

    published, merged = groupings
    assert published["published"] is True
    assert published["partition"] == [[0], [1], [2], [3]]
    assert merged["published"] is False
    assert merged["partition"] == [[0, 1], [2], [3]]
    assert merged["merged"] == [2, 1, 1]


def test_every_grouping_partitions_the_classes_exactly(
    dataset: tuple[Experiment, Dataset], tmp_path: Path
) -> None:
    """A grouping that dropped or duplicated a class would misstate every share."""
    experiment, data = dataset
    severity = data.__class__(
        X_train=data.X_train,
        y_train=pd.Series([0, 1, 2, 3] * (len(data.y_train) // 4 + 1))[: len(data.y_train)],
        X_test=data.X_test,
        y_test=pd.Series([0, 1, 2, 3] * (len(data.y_test) // 4 + 1))[: len(data.y_test)],
        features=data.features,
        target="SEV4",
        classes=(0, 1, 2, 3),
        dropped_constant=(),
    )
    result = export_for_browser(experiment, severity, tmp_path)
    meta = json.loads(result.meta_path.read_text(encoding="utf-8"))

    n_classes = len(meta["target"]["classes"])
    base = meta["target"]["base_rate"]
    for grouping in meta["target"]["groupings"]:
        covered = [index for group in grouping["partition"] for index in group]
        assert sorted(covered) == list(range(n_classes)), grouping["name"]
        folded = [sum(base[index] for index in group) for group in grouping["partition"]]
        assert sum(folded) == pytest.approx(1.0)
