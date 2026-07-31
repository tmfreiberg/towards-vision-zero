"""Compilation tests, run against the synthetic fixture."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from vision_zero.data.load import MissingYearsError
from vision_zero.data.prepare import (
    MANIFEST_NAME,
    PARQUET_NAME,
    SCHEMA_VERSION,
    is_stale,
    prepare,
    read_manifest,
)

FIXTURE_RAW = Path(__file__).resolve().parents[1] / "fixtures" / "raw"
YEARS = (2011, 2012, 2013)


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[Path, Path]:
    """A writable copy of the fixture, plus an empty output directory."""
    raw = tmp_path / "raw"
    shutil.copytree(FIXTURE_RAW, raw)
    return raw, tmp_path / "processed"


def test_writes_parquet_and_manifest(workspace: tuple[Path, Path]) -> None:
    raw, processed = workspace
    result = prepare(YEARS, raw, processed)

    assert result.parquet_path.name == PARQUET_NAME
    assert result.manifest_path.name == MANIFEST_NAME
    assert result.rows == 1200


def test_parquet_round_trips_with_types_intact(workspace: tuple[Path, Path]) -> None:
    """The point of the compiled file is that dtypes survive the trip."""
    raw, processed = workspace
    result = prepare(YEARS, raw, processed)
    frame = pd.read_parquet(result.parquet_path)

    assert str(frame["WEATHER"].dtype) == "Int64"
    assert str(frame["SEVERITY"].dtype) == "category"
    assert frame["SPD_LIM"].cat.ordered
    assert "TNRY_SEV" in frame.columns


def test_manifest_records_every_input_digest(workspace: tuple[Path, Path]) -> None:
    raw, processed = workspace
    prepare(YEARS, raw, processed)
    manifest = read_manifest(processed)

    assert manifest is not None
    assert set(manifest.inputs) == {f"saaq_{year}.csv" for year in YEARS}
    assert all(len(digest) == 64 for digest in manifest.inputs.values())
    assert manifest.schema_version == SCHEMA_VERSION
    assert manifest.rows == 1200
    assert manifest.rows_by_year == {str(year): 400 for year in YEARS}


def test_a_truncated_input_is_visible_in_the_manifest(workspace: tuple[Path, Path]) -> None:
    raw, processed = workspace
    frame = pd.read_csv(raw / "saaq_2012.csv").head(10)
    frame.to_csv(raw / "saaq_2012.csv", index=False)

    prepare(YEARS, raw, processed)
    manifest = read_manifest(processed)
    assert manifest is not None
    assert manifest.rows_by_year["2012"] == 10


def test_no_manifest_means_stale(workspace: tuple[Path, Path]) -> None:
    raw, processed = workspace
    assert is_stale(YEARS, raw, processed)


def test_fresh_compilation_is_not_stale(workspace: tuple[Path, Path]) -> None:
    raw, processed = workspace
    prepare(YEARS, raw, processed)
    assert not is_stale(YEARS, raw, processed)


def test_a_changed_input_makes_it_stale(workspace: tuple[Path, Path]) -> None:
    """Detected by digest, so an edit that preserves row count still registers."""
    raw, processed = workspace
    prepare(YEARS, raw, processed)
    assert not is_stale(YEARS, raw, processed)

    frame = pd.read_csv(raw / "saaq_2011.csv")
    frame.loc[0, "WEATHER"] = 99
    frame.to_csv(raw / "saaq_2011.csv", index=False)

    assert is_stale(YEARS, raw, processed)


def test_a_removed_input_makes_it_stale(workspace: tuple[Path, Path]) -> None:
    raw, processed = workspace
    prepare(YEARS, raw, processed)
    (raw / "saaq_2013.csv").unlink()
    assert is_stale(YEARS, raw, processed)


def test_an_old_schema_version_makes_it_stale(workspace: tuple[Path, Path]) -> None:
    raw, processed = workspace
    prepare(YEARS, raw, processed)

    path = processed / MANIFEST_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = SCHEMA_VERSION - 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert is_stale(YEARS, raw, processed)


def test_a_missing_year_is_an_error(workspace: tuple[Path, Path]) -> None:
    raw, processed = workspace
    with pytest.raises(MissingYearsError):
        prepare([*YEARS, 2099], raw, processed)


def test_compilation_is_deterministic(workspace: tuple[Path, Path]) -> None:
    raw, processed = workspace
    first = pd.read_parquet(prepare(YEARS, raw, processed).parquet_path)
    second = pd.read_parquet(prepare(YEARS, raw, processed).parquet_path)
    pd.testing.assert_frame_equal(first, second)


def test_a_changed_column_set_makes_it_stale(workspace: tuple[Path, Path]) -> None:
    """Adding a derived column changes what a compiled extract should contain.

    Relying on someone remembering to bump the schema version is not a check.
    An extract predating a new column otherwise fails much later, with a
    `KeyError` naming something the reader has no reason to recognise.
    """
    raw, processed = workspace
    prepare(YEARS, raw, processed)
    assert not is_stale(YEARS, raw, processed)

    path = processed / MANIFEST_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["columns"] = [column for column in payload["columns"] if column != "SEV4"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert is_stale(YEARS, raw, processed)


def test_the_manifest_records_every_expected_column(workspace: tuple[Path, Path]) -> None:
    from vision_zero.data.prepare import expected_columns

    raw, processed = workspace
    prepare(YEARS, raw, processed)
    manifest = read_manifest(processed)
    assert manifest is not None
    assert set(manifest.columns) == expected_columns()
