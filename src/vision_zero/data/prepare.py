"""Compiling the yearly extracts into a single columnar file.

Re-parsing the CSVs on every run is slow enough to make analysis unpleasant and
rendering impractical. This module reads them once and writes one Parquet file
with dtypes already correct, which removes an entire class of parsing bug along
with the wait.

The compiled file is a derived artifact and is not committed. What makes it
trustworthy is the manifest written beside it: the SHA-256 of every input, the
row count per year, the schema version, and the package version that produced
it. If an input changes, the manifest no longer matches and the file is known to
be stale rather than merely suspected of it.

Examples
--------
>>> from vision_zero.data.prepare import prepare  # doctest: +SKIP
>>> result = prepare(range(2011, 2023))  # doctest: +SKIP
>>> result.rows  # doctest: +SKIP
1717407
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, Literal

from vision_zero import __version__
from vision_zero.config import get_settings
from vision_zero.data.load import FILENAME_TEMPLATE, add_derived_columns, load_years
from vision_zero.data.schema import DERIVED, columns
from vision_zero.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

__all__ = [
    "Manifest",
    "PreparedData",
    "expected_columns",
    "is_stale",
    "prepare",
    "read_manifest",
]

log = get_logger(__name__)

PARQUET_NAME = "saaq.parquet"
MANIFEST_NAME = "saaq.manifest.json"

#: Bumped whenever the compiled layout changes. Version 2 added the published
#: four-level severity scale as a derived column.
SCHEMA_VERSION = 2

# Dictionary-encoded low-cardinality categoricals compress hard under zstd, which
# is what makes the compiled file an order of magnitude smaller than its inputs.
COMPRESSION: Final[Literal["zstd"]] = "zstd"


@dataclass(frozen=True, slots=True)
class Manifest:
    """Provenance of a compiled dataset.

    Attributes
    ----------
    schema_version
        Version of the column layout, so a stale file with an old layout is
        detectable rather than merely wrong.
    package_version
        Version of this package at the time of compilation.
    created
        UTC timestamp, ISO 8601.
    inputs
        Input filename to SHA-256 digest.
    rows_by_year
        Records read from each year, so a truncated input is visible.
    rows
        Total records written.
    columns
        Column names, in order.
    """

    schema_version: int
    package_version: str
    created: str
    inputs: dict[str, str]
    rows_by_year: dict[str, int]
    rows: int
    columns: list[str]


@dataclass(frozen=True, slots=True)
class PreparedData:
    """Result of a compilation."""

    parquet_path: Path
    manifest_path: Path
    rows: int


def _digest(path: Path) -> str:
    """Return the SHA-256 of a file, read in chunks."""
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def read_manifest(processed_dir: Path | None = None) -> Manifest | None:
    """Read the manifest beside a compiled dataset, if there is one.

    Returns
    -------
    Manifest or None
        The manifest, or ``None`` when no compiled dataset exists.
    """
    directory = processed_dir or get_settings().processed_data_dir
    path = directory / MANIFEST_NAME
    if not path.is_file():
        return None
    return Manifest(**json.loads(path.read_text(encoding="utf-8")))


def expected_columns() -> set[str]:
    """Columns a freshly compiled extract would contain."""
    return set(columns()) | {derived.name for derived in DERIVED}


def is_stale(
    years: Iterable[int],
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
) -> bool:
    """Whether the compiled dataset no longer matches its inputs or the schema.

    True when there is no compiled dataset, when the schema version has moved,
    when the columns a fresh compilation would produce differ from those
    recorded, or when any input's digest differs from the recorded one.

    The column comparison is what catches a schema change that nobody
    remembered to version. Adding a derived column changes what a compiled
    extract should contain, and an extract predating it fails much later with a
    ``KeyError`` naming a column the reader has no reason to recognise.

    Examples
    --------
    >>> is_stale([2011])  # doctest: +SKIP
    True
    """
    manifest = read_manifest(processed_dir)
    if manifest is None or manifest.schema_version != SCHEMA_VERSION:
        return True

    if set(manifest.columns) != expected_columns():
        log.info("Compiled extract predates a change to the column set")
        return True

    directory = raw_dir or get_settings().raw_data_dir
    for year in years:
        name = FILENAME_TEMPLATE.format(year=year)
        path = directory / name
        if not path.is_file() or manifest.inputs.get(name) != _digest(path):
            return True
    return False


def prepare(
    years: Iterable[int],
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
    *,
    strict: bool | None = None,
) -> PreparedData:
    """Compile the yearly extracts into one Parquet file with a manifest.

    Parameters
    ----------
    years
        Calendar years to compile.
    raw_dir
        Directory holding the extracts.
    processed_dir
        Destination. Created if absent.
    strict
        Whether an absent year is an error. Defaults to the configured setting.

    Returns
    -------
    PreparedData
        Paths written and the record count.

    Raises
    ------
    MissingYearsError
        If a requested year is absent and ``strict`` is true.
    """
    settings = get_settings()
    source = raw_dir or settings.raw_data_dir
    destination = processed_dir or settings.processed_data_dir
    destination.mkdir(parents=True, exist_ok=True)

    requested = list(years)
    frame = add_derived_columns(load_years(requested, source, strict=strict))

    parquet_path = destination / PARQUET_NAME
    frame.to_parquet(parquet_path, compression=COMPRESSION, index=False)

    present = [
        year for year in requested if (source / FILENAME_TEMPLATE.format(year=year)).is_file()
    ]
    manifest = Manifest(
        schema_version=SCHEMA_VERSION,
        package_version=__version__,
        created=datetime.now(UTC).isoformat(timespec="seconds"),
        inputs={
            FILENAME_TEMPLATE.format(year=year): _digest(
                source / FILENAME_TEMPLATE.format(year=year)
            )
            for year in present
        },
        rows_by_year={str(year): int((frame["YEAR"] == year).sum()) for year in present},
        rows=len(frame),
        columns=list(frame.columns),
    )
    manifest_path = destination / MANIFEST_NAME
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")

    size_mb = parquet_path.stat().st_size / (1 << 20)
    log.info("Compiled %d records to %s (%.1f MB)", len(frame), parquet_path.name, size_mb)
    return PreparedData(parquet_path=parquet_path, manifest_path=manifest_path, rows=len(frame))
