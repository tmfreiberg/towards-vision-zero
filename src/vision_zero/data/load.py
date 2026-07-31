"""Reading the SAAQ collision extracts.

One CSV per year sits under ``data/raw``. This module turns them into a single
typed frame, and does so loudly: a year that cannot be read is an error, not a
gap. Partial ingestion is invisible in every number computed downstream, and an
analysis whose conclusions concern the data cannot afford to lose a year of it
without saying so.

Every function here is pure with respect to the filesystem — it reads, it
returns, it logs. Nothing prints, nothing prompts, nothing mutates global state.

Examples
--------
>>> from vision_zero.data.load import load_years  # doctest: +SKIP
>>> frame = load_years(range(2011, 2023))  # doctest: +SKIP
>>> frame["YEAR"].nunique()  # doctest: +SKIP
12
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from vision_zero.config import get_settings
from vision_zero.data.schema import (
    DERIVED,
    UNDOCUMENTED_VALUES,
    columns,
    dtypes,
    ordered_categories,
)
from vision_zero.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

__all__ = [
    "FILENAME_TEMPLATE",
    "MissingYearsError",
    "SchemaMismatchError",
    "add_derived_columns",
    "load_year",
    "load_years",
    "read_raw_csv",
]

log = get_logger(__name__)

#: Filename pattern for the yearly extracts.
FILENAME_TEMPLATE = "saaq_{year}.csv"


class SchemaMismatchError(ValueError):
    """A file's columns do not match the documented schema."""


class MissingYearsError(FileNotFoundError):
    """One or more requested years are absent from the raw data directory."""


def _path_for(year: int, raw_dir: Path | None = None) -> Path:
    """Return the expected path of one year's extract."""
    directory = raw_dir or get_settings().raw_data_dir
    return directory / FILENAME_TEMPLATE.format(year=year)


def read_raw_csv(path: Path) -> pd.DataFrame:
    """Read one extract with no typing or validation applied.

    Separated from :func:`load_year` so that a file can be inspected before its
    contents are asserted to match the schema — useful when a new year is
    published and its columns may have shifted.

    Parameters
    ----------
    path
        The CSV to read.

    Returns
    -------
    pandas.DataFrame
        The file's contents, all columns as read by pandas.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"No extract at {path}")
    return pd.read_csv(path, low_memory=False)


def _validate_columns(frame: pd.DataFrame, path: Path) -> None:
    """Check a frame's columns against the documented schema.

    Extra columns are tolerated and logged; missing ones are fatal. A column
    that vanishes silently would be read as all-null everywhere it is used.
    """
    expected = set(columns())
    observed = set(frame.columns)

    missing = expected - observed
    if missing:
        raise SchemaMismatchError(f"{path.name} is missing documented columns: {sorted(missing)}")

    extra = observed - expected
    if extra:
        log.warning("%s carries undocumented columns: %s", path.name, sorted(extra))


def _drop_undocumented_values(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Null out values the documentation does not admit.

    Assigning a meaning to an undocumented code would be inventing
    documentation; leaving it in place would let it be encoded as though it
    meant something. Nulling it says what is actually known: the value was not
    recorded in a form the documentation covers.

    The count is logged at info level, because a coercion that grows from four
    records to four thousand is a change in the data worth noticing.

    Applied *after* typing. A column read straight from CSV with blanks in it
    arrives as float, so an undocumented ``0`` renders as ``"0.0"`` and would
    not match. Once the column is a nullable integer the comparison is exact.

    Ordinal columns held as ordered categories need no help here: their
    categories come from the codebook, so an undocumented value is already null
    by the time this runs.
    """
    cleaned = frame.copy()
    for name, values in UNDOCUMENTED_VALUES.items():
        if name not in cleaned.columns:
            continue
        mask = cleaned[name].astype("string").isin(values).fillna(value=False)
        count = int(mask.sum())
        if not count:
            continue
        log.info(
            "%s: %s carries %d undocumented value(s) %s, coerced to null",
            path.name,
            name,
            count,
            sorted(values),
        )
        cleaned.loc[mask, name] = None
        if isinstance(cleaned[name].dtype, pd.CategoricalDtype):
            cleaned[name] = cleaned[name].cat.remove_unused_categories()
    return cleaned


def _apply_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to their documented types.

    Integer-coded columns are coerced with ``errors="coerce"``, so a value the
    documentation does not admit becomes null rather than raising. That is the
    right behaviour here: such values do occur, and the count of them is itself
    worth reporting, which the caller can do from the resulting nulls.
    """
    typed = frame.copy()
    schema = dtypes()
    orders = ordered_categories()

    for name, dtype in schema.items():
        if name not in typed.columns:
            continue
        if dtype == "Int64":
            coerced = pd.to_numeric(typed[name], errors="coerce")
            lost = int(coerced.isna().sum() - typed[name].isna().sum())
            if lost > 0:
                log.warning("%s: %d value(s) were not numeric and became null", name, lost)
            typed[name] = coerced.astype("Int64")
        elif name in orders:
            typed[name] = pd.Categorical(
                typed[name].astype("string"), categories=list(orders[name]), ordered=True
            )
        elif dtype == "string":
            typed[name] = typed[name].astype("string")
        else:
            typed[name] = typed[name].astype("category")

    return typed


def load_year(year: int, raw_dir: Path | None = None) -> pd.DataFrame:
    """Read and type one year's extract.

    Parameters
    ----------
    year
        Calendar year of the extract.
    raw_dir
        Directory holding the extracts. Defaults to the configured raw data
        directory.

    Returns
    -------
    pandas.DataFrame
        The year's records, typed per the schema.

    Raises
    ------
    FileNotFoundError
        If the extract is absent.
    SchemaMismatchError
        If documented columns are missing from the file.
    """
    path = _path_for(year, raw_dir)
    frame = read_raw_csv(path)
    _validate_columns(frame, path)
    typed = _drop_undocumented_values(_apply_dtypes(frame), path)
    log.debug("%s: %d records", path.name, len(typed))
    return typed


def load_years(
    years: Iterable[int],
    raw_dir: Path | None = None,
    *,
    strict: bool | None = None,
) -> pd.DataFrame:
    """Read several years and concatenate them.

    Parameters
    ----------
    years
        Calendar years to read.
    raw_dir
        Directory holding the extracts.
    strict
        Whether an absent year is an error. Defaults to the configured
        ``strict_data`` setting, which is true.

        Under ``strict=False`` the years actually read are logged at warning
        level and recorded in the frame, so a partial load is always visible in
        the result rather than only in the console.

    Returns
    -------
    pandas.DataFrame
        All records, in year order, with a fresh index.

    Raises
    ------
    MissingYearsError
        If any requested year is absent and ``strict`` is true.
    ValueError
        If no year could be read at all.

    Examples
    --------
    >>> load_years([2011, 2012], strict=False)  # doctest: +SKIP
    """
    requested = list(years)
    if strict is None:
        strict = get_settings().strict_data

    available = [year for year in requested if _path_for(year, raw_dir).is_file()]
    absent = [year for year in requested if year not in available]

    if absent:
        message = f"No extract for {absent} in {raw_dir or get_settings().raw_data_dir}"
        if strict:
            raise MissingYearsError(message)
        log.warning("%s — continuing with %s", message, available)

    if not available:
        raise ValueError(f"None of the requested years {requested} could be read")

    frames = [load_year(year, raw_dir) for year in sorted(available)]
    combined = pd.concat(frames, ignore_index=True)
    log.info("Read %d records across %d year(s)", len(combined), len(frames))
    return combined


def add_derived_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Insert the derived targets beside the columns they come from.

    A derived value that cannot be determined is left null. Dropping such rows
    is the caller's decision, not this function's, because whether an
    undetermined target matters depends on which target is being modelled.

    Parameters
    ----------
    frame
        A typed frame of published columns.

    Returns
    -------
    pandas.DataFrame
        A copy with the derived columns added.

    Examples
    --------
    >>> import pandas as pd
    >>> frame = pd.DataFrame({"SEVERITY": ["Minor", "Fatal or serious"]})
    >>> add_derived_columns(frame)["TNRY_SEV"].tolist()
    [1, 2]
    """
    out = frame.copy()
    for derived in DERIVED:
        if derived.source not in out.columns:
            continue
        values = derived.apply(out[derived.source]).astype("Int64")
        position = list(out.columns).index(derived.source)
        out.insert(position + 1, derived.name, values)
        undetermined = int(values.isna().sum())
        if undetermined:
            log.debug("%s: %d record(s) with no determinable value", derived.name, undetermined)
    return out
