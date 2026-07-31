"""The bilingual SAAQ codebook.

The SAAQ publishes its collision extracts with French column names and
integer-coded values, documented in a French-language PDF. This module exposes an
English and French rendering of that documentation: what each variable means,
what each of its values means, and whether the variable has a documented
"unspecified" state.

The content lives in ``codebook/saaq.yaml`` and is loaded into validated models.
Keeping it as data rather than as Python literals means one copy serves the
package, the analysis documents, and the web front end alike, and that it can be
edited by someone who does not read Python.

Accessors return values. Nothing here prints, and nothing blocks for input, so
the codebook can be consulted from a script, a test, or a rendered document as
readily as from a terminal.

Examples
--------
>>> from vision_zero.data import load_codebook
>>> book = load_codebook()
>>> book["SEVERITY"].source_column
'GRAVITE'
>>> book.label("SEVERITY", "Minor", lang="fr")
'Léger'
>>> book["WEATHER"].label_for(11, lang="en")
'Clear'

Whether a blank field is meaningful is a property of the variable:

>>> book["ASPECT"].missing is not None
True
>>> book["MONTH"].missing is not None
False
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "Codebook",
    "Language",
    "Localised",
    "MissingState",
    "Value",
    "Variable",
    "load_codebook",
]

Language = Literal["en", "fr"]
VariableType = Literal["categorical", "ordinal"]

CODEBOOK_DIR = Path(__file__).parent / "codebook"
DEFAULT_CODEBOOK = CODEBOOK_DIR / "saaq.yaml"


class _Frozen(BaseModel):
    """Base for the immutable codebook models."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Localised(_Frozen):
    """A string in each supported language.

    Both languages are required for labels. Glosses may be absent in one
    language where the source documentation gave only one.
    """

    en: str | None = None
    fr: str | None = None

    def get(self, lang: Language) -> str | None:
        """Return the string for ``lang``.

        Parameters
        ----------
        lang
            Either ``"en"`` or ``"fr"``.

        Returns
        -------
        str or None
            The localised string, or ``None`` if the source had none.
        """
        return self.en if lang == "en" else self.fr


class Value(_Frozen):
    """One permitted value of a variable.

    Attributes
    ----------
    label
        The value as it appears in the data, and its French counterpart.
    gloss
        The SAAQ's explanation of what the value means.
    short
        An abbreviated label, for axis ticks and narrow table columns.
    code
        The integer the modelling layer encodes this value as. ``None`` where
        the value is not encoded.
    """

    label: Localised
    gloss: Localised | None = None
    short: Localised | None = None
    code: int | None = None


class MissingState(_Frozen):
    """The documented "unspecified" state of a variable.

    Present only on variables whose published documentation shows a
    *Non précisé* row. Such a row carries a blank *valeur possible*: a blank
    field is the code. It is therefore not a category, receives no integer
    encoding, and appears in the data as a null.

    Its absence is equally informative. ``MONTH`` has no missing state because
    every report carries a date, and ``SEVERITY`` has none because severity is
    what a report exists to record.
    """

    label: Localised


class Variable(_Frozen):
    """One column of the source data.

    Attributes
    ----------
    source_column
        The French column name as published by the SAAQ.
    label
        A short human-readable name, for axis labels and form fields. The
        codebook's ``summary`` says what a variable means at the length the
        documentation states it, which is right for a reference and too long for
        a label.
    type
        Whether the variable is categorical or ordinal. ``None`` for
        identifiers and free numerics.
    summary
        The SAAQ's description of the variable.
    values
        Permitted values, keyed by their string form as they appear in the data.
    missing
        The documented unspecified state, if the variable has one.
    notes
        Caveats about the variable, recorded where the published documentation is
        incomplete or where the data departs from it.
    """

    source_column: str
    label: Localised | None = None
    type: VariableType | None = None
    summary: Localised | None = None
    values: dict[str, Value] = Field(default_factory=dict)
    missing: MissingState | None = None
    notes: Localised | None = None

    @property
    def is_nullable(self) -> bool:
        """Whether the documentation admits an unspecified value."""
        return self.missing is not None

    def name_in(self, lang: Language = "en") -> str:
        """Return the short human-readable name, falling back to the identifier.

        Examples
        --------
        >>> from vision_zero.data import load_codebook
        >>> load_codebook()["PUB_PRIV_RD"].name_in("fr")
        'Catégorie de route'
        """
        if self.label is not None:
            named = self.label.get(lang)
            if named:
                return named
        return ""

    def label_for(self, value: object, lang: Language = "en") -> str | None:
        """Return the label for one value of this variable.

        Values are keyed by their string form, so an integer code and its string
        spelling resolve alike.

        Parameters
        ----------
        value
            A value of this variable. ``None`` resolves to the missing-state
            label where the variable has one.
        lang
            Language of the returned label.

        Returns
        -------
        str or None
            The label, or ``None`` if the value is not documented.

        Examples
        --------
        >>> from vision_zero.data import load_codebook
        >>> book = load_codebook()
        >>> book["WEATHER"].label_for(19, lang="en")
        'Black ice'
        >>> book["WEATHER"].label_for(None, lang="fr")
        'Non précisé'
        """
        if value is None:
            return self.missing.label.get(lang) if self.missing else None
        entry = self.values.get(str(value))
        return entry.label.get(lang) if entry else None

    def short_for(self, value: object, lang: Language = "en") -> str | None:
        """Return the abbreviated label, falling back to the full label."""
        entry = self.values.get(str(value))
        if entry is None:
            return self.label_for(value, lang)
        if entry.short is not None:
            short = entry.short.get(lang)
            if short is not None:
                return short
        return entry.label.get(lang)

    @property
    def codes(self) -> dict[str, int]:
        """Value key to integer encoding, for values that carry one."""
        return {key: v.code for key, v in self.values.items() if v.code is not None}


class Source(_Frozen):
    """Provenance of the documented content."""

    publisher: str
    url: str
    accessed: str


class Codebook(_Frozen):
    """The whole codebook for one dataset."""

    schema_version: int
    dataset: str
    source: Source
    languages: list[Language]
    variables: dict[str, Variable]

    def __getitem__(self, term: str) -> Variable:
        """Return a variable by name, raising ``KeyError`` if unknown."""
        return self.variables[term]

    def __contains__(self, term: object) -> bool:
        """Whether the codebook documents ``term``."""
        return term in self.variables

    def __iter__(self) -> Any:
        """Iterate over variable names."""
        return iter(self.variables)

    def __len__(self) -> int:
        """Return the number of documented variables."""
        return len(self.variables)

    def label(self, term: str, value: object, lang: Language = "en") -> str | None:
        """Return the label of one value of one variable."""
        return self.variables[term].label_for(value, lang)

    def summary(self, term: str, lang: Language = "en") -> str | None:
        """Return the description of a variable."""
        entry = self.variables[term].summary
        return entry.get(lang) if entry else None

    @property
    def nullable_variables(self) -> tuple[str, ...]:
        """Names of variables with a documented unspecified state, in order."""
        return tuple(name for name, v in self.variables.items() if v.is_nullable)

    @property
    def source_columns(self) -> dict[str, str]:
        """Map English variable name to published French column name."""
        return {name: v.source_column for name, v in self.variables.items()}


def _read(path: Path) -> Codebook:
    """Parse and validate a codebook YAML file."""
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return Codebook.model_validate(raw)


@lru_cache(maxsize=4)
def load_codebook(path: Path | None = None) -> Codebook:
    """Load and validate the codebook.

    Cached by path, since the content is immutable and parsing it repeatedly in
    a rendered document is wasted work.

    Parameters
    ----------
    path
        Codebook file. Defaults to the packaged SAAQ codebook.

    Returns
    -------
    Codebook
        The validated codebook.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    pydantic.ValidationError
        If the file does not conform to the schema.

    Examples
    --------
    >>> from vision_zero.data import load_codebook
    >>> len(load_codebook()) > 20
    True
    """
    return _read(path or DEFAULT_CODEBOOK)
