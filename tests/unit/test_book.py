"""Checks on the book sources.

Quarto reports a broken cross-reference or a missing citation as a warning
buried in a long render log, and a stub chapter that silently stopped executing
looks the same as one that never had any code. These catch both from outside.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "_quarto.yml"
BOOK = ROOT / "book"


@pytest.fixture(scope="module")
def config() -> dict[str, Any]:
    parsed: dict[str, Any] = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return parsed


@pytest.fixture(scope="module")
def sources() -> dict[str, str]:
    return {
        path.name: path.read_text(encoding="utf-8")
        for path in [ROOT / "index.qmd", *sorted(BOOK.glob("*.qmd"))]
    }


def test_no_two_chapter_files_share_a_name() -> None:
    """A file copied to a second directory is invisible to every other test here.

    The `sources` fixture is keyed by file name, so that tests can refer to a
    chapter without spelling out its directory. Two files sharing a name
    therefore collapse into one entry, and the one globbed later wins. A stale
    copy of `index.qmd` left in `book/` would be silently checked in place of
    the real one, and every assertion about the chapter would be about the
    wrong file.
    """
    paths = [ROOT / "index.qmd", *sorted(BOOK.glob("*.qmd"))]
    names = [path.name for path in paths]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    assert not duplicates, f"the same file name appears twice: {duplicates}"


def test_every_source_file_is_listed_by_path(config: dict[str, Any]) -> None:
    """Compares whole paths, not names, so a stray copy elsewhere is caught.

    `test_every_chapter_is_listed` compares names and so cannot see a file that
    shares a name with a listed one. This one can.
    """
    listed = {
        Path(entry).as_posix()
        for entry in config["book"]["chapters"] + config["book"].get("appendices", [])
    }
    found = {
        path.relative_to(ROOT).as_posix()
        for path in [ROOT / "index.qmd", *BOOK.glob("*.qmd")]
        if not path.name.startswith("_")
    }
    assert found == listed, f"not listed in _quarto.yml: {sorted(found - listed)}"


def test_every_listed_chapter_exists(config: dict[str, Any]) -> None:
    listed = config["book"]["chapters"] + config["book"].get("appendices", [])
    for entry in listed:
        assert (ROOT / entry).is_file(), entry


def test_every_chapter_is_listed(config: dict[str, Any], sources: dict[str, str]) -> None:
    """An unlisted chapter renders nowhere and is easy not to notice."""
    listed = {
        Path(entry).name
        for entry in config["book"]["chapters"] + config["book"].get("appendices", [])
    }
    present = {name for name in sources if not name.startswith("_")}
    assert present == listed, f"unlisted: {sorted(present - listed)}"


def test_results_are_frozen(config: dict[str, Any]) -> None:
    """Without this, CI would have to execute the analysis and read the data."""
    assert config["execute"]["freeze"] == "auto"


def test_the_explorer_is_published_alongside(config: dict[str, Any]) -> None:
    assert "app/" in config["project"]["resources"]


def test_every_cross_reference_resolves(sources: dict[str, str]) -> None:
    """A broken @sec- reference renders as literal text and reads as a typo."""
    labels = set()
    for text in sources.values():
        labels |= set(re.findall(r"\{#(sec-[a-z0-9-]+)\}", text))

    referenced = set()
    for text in sources.values():
        referenced |= set(re.findall(r"@(sec-[a-z0-9-]+)", text))

    assert referenced <= labels, f"referenced but undefined: {sorted(referenced - labels)}"


def test_every_citation_is_in_the_bibliography(sources: dict[str, str]) -> None:
    bib = (BOOK / "references.bib").read_text(encoding="utf-8")
    keys = set(re.findall(r"^@\w+\{([^,]+),", bib, flags=re.MULTILINE))

    cited: set[str] = set()
    for text in sources.values():
        cited |= set(re.findall(r"@([a-z]+\d{4})", text))

    assert cited <= keys, f"cited but absent from references.bib: {sorted(cited - keys)}"


def test_the_bibliography_has_no_duplicate_keys() -> None:
    bib = (BOOK / "references.bib").read_text(encoding="utf-8")
    keys = re.findall(r"^@\w+\{([^,]+),", bib, flags=re.MULTILINE)
    assert len(keys) == len(set(keys))


def test_chapters_that_include_setup_use_its_names(sources: dict[str, str]) -> None:
    """A chapter using EXPERIMENT or BOOK without the include fails at render."""
    for name, text in sources.items():
        if name.startswith("_"):
            continue
        uses_shared = re.search(r"\b(EXPERIMENT|BOOK)\b", text)
        if uses_shared:
            assert "{{< include _setup.qmd >}}" in text, name


def test_the_landing_page_embeds_the_explorer(sources: dict[str, str]) -> None:
    """It goes first on purpose: a reader who plays finds the limit themselves."""
    index = sources["index.qmd"]
    assert "app/index.html" in index
    assert "explorer-frame" in index


# Chapters still awaiting prose. Remove a name from this list when it is
# written; the test then requires that it no longer claims to be a stub.
UNWRITTEN: set[str] = set()


def test_unwritten_chapters_say_so(sources: dict[str, str]) -> None:
    """Unwritten prose must be visibly unwritten, not quietly absent."""
    for name in UNWRITTEN:
        assert "callout-note" in sources[name], f"{name} is listed as unwritten but has no notice"


def test_written_chapters_do_not_still_claim_to_be_stubs(sources: dict[str, str]) -> None:
    """A stub notice left behind on a finished chapter is worse than none."""
    for name, text in sources.items():
        if name in UNWRITTEN or name.startswith("_"):
            continue
        assert "**Stub." not in text, f"{name} is written but still carries a stub notice"


def test_the_information_appendix_stands_alone(sources: dict[str, str]) -> None:
    """It is pedagogy, and must not depend on the data being present."""
    appendix = sources["a1-information.qmd"]
    assert "```{python}" not in appendix


def test_the_explorer_caption_precedes_the_frame(sources: dict[str, str]) -> None:
    """Scope and the full-screen link belong above an 80vh panel, not below it."""
    index = sources["index.qmd"]
    assert index.index("explorer-actions") < index.index("<iframe")


def test_the_prose_uses_no_em_dashes(sources: dict[str, str]) -> None:
    """A house style rule, enforced rather than remembered.

    Checked only in the book sources. The SAAQ's own French text uses a
    horizontal bar (U+2015) in some place names, which is a different character
    and is left alone.
    """
    offenders = {name: text.count("\u2014") for name, text in sources.items() if "\u2014" in text}
    assert not offenders, f"em dashes found: {offenders}"


def test_no_chapter_declares_a_title_twice(sources: dict[str, str]) -> None:
    """Quarto takes the title from the frontmatter and then numbers the body
    heading separately, so a chapter with both is announced twice and every
    subsequent number runs one ahead."""
    for name, text in sources.items():
        if not text.startswith("---"):
            continue
        frontmatter = text[3 : text.index("---", 3)]
        assert "title:" not in frontmatter, f"{name} sets a title in both places"


def test_the_bibliography_has_somewhere_to_render(config: dict[str, Any]) -> None:
    """Quarto puts the reference list wherever `#refs` appears. Without a page
    holding it, citations link to nothing."""
    listed = config["book"]["chapters"] + config["book"].get("appendices", [])
    assert "book/references.qmd" in listed
    assert "{#refs}" in (BOOK / "references.qmd").read_text(encoding="utf-8")


# Names removed from the package. A chapter still referring to one of these is
# stale, and the only place that shows up otherwise is part-way through a render.
RETIRED_NAMES = ("f_star", "f_beta", "fbeta", "severe_recall", "increments")


def test_no_chapter_refers_to_a_removed_name(sources: dict[str, str]) -> None:
    """Catch a stale chapter here rather than several minutes into a render.

    A book source is not imported by anything, so nothing else notices when it
    falls behind the package. This has now happened often enough during the
    rewrite to be worth a test.
    """
    offenders = {
        name: [term for term in RETIRED_NAMES if term in text]
        for name, text in sources.items()
        if any(term in text for term in RETIRED_NAMES)
    }
    assert not offenders, f"chapters referring to removed names: {offenders}"


def test_the_retired_names_really_are_gone_from_the_package() -> None:
    """Guards the guard. A name still in use would make the test above wrong."""
    package = ROOT / "src" / "vision_zero"
    live = {
        term: sorted(
            path.name for path in package.rglob("*.py") if term in path.read_text(encoding="utf-8")
        )
        for term in RETIRED_NAMES
    }
    still_used = {term: files for term, files in live.items() if files}
    assert not still_used, f"listed as retired but still present: {still_used}"


def test_no_stray_module_shadows_a_package_init() -> None:
    """Catch a file copied in under a flattened name.

    Serving `viz/__init__.py` to a flat directory requires renaming it to
    something like `viz__init__.py`, and a copy that lands without being renamed
    back is a duplicate module inside the package. Nothing imports it, lint and
    type checks pass on it, and no other test notices.
    """
    package = ROOT / "src" / "vision_zero"
    strays = sorted(
        str(path.relative_to(ROOT))
        for path in package.rglob("*__init__.py")
        if path.name != "__init__.py"
    )
    assert not strays, f"modules shadowing a package __init__: {strays}"


def test_every_table_and_figure_cell_ends_in_an_expression(sources: dict[str, str]) -> None:
    """A cell whose last top-level node is a statement renders nothing at all.

    Jupyter displays the value of a cell's final top-level *expression*. An `if`
    statement is not one, so a table built inside either branch is computed and
    thrown away. The cell reports success, the code runs, and the document has a
    caption with nothing under it. Three tables in the conclusion were lost this
    way, and the failure gives no error to search for.

    Assigning to a name and putting that name last is enough to fix it, so this
    checks that every labelled table or figure cell does so.
    """
    offenders = []
    for name, text in sources.items():
        for block in re.findall(r"```\{python\}\n(.*?)```", text, flags=re.DOTALL):
            label = re.search(r"#\|\s*label:\s*(\S+)", block)
            if not label or not label.group(1).startswith(("tbl-", "fig-")):
                continue
            body = "\n".join(line for line in block.splitlines() if not line.startswith("#|"))
            try:
                parsed = ast.parse(body)
            except SyntaxError:  # pragma: no cover - a broken cell fails elsewhere
                continue
            if parsed.body and not isinstance(parsed.body[-1], ast.Expr):
                offenders.append(f"{name}:{label.group(1)}")

    assert not offenders, f"these cells produce no visible output: {offenders}"


def test_the_command_reference_covers_every_command() -> None:
    """A command absent from the reference is one nobody will find.

    The reference lives outside the book, so nothing else notices when a command
    is added and left undocumented.
    """
    reference = ROOT / "docs" / "cli.md"
    if not reference.is_file():  # pragma: no cover - the file is part of the repo
        pytest.skip("no command reference")

    documented = {
        match.split()[0]
        for match in re.findall(
            r"^### `tvz ([a-z ]+)`", reference.read_text(encoding="utf-8"), re.M
        )
    }

    from vision_zero.cli import app

    def name_of(entry: object) -> str:
        """The command's name as typed.

        Typer leaves `name` as a placeholder object when it was never set
        explicitly, so it cannot be tested for truth. A sub-application keeps
        its name a level deeper, on the instance it wraps.
        """
        declared = getattr(entry, "name", None)
        if isinstance(declared, str):
            return declared

        instance = getattr(entry, "typer_instance", None)
        if instance is not None and isinstance(instance.info.name, str):
            return str(instance.info.name)

        callback = getattr(entry, "callback", None)
        return getattr(callback, "__name__", str(entry))

    registered = {name_of(command) for command in app.registered_commands}
    registered |= {name_of(group) for group in app.registered_groups}

    assert not registered - documented, f"undocumented commands: {sorted(registered - documented)}"


def test_the_command_reference_uses_no_quarto_cross_references() -> None:
    """`@sec-` links resolve only inside the book, and render literally elsewhere."""
    reference = ROOT / "docs" / "cli.md"
    if not reference.is_file():  # pragma: no cover
        pytest.skip("no command reference")
    assert "@sec-" not in reference.read_text(encoding="utf-8")
