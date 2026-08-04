# Towards Vision Zero

An analysis of the collision reports published by Quebec's Société de
l'assurance automobile (SAAQ), asking whether the circumstances a police officer
records at a collision say anything useful about how badly it turned out.

The short answer is that they say something real and small, that almost all of
it concerns who was involved rather than the road or the conditions, and that
the limit belongs to the data rather than to the model. The long answer is the
book.

**[Read the book](https://tmfreiberg.github.io/towards-vision-zero/)** ·
**[Command reference](docs/cli.md)**

The book opens with an interactive explorer over all 1,717,407 published
reports. Tick the conditions recorded on a report and it tallies what became of
every collision that matches. It fits nothing and predicts nothing.

## What is here

| | |
|---|---|
| `src/vision_zero/` | The package: data preparation, modelling, and the bounds |
| `book/` | The Quarto book, one document per chapter |
| `conf/experiment/` | Experiment specifications, one YAML file each |
| `data/raw/` | The twelve published extracts, 2011 to 2022 |
| `app/` | The interactive explorer, a single static page |
| `tests/` | Unit, integration and end-to-end tests |

The raw extracts are committed, so a clone carries the data as well as the code.
Nothing needs downloading and no account is needed.

## Installing

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12. `uv` will fetch the
interpreter if it is missing.

```bash
git clone https://github.com/tmfreiberg/towards-vision-zero
cd towards-vision-zero
uv sync
```

That installs from a lockfile, so the dependency versions are the ones the
published figures were produced with.

Then compile the twelve extracts into a single table:

```bash
uv run tvz data prepare
```

This is the only step that reads the raw CSVs. Every later command reads its
output. Check that it worked with:

```bash
uv run tvz info
```

## Trying it

```bash
uv run tvz examples montreal_ternary_complete   # collisions recorded identically, outcomes differing
uv run tvz ceiling montreal_ternary_complete    # how well any method could do
uv run tvz sweep montreal_ternary_complete      # five estimator families under identical conditions
uv run tvz ablate montreal_ternary_complete     # what each group of variables contributes
```

The [command reference](docs/cli.md) covers all of them.

## Building the book

```bash
make render-fresh
```

Requires [Quarto](https://quarto.org/docs/get-started/). Rendered results are
cached under `_freeze/` and that cache is committed, so continuous integration
publishes the book without executing anything.

Use `make render` when only prose has changed and `make render-fresh` whenever
the package has, because Quarto tracks documents and not the modules they
import.

## Development

```bash
make check     # lint, type-check, and run the tests
make test      # tests only
make format    # apply formatting
```

`make help` lists the rest. The same targets work on Windows through `make.bat`.

## Acknowledgement

This began as a team project at the [Erdős Institute](https://www.erdosinstitute.org/)
Data Science Boot Camp in Spring 2024, with Elizabeth Kelly, Jack Wagner and
Supriyo Ghosh. The present repository is a rewrite, and the errors in it are my
own.

## Licence

Code under the MIT licence; see `LICENSE`. The collision data is published by
the SAAQ under the terms stated on
[Données Québec](https://www.donneesquebec.ca/).
