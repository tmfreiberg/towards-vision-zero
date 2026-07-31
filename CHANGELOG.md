# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning is [semantic](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Complete rebuild of the 2024 codebase. See `docs/adr/` for the reasoning and
`docs/defects.md` for the audit that motivated it.

### Added
- Architecture decision records (ADR-0001 through ADR-0008).
- Audit of the 2024 codebase documenting eighteen defects.
- Overhaul roadmap.
- `pyproject.toml` with a `src/` layout, Hatchling backend, and a committed
  `uv.lock`.
- `ruff`, `mypy` (strict), and `pytest` configuration.
- `pre-commit` hooks, including `nbstripout`.
- GitHub Actions CI: lint, type check, and test on Python 3.12 and 3.13.
- `vision_zero.config` — validated runtime settings via `pydantic-settings`.
- `vision_zero.logging` — structured logging, replacing library-level `print`.
- `towards-vision-zero` CLI entry point.

### Changed
- Project reframed around the predictability ceiling rather than around
  building a severity classifier (ADR-0003).
- Bilingual codebook moved from Python literals to YAML (ADR-0006).

### Removed
- NCDB dataset support (ADR-0007).
- `scripts/setup.py` import bootstrapping, replaced by package installation.
- The Streamlit application (ADR-0006).

## [0.1.0-bootcamp] — 2024-10-27

State of the repository at the end of the post-bootcamp revision work. Tagged
for reference; not maintained.
