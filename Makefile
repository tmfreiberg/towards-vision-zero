# Conventional entry points. `make help` lists them.
#
# The one-line onboarding path is:
#     git clone ... && cd towards-vision-zero && make install && make check

.DEFAULT_GOAL := help
.PHONY: help install lock lint format typecheck test test-fast test-data check clean

help:  ## Show this message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Create the environment and install pre-commit hooks
	uv sync --locked
	uv run pre-commit install

lock:  ## Re-resolve dependencies and update uv.lock
	uv lock

lint:  ## Lint without modifying files
	uv run ruff check .
	uv run ruff format --check .

format:  ## Apply lint fixes and formatting
	uv run ruff check --fix .
	uv run ruff format .

typecheck:  ## Run mypy in strict mode
	uv run mypy

test:  ## Run the full suite, enforcing the coverage floor
	uv run pytest --cov-fail-under=80

test-fast:  ## Run the suite, skipping tests marked slow
	uv run pytest -m "not slow" --no-cov

test-data:  ## Run the checks that need the full dataset
	uv run pytest -m requires_data --no-cov

check: lint typecheck test  ## Everything CI runs

clean:  ## Remove caches and build output
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov dist build
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
