PY := uv run
PKG := src/uplift

.PHONY: setup data test lint typecheck report ci clean

## setup: create the venv and install the package (editable) with dev tools
setup:
	uv sync

## data: fetch Hillstrom once, write manifest and deterministic splits
data:
	$(PY) python -m uplift.data.loader --fetch

## test: unit + property + integration tests with coverage gate
test:
	$(PY) pytest

## lint: ruff
lint:
	$(PY) ruff check .

## typecheck: mypy strict
typecheck:
	$(PY) mypy

## report: end-to-end pipeline -> reports/metrics.json, figures and decision memo
report:
	$(PY) python -m uplift.report --split test

## ci: everything that must pass before a commit
ci: lint typecheck test

## clean: remove caches and generated artefacts (keeps fetched raw data)
clean:
	$(PY) python -m uplift.data.clean
