UV ?= $(or $(shell command -v uv 2>/dev/null),$(HOME)/.local/bin/uv)
UV_CACHE_DIR ?= .uv-cache
export UV_CACHE_DIR

.PHONY: setup lint typecheck test smoke ci lock

setup:
	$(UV) sync --frozen --dev

lock:
	$(UV) lock

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

typecheck:
	$(UV) run mypy

test:
	$(UV) run pytest -m "not external"

smoke:
	$(UV) run python scripts/smoke_foundation.py

# Deliberately network-free. External ESCI verification is never part of CI.
ci: lint typecheck test smoke
