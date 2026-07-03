UV ?= $(or $(shell command -v uv 2>/dev/null),$(HOME)/.local/bin/uv)
UV_CACHE_DIR ?= .uv-cache
export UV_CACHE_DIR

.PHONY: setup lint typecheck test smoke ci lock official-sample esci-benchmark esci-large \
	retrieval-fixture-bm25 retrieval-smoke-bm25 retrieval-smoke-dense retrieval-smoke-hybrid \
	retrieval-full-bm25 retrieval-full-dense retrieval-full-hybrid

setup:
	$(UV) sync --frozen --dev

lock:
	$(UV) lock

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

typecheck:
	$(UV) run mypy src scripts tests

test:
	$(UV) run pytest -m "not external"

smoke:
	$(UV) run python scripts/run_experiment.py --config configs/data/esci_fixture.yaml

# Deliberately network-free. External ESCI verification is never part of CI.
ci: lint typecheck test smoke

# Downloads all commit-pinned official source files, then processes a deterministic sample.
official-sample:
	$(UV) run python scripts/run_experiment.py --config configs/data/esci_official_sample.yaml

# Explicit scientific/full-data targets. These are never dependencies of setup, smoke, or CI.
esci-benchmark:
	$(UV) run python scripts/run_experiment.py --config configs/data/esci_small_us.yaml

esci-large:
	$(UV) run python scripts/run_experiment.py --config configs/data/esci_large_us.yaml

retrieval-fixture-bm25:
	$(UV) run python scripts/run_retrieval.py --config configs/retrieval/fixture.yaml --method bm25

retrieval-smoke-bm25:
	$(UV) run python scripts/run_retrieval.py --config configs/retrieval/official_sample.yaml --method bm25

retrieval-smoke-dense:
	$(UV) run python scripts/run_retrieval.py --config configs/retrieval/official_sample.yaml --method dense

retrieval-smoke-hybrid:
	$(UV) run python scripts/run_retrieval.py --config configs/retrieval/official_sample.yaml --method hybrid

retrieval-full-bm25:
	$(UV) run python scripts/run_retrieval.py --config configs/retrieval/full.yaml --method bm25

retrieval-full-dense:
	$(UV) run python scripts/run_retrieval.py --config configs/retrieval/full.yaml --method dense

retrieval-full-hybrid:
	$(UV) run python scripts/run_retrieval.py --config configs/retrieval/full.yaml --method hybrid
