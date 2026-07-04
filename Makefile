UV ?= $(or $(shell command -v uv 2>/dev/null),$(HOME)/.local/bin/uv)
UV_CACHE_DIR ?= .uv-cache
export UV_CACHE_DIR

.PHONY: setup lint typecheck test smoke ci lock official-sample esci-benchmark esci-large \
	retrieval-fixture-bm25 retrieval-smoke-bm25 retrieval-smoke-dense retrieval-smoke-hybrid \
	retrieval-full-bm25 retrieval-full-dense retrieval-full-hybrid \
	retrieval-m3-bm25 retrieval-m3-dense retrieval-m3-hybrid \
	rank-m3-handoff-analysis rank-m3-features rank-m3-learned rank-m3-analysis rank-m3-ce-union \
	rank-smoke-cross-encoder rank-m3-cross-encoder rank-m3-ce-evaluate rank-m3-ce-package \
	validate-m3-ce-notebook

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

# M3 three-split retrieval handoff (train + validation + test). Separate artifact_name;
# never overwrites the canonical M2 `full_scientific` artifacts. Dense runs on CUDA (Colab).
retrieval-m3-bm25:
	$(UV) run python scripts/run_retrieval.py --config configs/retrieval/m3_three_split.yaml --method bm25

retrieval-m3-dense:
	$(UV) run python scripts/run_retrieval.py --config configs/retrieval/m3_three_split.yaml --method dense

retrieval-m3-hybrid:
	$(UV) run python scripts/run_retrieval.py --config configs/retrieval/m3_three_split.yaml --method hybrid

rank-m3-handoff-analysis:
	$(UV) run python scripts/analyze_m3_handoff.py --config configs/ranking/m3_handoff_analysis.yaml

rank-m3-features:
	$(UV) run python scripts/build_ranking_features.py --config configs/ranking/m3_features.yaml

rank-m3-learned:
	$(UV) run python scripts/train_rankers.py --config configs/ranking/m3_learned.yaml

rank-m3-analysis:
	$(UV) run python scripts/analyze_m3_ranking.py --config configs/ranking/m3_learned.yaml

rank-m3-ce-union:
	$(UV) run python scripts/build_ce_union.py --config configs/ranking/m3_ce_union.yaml

# M3 cross-encoder reranking scorer. Smoke runs on CPU over the official-sample contract;
# the full CUDA run scores the deduplicated Hybrid-top-100/LambdaMART-top-50 union.
rank-smoke-cross-encoder:
	$(UV) run python scripts/score_cross_encoder.py --config configs/ranking/cross_encoder_smoke.yaml

rank-m3-cross-encoder:
	$(UV) run python scripts/score_cross_encoder.py --config configs/ranking/cross_encoder_union_m3.yaml

rank-m3-ce-evaluate:
	$(UV) run python scripts/evaluate_ce_cascades.py --config configs/ranking/m3_ce_evaluate.yaml

rank-m3-ce-package:
	$(UV) run python scripts/package_m3_ce_handoff.py

validate-m3-ce-notebook:
	$(UV) run python scripts/validate_m3_ce_notebook.py
