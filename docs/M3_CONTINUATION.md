# M3 Learning-to-Rank — Continuation Handoff

This is a working handoff so another agent can resume M3 exactly where it paused. Read
`AGENTS.md`, `docs/TASKS.md`, `docs/RESULTS.md`, `docs/DATA.md`, `docs/DECISIONS.md` first.
**M3 Learning-to-Rank only. Do NOT begin M4+ (response prediction, simulator, bandits, OPE,
auctions, RL, multi-agent, serving, agentic).**

## Where we are (reconciled 2026-07-04 Pacific)

- ✅ Dense and BM25 three-split retrieval are complete and locally verified. BM25 run
  `...-77793439` is clean (`git_dirty=false`) and validation selected title-only.
- ✅ Weighted fusion and RRF are complete; validation selected alpha 0.5. The M3 contract contains
  25,869,068 unique query-product rows across all 29,844 queries, with zero split overlap and
  correct judged/unjudged null semantics. The first development hybrid run is `...-22416ad8`.
- ✅ M2-vs-M3 metrics, candidate Jaccard, rank correlation, and query outliers are persisted under
  `m3_three_split/handoff_analysis/`. Canonical M2 remains byte-identical (`02bb40b2...`).
- ✅ Label-free Hybrid-top-500 ranking features, heuristic, pointwise, and validation-selected
  LambdaMART runs are complete locally. LambdaMART test NDCG@10 is 0.659345 versus 0.656440 for
  weighted hybrid; full shared-harness and slice analysis exists.
- ✅ The CE-A/CE-B union is complete: 3,156,056 unique pairs = Hybrid top-100 union LambdaMART
  top-50, with zero missing LambdaMART pairs and SHA-256
  `16a43b01f0ba159e5950c1fe7d4363b6c05d7b0c9ffe6c581272379ef9c8488d`.
- ✅ Cross-encoder smoke/resume remains valid; scalable per-block checkpointing is implemented.
- ✅ Canonical A100 run-all notebook is ready at `notebooks/m3_cross_encoder_a100_runall.ipynb`.
- **Durable CE artifact contract (no-rerun downstream):** after the A100 run,
  `finalize_m3_ce_run` writes canonical artifacts under Drive `final/` and `metadata/`:
  `pair_union.parquet`, `scores.parquet`, `scores_enriched.parquet`, `scoring_stats.json`,
  `benchmark.json`, `validation_report.json`, `score_distribution.json`, `provenance.json`,
  `runtime.json`, `artifact_manifest.json`, plus `m3_ce_local_transfer.tar.gz` for local import.
  Transfer the bundle back and run `make import-m3-ce-outputs`, then
  `make rank-m3-ce-evaluate` without rerunning CE scoring.

- ⏳ Full A100 CE union scoring and cascade evaluation remain pending. M3 is therefore not complete.
- ⚠️ Runs created while implementing the new local pipeline record `git_dirty=true`; they are
  development evidence. Clean-provenance reruns are required after committing the implementation.

## Canonical identifiers (do not lose these)

- **Repo:** `github.com/YashDThapliyal/adaptirank` (public). Canonical CE Colab scoring commit =
  **`eb67d8f1d8bbba14a58e9a0a12fd787b5efaa01d`** (`eb67d8f`). Drive-staged CE union/archive artifacts were built at
  **`4f327ff86c5a50b11e850620e8b2f8d74311721c`** (`4f327ff`).
  Commit chain: `21842f8` (M2 complete) → `efe43a5` (M3 three-split config) →
  `1bc808b` (cross-encoder) → `4f327ff` (deterministic ties + artifact base) →
  `eb67d8f` (CE A100 workflow/notebook).
- **Dataset fingerprint (M1.5 canonical):** `dda38161938e829f2c2fc9b73d40d6cf922a5470c3b45bf176f742ee0ca7c667`
  at `artifacts/datasets/esci/processed/<fingerprint>/`.
- **M2 canonical retrieval (NEVER overwrite; artifact_name `full_scientific`):**
  BM25 run `…e8eb8aac`, dense run `…4abe16e3` (Colab), hybrid run `…254c0f0c`.
  M2 candidate contract sha256 `02bb40b2…32ee5`.
- **M3 three-split handoff (artifact_name `m3_three_split`, kept SEPARATE from M2):**
  - Dense run `20260704T043322125059Z-retrieval_m3_three_split_dense-18e175c6`, git `efe43a5`,
    `git_dirty=false`, device `cuda` (A100), fingerprint matches M1.5.
  - Dense `raw_candidates.parquet` sha256 `fd6fe92e1789e26525c85b46561b2d2785d4d73bf4b0ced96d963666c15be97e`;
    14,922,000 rows = 29,844 queries × 500; splits train 18,799 / val 2,089 / test 8,956; 0 NaN.
  - Durable index on Drive `MyDrive/adaptirank/m3_dense_index/` shas:
    embeddings `156da62b…`, faiss.index `934aeefa…`, product_keys `6e2f837b…`, index_metadata `3db6194b…`.
- **Dense encoder:** `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` @ `b207367332321f8e44f96e224ef15bc607f4dbf0`, 384-dim, fields title+description+brand, IVF nlist 1102 nprobe 64, not fine-tuned.
- **Cross-encoder (pinned):** `cross-encoder/ms-marco-MiniLM-L12-v2` @ `7b0235231ca2674cb8ca8f022859a6eba2b1c968`.
  **Substitution note:** the originally requested `ms-marco-MiniLM-L6-v2` emits NaN logits under
  transformers 5.13.0 (checkpoint-specific; L2/L4/L12 work). L12-v2 is its higher-quality sibling,
  documented as an MS MARCO pretrained baseline (not fine-tuned).

## Environment gotchas

- **Do NOT attempt full dense encoding locally.** The local 18 GB machine is memory-bound
  (measured ~50–100 rows/s, swap exhaustion, OOM at index build). Dense runs on Colab CUDA only.
  The durable index is on Drive if ever needed.
- **macOS faiss/torch OpenMP:** `scripts/run_retrieval.py` and `scripts/score_cross_encoder.py`
  set `OMP/OPENBLAS/VECLIB/MKL_NUM_THREADS=1` + `KMP_DUPLICATE_LIB_OK` on darwin before torch/faiss
  import (ADR-007). Keep this.
- **transformers 5.13.0** breaks `ms-marco-MiniLM-L6-v2` (NaN). Use the pinned L12-v2.
- All heavy artifacts are gitignored (`artifacts/**`); they live locally + on Drive. Docs/code are tracked.
- Local files: `~/Documents/m3_dense_candidates.tar.gz` (extracted), `~/Documents/adaptirank_dataset.tar`,
  `~/Documents/adaptirank_m3_colab.ipynb`. `adaptirank_dataset.tar` is also on Drive `MyDrive/adaptirank/`.

## Locked experimental-design constraints (from the user — enforce strictly)

1. **Never overwrite M2 canonical (`full_scientific`) artifacts or results.** The M3 handoff is a
   distinct `m3_three_split` artifact with its own provenance. Preserve both.
2. **Split roles:** train = model fitting; validation = selection/early-stopping ONLY; official test
   = final evaluation ONLY. Never tune on test.
3. **Unjudged handling (locked):** primary LambdaMART + pointwise train on **judged examples only**
   (E/S/C/I grades preserved). NEVER silently map unjudged → grade 0 or ESCI `I`; unjudged stays
   `judgment_status="unjudged"` with null label/grade. Sampled unjudged pseudo-negatives are a
   **separate explicit ablation** only. **At inference/eval, score/rank ALL candidate rows** (judged
   and unjudged). `category` feature is dropped (0% populated in catalog).
4. **Primary LambdaMART uses label-free engineered features ONLY** — no cross-encoder score. The
   cross-encoder is evaluated separately. `LambdaMART + CE-score` is a separate explicit ablation.
5. **Cross-encoder coverage (critical):** the full CE run must score the **deduplicated union** of
   two pair sets, scored ONCE on A100:
   - `Hybrid top-100` per query (standalone baseline: Hybrid→CE)
   - `LambdaMART top-50` per query (cascade: Hybrid top-500 → LambdaMART → top-50 → CE)
   A LambdaMART top-50 item can originate below Hybrid rank 100, so Hybrid-top-100 alone does NOT
   cover the cascade. **Verify the union actually contains every LambdaMART-top-50 pair** before
   claiming cascade coverage. Evaluate the two CE experiments separately from the shared scores.
6. **Latency is hardware-mixed** (BM25/hybrid local CPU; dense Colab CUDA; CE Colab CUDA). Never
   present latency as cross-method comparable without a same-hardware caveat; only quality metrics
   are cross-comparable.

## Immediate next steps (in order)

### Immediate continuation

1. Commit the M3 implementation, then rerun hybrid, handoff analysis, features, learned ranking,
   ranking analysis, and CE union from a clean worktree so promoted run metadata is clean.
2. Package `make rank-m3-ce-package` and run the pinned union scorer on an A100 only after a
   deterministic throughput/memory benchmark. Keep the block checkpoints on Google Drive.
3. Transfer scores back, verify the exact 3,156,056-pair coverage and hashes, then run
   `make rank-m3-ce-evaluate`.
4. Update RESULTS/TASKS only from verified clean runs. Do not mark M3 complete before full CE and
   cascade evidence exist.

### Phase 3 — modeling (local, CPU) then one CE GPU run
5. **Ranking dataset (Part B):** build query-grouped features from the contract. Label-free
   features: bm25/dense/hybrid/rrf score+rank, query length, product-title length, query/title
   lexical overlap, exact-token overlap, brand-match (brand 94% populated). Preserve raw label +
   grade + judgment_status. Document every feature. (No `category` — 0% populated.)
6. **Baselines (Part C):** BM25 / dense / hybrid / RRF ranking, heuristic feature model, pointwise
   (LightGBM/linear). One shared eval harness (reuse `src/adaptirank/retrieval/evaluate.py` metric
   functions: condensed MRR, graded NDCG, recall — see `docs/EXPERIMENTS.md` for exact semantics).
7. **LambdaMART (Part D):** LightGBM `lambdarank`, grouped by query, **train on train**, tune +
   early-stop on **validation only**, freeze config, evaluate on test. Judged-only training targets.
   Record params, training time, inference latency, feature importances. Metrics: NDCG@5/@10, MRR,
   MAP, Recall@K, p50/p95 rerank latency, throughput. Compare vs fixed fusion.
8. **Build CE union pairs (after LambdaMART):** complete. Canonical union has 3,156,056 unique
   pairs, zero duplicate pairs, and zero LambdaMART-top-50 pairs missing from the union.
9. **CE full run (one fresh A100 Colab session):** use
   `notebooks/m3_cross_encoder_a100_runall.ipynb`. It clones commit `eb67d8f` (scoring code; artifacts built at `4f327ff`), installs from the
   lockfile, verifies `m3_ce_a100_input.tar.gz`, extracts the union pairs + dataset from Drive,
   benchmarks throughput on a deterministic validation subset, scores the union with Drive
   checkpoints via `adaptirank.ranking.ce_workflow.score_union_with_checkpoints`, consolidates final
   scores, and writes metadata + SHAs to Drive. Transfer back; re-verify locally.
10. **Cascade (Part F) + analysis (Part G):** evaluate Hybrid-only, Hybrid→LambdaMART,
    Hybrid→CE (depths 20/50/100), Hybrid→LambdaMART→CE. Report `LambdaMART+CE-score` as a separate
    ablation. Produce overall comparison, quality-latency Pareto (hardware-labeled), per-query
    metrics, and slices: query-length, lexical-overlap, BM25-vs-dense disagreement, short-query,
    zero-lexical-overlap, wins & failures. Explicitly probe the four M2 findings (dense<BM25,
    fusion>components, title-only BM25, hard short/zero-overlap queries) — measure, don't assume.
11. **Artifacts (Part H):** ranking dataset fingerprint, feature schema, model configs + checkpoints,
    per-query predictions + metrics, final rankings, latency + analysis reports. Design final
    ranked-candidate artifacts so later stages can consume them without rerunning M3.

## M3 completion gate (mark COMPLETE only if all hold)

Handoff audit passes · simple baselines run · pointwise runs · LambdaMART full eval succeeds ·
CE smoke succeeds · designated CE full eval succeeds · cascade comparison succeeds · validation-only
selection verified · quality-latency report exists · per-query analysis exists · all tests pass ·
worktree clean · **no M4+ work begun**.

## Already-implemented code (M3)

- `src/adaptirank/ranking/cross_encoder.py` — `CrossEncoderScorer` (deterministic, batched,
  CUDA/MPS/CPU, checkpoint/resume), `build_product_text`, `score_top_m`, `scoring_stats`.
- `src/adaptirank/ranking/ce_workflow.py` — canonical CE A100 constants, file/hash validation,
  GPU gate, union/final-score checks, benchmark helper, block manifests, consolidation, and
  `score_union_with_checkpoints`.
- `src/adaptirank/ranking/config.py` — `CrossEncoderConfig`, `CrossEncoderRunConfig`.
- `scripts/score_cross_encoder.py` — runner (reads candidate contract, scores top-M, persists).
- `scripts/validate_m3_ce_notebook.py` — structural validation for the canonical run-all notebook.
- `notebooks/m3_cross_encoder_a100_runall.ipynb` — canonical CE A100 Colab entry point.
- `configs/ranking/cross_encoder_smoke.yaml` (official-sample, CPU, top-20, capped) and
  `cross_encoder_m3.yaml` (m3_three_split, top-100, device auto).
- `tests/unit/test_cross_encoder.py` — 4 tests (fake scorer, no download).
- Makefile: `retrieval-m3-bm25/dense/hybrid`, `rank-smoke-cross-encoder`, `rank-m3-cross-encoder`.

## Useful commands

- `make ci` — lint + typecheck + 46 tests + smoke (all green at pause).
- `make retrieval-m3-bm25` / `retrieval-m3-hybrid` — build the m3 contract (local).
- `make rank-smoke-cross-encoder` — CE CPU smoke (passes).
- Verify things with small polars scripts against `artifacts/retrieval/<fp>/m3_three_split/…`.
