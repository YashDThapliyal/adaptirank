# Results Ledger

Retrieval performance results: `NOT_RUN`.

Engineering verification entries will be added only after a command succeeds and will point to
its artifact run directory.

## M0 foundation smoke

- Status: `SUCCESS`
- Scope: local engineering verification; not a scientific result
- Artifact: `artifacts/runs/20260703T224453681798Z-foundation_smoke-9a99cd0c`

## M1 fixture verification

- Status: `SUCCESS`
- Scope: deterministic fixture engineering verification; not a scientific result
- Run artifact: `artifacts/runs/20260703T225302166569Z-esci_fixture-26152692`
- Dataset artifact: `artifacts/datasets/esci/processed/b0359f227e93d354ca319be9d10e2b101bbcaa922d39e46793f530c0a4009293`
- Counts: 20 products, 9 query groups, 18 judgments, 2 unjudged background products
- Scientific result eligibility: `false`

## M1 official-source integration verification

- Status: `SUCCESS`
- Scope: sampled processing of fully downloaded, commit-pinned official sources
- Purpose: `integration_verification`; not a scientific benchmark result
- Pinned Amazon commit: `7916cdf6ab75a462e77f20ab40428a10923998d5`
- Run artifact: `artifacts/runs/20260703T225334918794Z-esci_official_sample-79c27c51`
- Dataset artifact: `artifacts/datasets/esci/processed/06ffe5f6c1062a5ae3c0bbe2ba6da6f9610f351e04fb51165e091038adbda166`
- Counts: 18,190 products, 400 query groups, 8,230 judgments, 10,000 unjudged background products
- Splits: 270 train, 30 validation, 100 preserved source-test query groups
- Catalog coverage: 1.0
- Scientific result eligibility: `false`
- Source size: 1,161,827,075 observed bytes across the three official files
- Checksum status: locally observed SHA-256 fingerprints; no independently published expected values

## Explicitly not run

- Explicit US large variant: `NOT_RUN`
- Hosted GitHub Actions: `NOT_RUN`
- E1-E9 scientific experiments: `NOT_RUN`

## M1.5 full canonical ESCI benchmark

- Status: `SUCCESS`
- Scientific eligibility: `true`
- Scope: uncapped `small_version == 1`, `product_locale == "us"`
- Run artifact: `artifacts/runs/20260703T230014349802Z-esci_small_us_benchmark-7795ca57`
- Dataset artifact: `artifacts/datasets/esci/processed/dda38161938e829f2c2fc9b73d40d6cf922a5470c3b45bf176f742ee0ca7c667`
- Dataset fingerprint: `dda38161938e829f2c2fc9b73d40d6cf922a5470c3b45bf176f742ee0ca7c667`
- Counts: 1,215,854 products, 29,844 query groups, 601,354 judgments
- Splits: 18,799 train, 2,089 validation, 8,956 preserved official test query groups
- Query overlap: zero for train/validation, train/test, and validation/test
- Catalog coverage: 1.0
- Source provenance: complete for Amazon commit `7916cdf6ab75a462e77f20ab40428a10923998d5`
- Interpretation: this validates the benchmark dataset contract; it is not a retrieval-performance claim.

## M2A official-sample BM25 verification

- Command status: `SUCCESS`
- Evidence status: integration verification only
- Run artifact: `artifacts/runs/20260703T231218497035Z-retrieval_official_sample_bm25-0f690cbd`
- Dataset fingerprint: `fec41b6515559c1df4db565d1e526175c64a33a1a8b73d5a645c71ed2fd1bd09`
- Selected fields on validation: title + description + brand
- Interpretation: smoke evidence only; these metrics are not promoted to scientific results.

## M2A full BM25 benchmark (canonical, clean provenance)

- Command status: `SUCCESS`
- Evidence status: `CANONICAL` — clean-provenance rerun of `make retrieval-full-bm25`
- Run artifact: `artifacts/runs/20260703T233135889680Z-retrieval_full_scientific_bm25-e8eb8aac`
- Provenance: `git_commit = 9cd251f11bc35bd8fb80e036feacf581af43d203`, `git_dirty = false`,
  `status = SUCCESS`, `seed = 42`, index `cache_reused = false` (index built fresh under this commit)
- Dataset fingerprint: `dda38161938e829f2c2fc9b73d40d6cf922a5470c3b45bf176f742ee0ca7c667` (matches M1.5 canonical)
- Validation-selected fields: title only (title beat title+description and title+description+brand on
  validation `recall_primary_100` then `ndcg_10`)
- Index: `artifacts/retrieval/dda38161938e829f2c2fc9b73d40d6cf922a5470c3b45bf176f742ee0ca7c667/full_scientific/bm25/shared_index/index`
- Selected candidates: `.../full_scientific/bm25/selected_candidates.parquet`
- Per-query metrics: `.../full_scientific/bm25/title/per_query_metrics.parquet`
- Failure cases: `.../full_scientific/bm25/title/failure_cases.json`
- Index engine: tantivy v0.26.0 (index_format v7), BM25 scoring
- Index build time: 7.226101 seconds
- Index size: 377,349,132 bytes
- Document count: 1,215,854
- Test primary Recall E+S @10/@50/@100/@500: 0.151241 / 0.298310 / 0.368280 / 0.527724
- Test sensitivity Recall E+S+C @10/@50/@100/@500: 0.151815 / 0.299758 / 0.370045 / 0.528732
- Test condensed MRR: 0.848895
- Test condensed NDCG@5/@10: 0.641324 / 0.609754
- Query latency p50/p95: 0.455084 / 2.884825 ms
- Throughput: 1,200.394054 queries/second
- Interpretation: measured on the eligible full dataset from a clean commit; this is the promoted
  final BM25 evidence. Latency is wall-clock and not deterministic across machines.

### Clean-vs-dirty comparison

- Prior dirty run: `artifacts/runs/20260703T231346458330Z-retrieval_full_scientific_bm25-f3a3c0e8`
  (`git_commit = 5f0049d`, `git_dirty = true`), now demoted to superseded evidence.
- Selected ablation is identical (title only) in both runs.
- Quality deltas are at the ~1e-4 level, e.g. MRR 0.848895 (clean) vs 0.848492 (dirty),
  Recall E+S @100 0.368280 vs 0.368323, Recall E+S @500 0.527724 vs 0.527609.
- Cause: BM25 score-tie handling is sensitive to tantivy's internal segment layout, which differs
  between a fresh build and a reused index. This is index-build non-determinism among equal-scoring
  documents, not a change in retrieval or metric logic. The magnitude is negligible and does not
  change the title-only selection or any ranking of methods.
- A first clean-commit rerun that reused the dirty index (`aa8c47d3`) reproduced the dirty numbers
  bit-for-bit, confirming the retrieval/eval code is deterministic given a fixed index; the canonical
  run above was then rebuilt from scratch to also give the index clean provenance.

## M2B dense official-sample smoke

- Command status: `SUCCESS`
- Evidence status: integration verification only (`purpose = integration_verification`, 100 queries/split)
- Run artifact: `artifacts/runs/20260703T235155190737Z-retrieval_official_sample_dense-5810ea72`
- Dense model: `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` @ revision `b207367332321f8e44f96e224ef15bc607f4dbf0`, `fine_tuned = false`
- Embedding dimension: 384; device: cpu; fields: title + description + brand
- Index: `IndexIVFFlat` (nlist 128, nprobe 32), 18,190 products; FAISS index and normalized
  embedding memmap persisted and reloadable (`build_seconds = 0` on cache reload)
- Candidate schema: `query_key, product_key, split, method, score, rank`; annotated with
  `esci_label, relevance_grade, judgment_status` (judged 2,389 / unjudged 62,611; unjudged never relabeled)
- MPS encode path verified separately (384-dim, unit-normalized)
- macOS OpenMP segfault in faiss IVF training fixed by ADR-007
- Interpretation: plumbing verified end to end; these metrics are not promoted to scientific results.

## M2C hybrid / candidate contract / final comparison

- Status: pending execution now that clean BM25 and dense-smoke artifacts exist; no dense or
  hybrid metrics are fabricated until the commands succeed.
