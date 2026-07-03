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

## M2A full BM25 benchmark

- Command status: `SUCCESS`
- Evidence status: `PARTIAL` because the run correctly recorded `git_dirty: true`
- Run artifact: `artifacts/runs/20260703T231346458330Z-retrieval_full_scientific_bm25-f3a3c0e8`
- Dataset fingerprint: `dda38161938e829f2c2fc9b73d40d6cf922a5470c3b45bf176f742ee0ca7c667`
- Validation-selected fields: title only
- Index: `artifacts/retrieval/dda38161938e829f2c2fc9b73d40d6cf922a5470c3b45bf176f742ee0ca7c667/full_scientific/bm25/shared_index/index`
- Selected candidates: `artifacts/retrieval/dda38161938e829f2c2fc9b73d40d6cf922a5470c3b45bf176f742ee0ca7c667/full_scientific/bm25/title/candidates.parquet`
- Per-query metrics: `artifacts/retrieval/dda38161938e829f2c2fc9b73d40d6cf922a5470c3b45bf176f742ee0ca7c667/full_scientific/bm25/title/per_query_metrics.parquet`
- Failure cases: `artifacts/retrieval/dda38161938e829f2c2fc9b73d40d6cf922a5470c3b45bf176f742ee0ca7c667/full_scientific/bm25/title/failure_cases.json`
- Index build time: 9.605210 seconds
- Index size: 364,014,122 bytes
- Test primary Recall E+S @10/@50/@100/@500: 0.151277 / 0.298436 / 0.368323 / 0.527609
- Test sensitivity Recall E+S+C @10/@50/@100/@500: 0.151877 / 0.299855 / 0.370083 / 0.528654
- Test condensed MRR: 0.848492
- Test condensed NDCG@5/@10: 0.641366 / 0.609680
- Query latency p50/p95: 0.372208 / 2.669017 ms
- Throughput: 1,365.193150 queries/second
- Interpretation: measured on the eligible full dataset, but not a final claim until repeated from a clean M2 commit.

## M2B/M2C blocked evidence

- Dense model: `sentence-transformers/multi-qa-MiniLM-L6-cos-v1`
- Pinned model revision: `b207367332321f8e44f96e224ef15bc607f4dbf0`
- Dense smoke: `BLOCKED` because the required Hugging Face download escalation was rejected after this Codex session reached its approval/usage limit.
- Hybrid smoke/full: `BLOCKED` because no pretrained dense candidate artifact exists.
- Candidate contract and final comparison: `BLOCKED`; no dense or hybrid metrics were fabricated.
- Git commit for current M2 changes: `BLOCKED` by the same approval/usage limit.
