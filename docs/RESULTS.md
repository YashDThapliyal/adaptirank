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

## M2B full dense benchmark (canonical, clean provenance)

- Command status: `SUCCESS`
- Evidence status: `CANONICAL`
- Execution: run on Google Colab CUDA (T4) because encoding 1.2M products was infeasible on the
  local 18 GB machine (memory-bound: measured ~50-100 rows/s with swap exhaustion and OOM risk at
  the ~1.9 GB FAISS index build). Code was a clean checkout of commit `21842f8` pushed to
  `github.com/YashDThapliyal/adaptirank`; artifacts were transferred back and re-verified locally.
- Run artifact: `artifacts/runs/20260704T022330796938Z-retrieval_full_scientific_dense-4abe16e3`
- Provenance: `git_commit = 21842f8ac4fc379d3dd2a989746249f2adfef0e1`, `git_dirty = false`,
  `status = SUCCESS`, `seed = 42`, `device = cuda`
- Dataset fingerprint: `dda38161938e829f2c2fc9b73d40d6cf922a5470c3b45bf176f742ee0ca7c667` (matches M1.5)
- Encoder: `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` @ `b207367332321f8e44f96e224ef15bc607f4dbf0`,
  `fine_tuned = false`, 384-dim, fields title + description + brand
- Index: `IndexIVFFlat`, nlist 1102, nprobe 64, document_count 1,215,854
- Local re-verification: embeddings/product_keys/catalog counts all equal 1,215,854; candidate
  schema valid; 5,522,500 candidate rows = 11,045 queries x 500; zero NaN/null scores; cosine
  scores in [0.2145, 1.0]; ranks 1-500; every query has exactly 500 candidates. The 1.87 GB
  `product_embeddings.npy` and `faiss.index` remain on Colab/Drive (excluded from transfer);
  their reload and `ntotal` were exercised on Colab by the retrieval that produced the candidates.
- Test primary Recall E+S @10/@50/@100/@500: 0.116494 / 0.247943 / 0.315783 / 0.486097
- Test sensitivity Recall E+S+C @10/@50/@100/@500: 0.114132 / 0.245068 / 0.312472 / 0.481779
- Test condensed MRR: 0.837209; NDCG@5/@10: 0.617297 / 0.579274
- Interpretation: the untuned pretrained dense encoder **underperforms BM25** on this ESCI catalog.
  This is a measured E1 finding, consistent with lexical-heavy product search; it is not a defect.

## M2C hybrid full benchmark (canonical, clean provenance)

- Command status: `SUCCESS`; `git_commit = 21842f8`, `git_dirty = false`, fingerprint `dda38161...`
- Run artifact: `artifacts/runs/20260704T033507773901Z-retrieval_full_scientific_hybrid-254c0f0c`
- Consumes persisted canonical BM25 (`selected_candidates`, title) and canonical dense
  (`raw_candidates`) artifacts on the same fingerprint; keys align by (query_key, product_key).
- Fusion: per-query min-max weighted fusion and reciprocal rank fusion (rrf_k = 60).
- Weighted-fusion alpha selected on **validation only** (objective validation Recall@100 then
  NDCG@10), then frozen before any test evaluation. `score = alpha*bm25_norm + (1-alpha)*dense_norm`
  (alpha=1 is pure BM25, alpha=0 is pure dense).
- Validation alpha sweep (Recall@100 / NDCG@10):
  0.0 = 0.3159 / 0.5754; 0.25 = 0.3645 / 0.6452; **0.5 = 0.4039 / 0.6482 (selected)**;
  0.75 = 0.3920 / 0.6451; 1.0 = 0.3649 / 0.6191. Full sweep recorded in
  `artifacts/retrieval/<fp>/full_scientific/hybrid/tuning.json`.

### E1 final test comparison (frozen alpha = 0.5)

| Method | R@10 E+S | R@50 | R@100 | R@500 | MRR | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 (title) | 0.1512 | 0.2983 | 0.3683 | 0.5277 | 0.8489 | 0.6098 |
| Dense (MiniLM) | 0.1165 | 0.2479 | 0.3158 | 0.4861 | 0.8372 | 0.5793 |
| Weighted hybrid | 0.1559 | 0.3258 | 0.4080 | 0.5896 | 0.8800 | 0.6563 |
| RRF | 0.1472 | 0.3276 | 0.4108 | 0.5884 | 0.8795 | 0.6549 |

- Finding: dense alone < BM25, but **both fusions beat both components** on Recall@50/100/500, MRR,
  and NDCG@10. Weighted and RRF are near-identical (RRF marginally higher Recall@100 0.4108 vs
  0.4080; weighted marginally higher MRR/NDCG). No claim is made that one fusion dominates.
- Latency (**hardware-mixed — not directly comparable**): BM25 p50/p95 = 0.455 / 2.885 ms on local
  CPU; dense = 10.705 / 15.695 ms with query encoding + FAISS search on Colab T4 (CUDA); hybrid
  latency is the conservative sequential sum of both plus fusion (weighted 11.813 / 18.001 ms, RRF
  11.738 / 17.926 ms). Because BM25 and dense were timed on different hardware, only the quality
  metrics above are comparable across methods; a same-hardware latency study is deferred.

### Supporting reports (artifact paths under `artifacts/retrieval/<fp>/full_scientific/`)

- BM25 field ablation (validation): title 0.3645 > title+desc+brand 0.2706 > title+desc 0.2582
  (Recall@100); title-only won — adding description/brand hurt BM25. See `bm25/comparison.json`.
- Query-length slices (weighted hybrid, test Recall@100): long_6+ 0.4569, medium_3-5 0.4214,
  short_1-2 0.3197 — short queries are hardest.
- Lexical-overlap slices (test Recall@100): high 0.4403, low 0.1594, none 0.0311 — queries whose
  relevant items share no title tokens are near-unretrievable by either signal.
- Label-structure slices and per-query metrics: `hybrid/weighted/metrics.json`,
  `hybrid/weighted/per_query_metrics.parquet`.
- Representative failure: query "recently viewed" (navigational, non-product) -> Recall@100 = 0,
  5 relevant missed. See `hybrid/failure_analysis.json` and per-method `failure_cases.json`.

## M2 M3-ready candidate contract

- Artifact: `artifacts/retrieval/dda38161.../full_scientific/hybrid/candidate_contract.parquet`
- Rows: 9,585,620 (union of BM25/dense/hybrid/RRF candidates per query, validation + test)
- Columns: `query_key, product_key, split, bm25_score, bm25_rank, dense_score, dense_rank,
  hybrid_score, hybrid_rank, rrf_score, rrf_rank, esci_label, relevance_grade, judgment_status`
- Judgment status: judged 137,002 / unjudged 9,448,618; every judged row carries `esci_label` and
  `relevance_grade`; unjudged rows are never relabeled. M3 can consume this without rerunning retrieval.

## M3 ranking closeout

- Status: `SUCCESS`; `CANONICAL` clean-provenance M3 CE evaluation and analysis rerun promoted.
- Scope: M3 offline ranking closeout only. No M4 simulator, bandit, OPE, auction, or RL work was started.
- Dataset fingerprint: `dda38161938e829f2c2fc9b73d40d6cf922a5470c3b45bf176f742ee0ca7c667`
- CE evaluation run: `artifacts/runs/20260704T212742425417Z-ranking_m3_ce_evaluate-e495999c` (`SUCCESS`, canonical clean-provenance evidence, `git_commit = c7530b25317708a699c8d38a01ce968f0ddea0b1`, `git_dirty = false`)
- Historical CE evaluation run: `artifacts/runs/20260704T194646524675Z-ranking_m3_ce_evaluate-90a41912` (`SUCCESS`, historical evidence, dirty due notebook state)
- Analysis output: `artifacts/ranking/dda38161938e829f2c2fc9b73d40d6cf922a5470c3b45bf176f742ee0ca7c667/m3_three_split/cross_encoder/analysis/`
- Analysis run metadata: `artifacts/ranking/dda38161938e829f2c2fc9b73d40d6cf922a5470c3b45bf176f742ee0ca7c667/m3_three_split/cross_encoder/analysis/run_metadata.json` (`git_commit = 3900fb45e94838a2339283920d2bc280991cc09f`, `git_dirty = false`)
- CE import facts: 3,156,056 union rows and 3,156,056 score rows; union SHA `16a43b01f0ba159e5950c1fe7d4363b6c05d7b0c9ffe6c581272379ef9c8488d`; scores SHA `923960c5caeef63b33738cb5b4b9ea6cf2163a3a51676359f77dc68a291dd442`; completeness `PASS`.
- CE audit: `PASS`; missing scores 0, invalid scores 0, ordering violations 0, membership violations 0.
- Judgment policy: `UNJUDGED` remains distinct from `IRRELEVANT`; unjudged candidates were not relabeled as negatives.

### Final M3 test comparison

| Method | NDCG@10 | MRR | Recall@10 | Recall@50 | Recall@100 | Recall@500 | Depth |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.609906 | 0.848967 | 0.151161 | 0.298437 | 0.368246 | 0.527779 | 500 |
| Dense | 0.579609 | 0.837741 | 0.116574 | 0.248115 | 0.315969 | 0.486403 | 500 |
| Weighted Hybrid | 0.656273 | 0.880465 | 0.156111 | 0.325851 | 0.408111 | 0.589773 | 500 |
| RRF | 0.654843 | 0.879373 | 0.147545 | 0.328005 | 0.410666 | 0.588507 | 500 |
| Pointwise | 0.658381 | 0.883380 | 0.150300 | 0.291320 | 0.363628 | 0.589773 | 500 |
| LambdaMART | 0.659193 | 0.881553 | 0.157479 | 0.311810 | 0.387248 | 0.589773 | 500 |
| Hybrid->CE | 0.553624 | 0.835182 | 0.188884 | 0.353951 | 0.408111 | 0.408111 | 100 |
| Hybrid->LambdaMART->CE | 0.481169 | 0.795038 | 0.183020 | 0.311810 | 0.311810 | 0.311810 | 50 |

Recall columns for CE cascades are depth-constrained by the reranked candidate set. CE-A reranks Hybrid top 100, and CE-B has only 50 reranked candidates, so CE-B Recall@100 and Recall@500 equal Recall@50 by construction.

Interpretation: LambdaMART is best by test NDCG@10. The pretrained MS MARCO cross-encoder is a negative result on this ESCI cascade, and the audit found no evaluator/import bug explaining the regression.

