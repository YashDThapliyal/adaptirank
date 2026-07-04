# Architectural Decisions

## ADR-001: Canonical project identity

- **Decision:** use `AdaptiRank` for the repository/distribution and `adaptirank` for Python,
  configuration, and run metadata.
- **Reason:** it matches the established checkout and avoids a later namespace migration.
- **Compatibility:** the research story in `AGENTS.md` remains unchanged.

## ADR-002: Local structured run tracking

- **Decision:** use the filesystem artifact contract rather than require paid experiment SaaS.
- **Reason:** local runs remain inspectable and reproducible; an external tracker can be added
  later behind the same interface.

## ADR-003: ESCI identity and missing-judgment semantics

- **Decision:** preserve source composite keys and raw labels; use locale-aware query groups.
- **Decision:** represent missing query-product judgments as `unjudged` with null label/grade.
- **Reason:** broad-catalog retrieval introduces candidates outside the judged pool, and silently
  treating those candidates as irrelevant would corrupt evaluation.

## ADR-004: Sampled official verification still downloads full sources

- **Decision:** `official-sample` caches all three commit-pinned official files before sampling
  query groups for processing.
- **Reason:** this gives exact whole-file observed provenance. It is not advertised as a
  lightweight or byte-range download, and its outputs are not scientific benchmark results.

## ADR-005: Disk-backed BM25 through Tantivy

- **Decision:** use Tantivy's maintained Python bindings for persistent BM25 indexes.
- **Reason:** the uncapped catalog has more than one million US products; a disk-backed,
  memory-mapped search engine is safer than constructing an eager Python sparse matrix in 18 GB
  of local memory. Title, description, and brand remain independently queryable fields.

## ADR-006: Pretrained dense baseline and FAISS

- **Decision:** use `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` pinned to revision
  `b207367332321f8e44f96e224ef15bc607f4dbf0`, without fine-tuning.
- **Decision:** persist normalized embeddings and a FAISS inner-product index.
- **Reason:** the model is explicitly designed for semantic search, has a compact 384-dimensional
  representation, and supports CPU, CUDA, and Apple MPS encoding paths.

## ADR-007: Single-thread the dense build's threading layers on macOS

- **Decision:** on `sys.platform == "darwin"`, before importing torch or faiss, set
  `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=VECLIB_MAXIMUM_THREADS=MKL_NUM_THREADS=1` and
  `KMP_DUPLICATE_LIB_OK=TRUE` (in `scripts/run_retrieval.py`). Also call
  `faiss.omp_set_num_threads(1)` in the dense build as a secondary guard.
- **Reason:** PyTorch and faiss-cpu each bundle their own OpenMP/BLAS runtimes. On macOS, once
  torch's runtime is initialized, faiss IVF k-means training first segfaulted (exit 139); pinning
  only `faiss.omp_set_num_threads(1)` after import removed the segfault but then the BLAS thread
  pool used by k-means **deadlocked** at full-catalog scale (observed: >60 min at ~0% CPU on 1.2M
  vectors with no index written). Constraining every threading layer to one thread *before* torch
  and faiss load prevents the conflicting thread pools from forming: the same 1.2M IVF
  train+add+search then completes in ~4 seconds. Retrieval runs one query per FAISS search call in
  a Python loop, so single-threaded faiss changes neither results nor the per-query latency path.
  The guard is macOS-only so Linux/CUDA hosts, which share one runtime, keep default parallelism.

## ADR-008: Canonical full dense benchmark executed on Colab CUDA

- **Decision:** produce the canonical full dense retrieval artifact on a Google Colab CUDA GPU
  from a clean checkout of the committed code (`21842f8`) using the same processed dataset
  (fingerprint `dda38161...`), rather than locally.
- **Reason:** encoding 1.2M product texts on the local 18 GB machine was memory-bound
  (measured ~50-100 rows/s, swap exhaustion, OOM risk at the ~1.9 GB IVF index build). CUDA is
  faster and has RAM headroom, and Linux shares one OpenMP/BLAS runtime so the macOS faiss/torch
  conflict (ADR-007) does not arise. Provenance is preserved: the run records `git_commit=21842f8`,
  `git_dirty=false`, `dataset_fingerprint=dda38161...`, and `device=cuda`. Artifacts were
  transferred back and every invariant (counts, dim, schema, NaN, fingerprint, provenance) was
  re-verified locally before use.
- **Consequence:** quality metrics (recall/MRR/NDCG) are hardware-independent and directly
  comparable to the local BM25 run. Latency is **not** cross-method comparable (BM25 local CPU vs
  dense Colab T4); RESULTS.md labels latency by hardware and defers a same-hardware latency study.
  The pinned encoder is used without fine-tuning, so device-level float differences do not affect
  the qualitative finding that untuned dense underperforms BM25 while fusion beats both.

## ADR-009: M3 label isolation and primary learned-ranking policy

- **Decision:** fit pointwise and LambdaMART models only on judged training rows. Preserve the
  configured graded targets `E=3`, `S=2`, `C=1`, `I=0`; missing judgments remain null and are
  never converted to `I` or grade 0. Score every candidate at inference time.
- **Decision:** use train for fitting, validation for model/feature selection and early stopping,
  and the official test split only once the selected configuration is frozen. Test labels are
  evaluation inputs only and are never feature values or fit targets.
- **Decision:** the primary LambdaMART model uses label-free retrieval/text features and excludes
  cross-encoder scores. A CE-score feature is permitted only as a separately named ablation.
- **Decision:** omit category features because the canonical catalog has 0% category coverage.
  Brand and title features are permitted because they are observed source fields.
- **Reason:** broad-catalog candidates are mostly unjudged; coercing unknown exposure into a
  negative label would create fabricated supervision. Keeping CE separate also makes the learned
  cascade and pretrained neural reranker independently interpretable.

## ADR-010: Canonical CE A100 Colab notebook

- **Decision:** use `notebooks/m3_cross_encoder_a100_runall.ipynb` as the canonical entry point
  for full M3 cross-encoder union scoring on Colab A100.
- **Decision:** centralize full-run constants, SHA checks, GPU gating, union validation,
  deterministic benchmark selection, checkpoint manifests, consolidation, and final score
  verification in `adaptirank.ranking.ce_workflow`.
- **Decision:** the notebook clones commit `4f327ff86c5a50b11e850620e8b2f8d74311721c`, verifies
  `m3_ce_a100_input.tar.gz` SHA-256
  `a79bb8ad98b2cdbfb56b6f6680c95ce87ef1dd792a16ac91d95fec563ee67f5f`, and refuses to score unless
  the CE union is exactly 3,156,056 rows with SHA-256
  `16a43b01f0ba159e5950c1fe7d4363b6c05d7b0c9ffe6c581272379ef9c8488d`.
- **Reason:** full CE scoring is expensive and hardware-specific. A single run-all notebook with
  reusable library checks keeps the Colab procedure reproducible, resumable, and auditable without
  mixing notebook-only logic into the research code path.
