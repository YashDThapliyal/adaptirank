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
