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
