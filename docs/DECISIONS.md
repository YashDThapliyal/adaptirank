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
