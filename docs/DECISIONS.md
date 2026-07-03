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

