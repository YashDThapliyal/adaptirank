# AdaptiRank

AdaptiRank is a research-engineering project for studying the separate roles of offline
ranking, online exploration, off-policy evaluation, and long-horizon budget optimization in
sponsored-product systems. The current implementation is intentionally limited to M0–M1:
reproducible infrastructure and the ESCI data pipeline.

## Quickstart

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
make setup
make ci
```

`make ci` is network-free. It runs linting, static typing, tests, and a deterministic fixture
smoke run. External data is never downloaded by CI.

## Architecture status

The package reserves independent namespaces for retrieval, ranking, prediction, simulation,
bandits, OPE, auctions/RL, multi-agent experiments, evaluation, and serving. Those algorithms
are `NOT_RUN` and not implemented in M0–M1.

See `docs/ARCHITECTURE.md`, `docs/DATA.md`, and `docs/TASKS.md` for the operational status.

