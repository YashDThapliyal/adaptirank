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

The smoke run writes the experiment contract under `artifacts/runs/` and normalized tables under
`artifacts/datasets/esci/processed/<fingerprint>/`.

## ESCI M1 commands

The canonical ranking path matches Amazon Task 1: `small_version == 1` and
`product_locale == "us"`.

```bash
# Tracked fixture: stage, build, validate, and report.
uv run python scripts/download_data.py --config configs/data/esci_fixture.yaml
uv run python scripts/build_catalog.py --config configs/data/esci_fixture.yaml

# Downloads the complete commit-pinned official files (about 1.1 GB), then processes
# 300 source-train and 100 source-test query groups.
make official-sample
```

`official-sample` is integration verification, not a scientific benchmark. These explicit
targets remain `NOT_RUN` until requested:

```bash
make esci-benchmark  # uncapped US Task 1 small-version benchmark
make esci-large      # explicit US large-version extension
```

Raw `E/S/C/I` labels and source split/version flags are preserved. Numeric relevance grades are
derived fields. Products without a judgment for a query are `unjudged`: their label and grade
remain null and must never be converted to `I`/0.

## Architecture status

The package reserves independent namespaces for retrieval, ranking, prediction, simulation,
bandits, OPE, auctions/RL, multi-agent experiments, evaluation, and serving. Those algorithms
are `NOT_RUN` and not implemented in M0–M1.

See `docs/ARCHITECTURE.md`, `docs/DATA.md`, and `docs/TASKS.md` for the operational status.
