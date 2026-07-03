"""Fast deterministic M0 smoke command."""

from __future__ import annotations

from adaptirank.common.reproducibility import seed_everything
from adaptirank.common.run import ExperimentRun


def main() -> None:
    config = {"project": "adaptirank", "experiment": "foundation_smoke", "seed": 42}
    with ExperimentRun(
        experiment="foundation_smoke",
        purpose="smoke_test",
        seed=42,
        config=config,
    ) as run:
        seeded = seed_everything(42)
        run.record_metrics({"seeded": seeded, "status": "fixture_only"})
        print(run.run_dir)


if __name__ == "__main__":
    main()
