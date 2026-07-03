# Architecture

The canonical project name is **AdaptiRank** and the Python namespace is `adaptirank`.

M0–M1 implement shared configuration, provenance, run artifacts, canonical domain types, and
ESCI ingestion. Later components remain independently replaceable namespaces: retrieval,
ranking, response prediction, simulator, bandit policy, logging policy, OPE estimator, auction
mechanism, RL policy, multi-agent environment, evaluation, and serving.

No later-stage algorithm is implemented in the current milestone.

