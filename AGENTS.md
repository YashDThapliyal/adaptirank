# AGENTS.md

# AdaptiveAds
## A Multi-Stage Ads System with Offline Ranking, Online Bandit Learning, Off-Policy Evaluation, and RL Budget Optimization

This repository is a research-engineering project for building and evaluating a realistic, modular ads decision stack.

The project must combine four distinct forms of decision-making:

1. **Offline retrieval and ranking** to estimate which ads/products are relevant.
2. **Online contextual bandit learning** to balance exploration and exploitation from interaction feedback.
3. **Off-policy evaluation (OPE)** to estimate new policy quality from logged feedback without deploying every policy online.
4. **Reinforcement learning for bidding and budget pacing** to optimize long-horizon advertiser value under finite budgets and repeated auctions.

The project is intentionally ambitious. Do not reduce it to a toy recommender notebook, a single CTR model, or an RL demo. The goal is a coherent end-to-end system with real public datasets where appropriate, simulation where public data is structurally unavailable, strong baselines, reproducible experiments, and honest separation between observed data and simulated assumptions.

---

# 1. Core Research Question

The central question is:

> **How should an ads system combine offline relevance models, online exploration, counterfactual policy evaluation, and long-horizon budget optimization under biased feedback, non-stationary user preferences, and finite advertiser budgets?**

The project should make the following distinctions explicit:

- A **ranker predicts** which candidates are likely to be relevant or clicked.
- A **bandit explores** uncertain actions while learning from online feedback.
- An **OPE estimator evaluates** candidate policies from logged data.
- An **RL policy optimizes sequential decisions** whose current actions affect future opportunity and budget availability.

These are not interchangeable.

---

# 2. Product Story

The canonical scenario is sponsored-product search.

A user issues a query such as:

> `waterproof trail running shoes`

The system must:

1. retrieve plausible sponsored products from a large catalog,
2. rank them for relevance,
3. estimate immediate response probabilities,
4. choose what to expose while accounting for uncertainty,
5. learn from clicks and conversions,
6. correct for biased exposure and position effects,
7. participate in repeated auctions,
8. manage advertiser budget across the day,
9. adapt when user preferences or competition shift.

The system should support both single-agent and multi-agent auction experiments.

---

# 3. Non-Negotiable Design Principles

## 3.1 Real data where it exists; simulation where it does not

Do not invent synthetic relevance labels if a real public retrieval dataset can be used.

Do not pretend a public dataset contains auction budgets, bids, or long-horizon counterfactual outcomes if it does not.

The canonical data strategy is:

- **Amazon Shopping Queries / ESCI** for query-product semantic matching and graded relevance.
- **Open Bandit Dataset / Open Bandit Pipeline** for logged bandit feedback and real-world OPE validation.
- **Criteo click logs** as an auxiliary CTR-prediction benchmark, kept separate from the semantic ranking backbone.
- **KuaiRand or KuaiRec** as an optional sequential/exposure-bias validation track.
- **A controlled simulator** for repeated auctions, advertiser budgets, pacing, non-stationarity, counterfactual ground truth, and RL.

Never merge unrelated datasets into a fake unified production log.

## 3.2 No fake results

Never fabricate metrics, model improvements, latency numbers, dataset sizes, or experiment outcomes.

If an experiment has not run, mark it as:

- `NOT_RUN`
- `BLOCKED`
- `PARTIAL`

Do not write plausible-looking placeholder values into result tables.

## 3.3 Baselines before complexity

Every advanced model must be compared against meaningful simpler baselines.

Examples:

- Dense retrieval must be compared against BM25.
- Hybrid retrieval must be compared against both lexical and dense components.
- Neural ranking must be compared against heuristic and tree-based LTR baselines.
- Contextual bandits must be compared against random, greedy, and epsilon-greedy policies.
- RL pacing must be compared against fixed bids, uniform pacing, greedy pacing, and PID/control baselines.
- Multi-agent learning must be compared against fixed-strategy competitors.

## 3.4 Reproducibility is a first-class feature

Every experiment must record:

- config,
- git commit,
- random seed,
- dataset version or fingerprint,
- dependency environment,
- start/end timestamp,
- metrics,
- artifact locations.

## 3.5 Simulation assumptions must be inspectable

The simulator must expose configurable assumptions rather than hide them in code.

Examples:

- position examination probabilities,
- click model coefficients,
- conversion model,
- user preference vectors,
- preference drift schedule,
- advertiser budgets,
- auction mechanism,
- bid distributions,
- reward coefficients,
- pacing penalties.

## 3.6 Do not use LLM APIs inside the training or simulation loop

This project is about ranking, bandits, OPE, and RL.

LLMs may be used for development assistance or optional offline qualitative analysis, but not as an expensive hidden component inside millions of environment steps.

---

# 4. Canonical Architecture

```text
                         USER QUERY + CONTEXT
                                  |
                                  v
                    +---------------------------+
                    | Candidate Retrieval       |
                    | BM25 / Dense / Hybrid     |
                    +-------------+-------------+
                                  |
                               top N
                                  |
                                  v
                    +---------------------------+
                    | Lightweight LTR Ranker    |
                    | LambdaMART / GBDT         |
                    +-------------+-------------+
                                  |
                               top M
                                  |
                                  v
                    +---------------------------+
                    | Heavy Neural Reranker     |
                    | Cross-Encoder             |
                    +-------------+-------------+
                                  |
                               top K
                                  |
                                  v
                    +---------------------------+
                    | Response Prediction       |
                    | pCTR / pCVR + Calibration |
                    +-------------+-------------+
                                  |
                                  v
                    +---------------------------+
                    | Online Bandit Layer       |
                    | Explore vs Exploit        |
                    +-------------+-------------+
                                  |
                           chosen exposure
                                  |
                                  v
                    +---------------------------+
                    | Auction / Eligibility     |
                    | bids, quality, budget     |
                    +-------------+-------------+
                                  |
                                  v
                    +---------------------------+
                    | RL Bidding & Pacing       |
                    | long-horizon optimization |
                    +-------------+-------------+
                                  |
                                  v
                         USER INTERACTION
                       click / convert / ignore
                                  |
                   +--------------+--------------+
                   |                             |
                   v                             v
          BANDIT UPDATE / LOGGING        RL TRAJECTORY UPDATE
```

The implementation must keep these modules separable. A future experiment must be able to replace one stage without rewriting the rest.

---

# 5. Repository Structure

Use the following target structure unless a clearly better structure is justified in `docs/DECISIONS.md`.

```text
adaptive-ads/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── uv.lock or equivalent lockfile
├── Makefile
├── .env.example
├── .gitignore
│
├── configs/
│   ├── data/
│   ├── retrieval/
│   ├── ranking/
│   ├── prediction/
│   ├── simulator/
│   ├── bandits/
│   ├── ope/
│   ├── rl/
│   └── experiments/
│
├── src/adaptive_ads/
│   ├── common/
│   │   ├── config.py
│   │   ├── logging.py
│   │   ├── paths.py
│   │   ├── reproducibility.py
│   │   └── types.py
│   │
│   ├── data/
│   │   ├── esci.py
│   │   ├── open_bandit.py
│   │   ├── criteo.py
│   │   ├── kuairand.py
│   │   ├── schemas.py
│   │   └── splits.py
│   │
│   ├── retrieval/
│   │   ├── base.py
│   │   ├── bm25.py
│   │   ├── dense.py
│   │   ├── hybrid.py
│   │   ├── faiss_index.py
│   │   └── evaluate.py
│   │
│   ├── ranking/
│   │   ├── features.py
│   │   ├── pointwise.py
│   │   ├── pairwise.py
│   │   ├── lambdamart.py
│   │   ├── cross_encoder.py
│   │   ├── cascade.py
│   │   └── evaluate.py
│   │
│   ├── prediction/
│   │   ├── ctr.py
│   │   ├── cvr.py
│   │   ├── calibration.py
│   │   └── evaluate.py
│   │
│   ├── simulator/
│   │   ├── users.py
│   │   ├── products.py
│   │   ├── advertisers.py
│   │   ├── campaigns.py
│   │   ├── click_model.py
│   │   ├── conversion_model.py
│   │   ├── position_bias.py
│   │   ├── drift.py
│   │   ├── auction.py
│   │   ├── environment.py
│   │   └── oracle.py
│   │
│   ├── bandits/
│   │   ├── base.py
│   │   ├── random_policy.py
│   │   ├── greedy.py
│   │   ├── epsilon_greedy.py
│   │   ├── ucb.py
│   │   ├── linucb.py
│   │   ├── thompson.py
│   │   ├── neural_ucb.py
│   │   └── evaluate.py
│   │
│   ├── ope/
│   │   ├── logging_policy.py
│   │   ├── replay.py
│   │   ├── ips.py
│   │   ├── snips.py
│   │   ├── direct_method.py
│   │   ├── doubly_robust.py
│   │   ├── switch_dr.py
│   │   └── evaluate.py
│   │
│   ├── rl/
│   │   ├── envs/
│   │   │   ├── bidding_env.py
│   │   │   ├── pacing_env.py
│   │   │   └── wrappers.py
│   │   ├── baselines/
│   │   │   ├── fixed_bid.py
│   │   │   ├── uniform_pacing.py
│   │   │   ├── greedy.py
│   │   │   └── pid.py
│   │   ├── dqn.py
│   │   ├── ppo.py
│   │   ├── sac.py
│   │   └── evaluate.py
│   │
│   ├── multi_agent/
│   │   ├── env.py
│   │   ├── agents.py
│   │   ├── competitors.py
│   │   └── evaluate.py
│   │
│   ├── serving/
│   │   ├── app.py
│   │   ├── retrieval_service.py
│   │   ├── ranking_service.py
│   │   └── policy_service.py
│   │
│   └── evaluation/
│       ├── ranking_metrics.py
│       ├── calibration_metrics.py
│       ├── bandit_metrics.py
│       ├── ope_metrics.py
│       ├── rl_metrics.py
│       ├── latency.py
│       └── reports.py
│
├── scripts/
│   ├── download_data.py
│   ├── build_catalog.py
│   ├── build_bm25.py
│   ├── build_dense_index.py
│   ├── train_ranker.py
│   ├── train_response_model.py
│   ├── generate_logged_feedback.py
│   ├── run_bandit.py
│   ├── run_ope.py
│   ├── train_rl.py
│   ├── evaluate_rl.py
│   └── run_experiment.py
│
├── experiments/
│   ├── retrieval/
│   ├── ranking/
│   ├── bias_correction/
│   ├── bandits/
│   ├── drift/
│   ├── ope/
│   ├── rl_pacing/
│   ├── multi_agent/
│   └── end_to_end/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── simulator/
│   └── smoke/
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DATA.md
│   ├── SIMULATOR.md
│   ├── EXPERIMENTS.md
│   ├── DECISIONS.md
│   └── LIMITATIONS.md
│
└── artifacts/
    ├── indexes/
    ├── models/
    ├── logs/
    ├── metrics/
    ├── plots/
    └── reports/
```

Core production logic must live in `src/`, not notebooks.

Notebooks may be added for exploration and visualization, but they must call reusable library code.

---

# 6. Technology Preferences

Prefer:

- Python 3.11+
- PyTorch
- Hugging Face Transformers / sentence-transformers where appropriate
- FAISS for ANN retrieval
- BM25 implementation through a maintained search package
- LightGBM for LambdaMART / tree-based LTR
- XGBoost or LightGBM for tabular response baselines
- Polars and/or PyArrow for large tabular processing
- DuckDB for local analytics
- Gymnasium-compatible RL environments
- Stable-Baselines3 for first-pass DQN/PPO/SAC implementations
- Open Bandit Pipeline where useful for real logged bandit data and OPE validation
- Hydra, OmegaConf, or an equivalently clean typed configuration layer
- MLflow, Weights & Biases, or a local structured experiment tracker; do not make paid SaaS mandatory
- pytest
- ruff
- mypy or pyright
- FastAPI for the optional serving layer

Do not introduce heavy distributed infrastructure until profiling proves it is necessary.

---

# 7. Canonical Data Model

Define explicit typed schemas.

At minimum:

## 7.1 Product

```python
Product(
    product_id: str,
    title: str,
    description: str | None,
    brand: str | None,
    category: str | None,
    locale: str,
    metadata: dict[str, Any],
)
```

## 7.2 QueryEvent

```python
QueryEvent(
    event_id: str,
    query_id: str,
    query_text: str,
    user_id: str | None,
    timestamp: datetime,
    device: str | None,
    locale: str,
    context: dict[str, Any],
)
```

## 7.3 Candidate

```python
Candidate(
    query_id: str,
    product_id: str,
    retrieval_score: float,
    rank: int,
    source: str,
    features: dict[str, float],
)
```

## 7.4 RelevanceLabel

Internally map ESCI labels to a documented graded scale.

Example default:

```text
Exact       -> 3
Substitute  -> 2
Complement  -> 1
Irrelevant  -> 0
```

This mapping must be configurable and documented.

## 7.5 Advertiser

```python
Advertiser(
    advertiser_id: str,
    budget: float,
    value_per_conversion: float,
    risk_tolerance: float,
    quality_score: float,
)
```

## 7.6 Campaign

```python
Campaign(
    campaign_id: str,
    advertiser_id: str,
    product_ids: list[str],
    daily_budget: float,
    base_bid: float,
    target_categories: list[str],
    target_roi: float | None,
)
```

## 7.7 AuctionOpportunity

```python
AuctionOpportunity(
    opportunity_id: str,
    query_event: QueryEvent,
    eligible_campaign_ids: list[str],
    candidate_product_ids: list[str],
    timestamp: datetime,
)
```

## 7.8 LoggedBanditFeedback

```python
LoggedBanditFeedback(
    event_id: str,
    context: np.ndarray,
    action: int,
    reward: float,
    propensity: float,
    position: int | None,
    candidate_set: list[str],
    metadata: dict[str, Any],
)
```

Propensity is mandatory whenever the logging policy is stochastic and the log is intended for OPE.

---

# 8. Dataset Plan

## 8.1 Primary backbone: Amazon Shopping Queries / ESCI

Use ESCI for:

- semantic query-product matching,
- candidate relevance,
- graded ranking evaluation,
- retrieval and reranking.

Default initial locale: English.

Do not limit the project permanently to the provided judged candidate set.

Build a catalog index from the available product corpus and support retrieval from a substantially broader catalog.

Canonical labels:

- Exact
- Substitute
- Complement
- Irrelevant

### Required outputs

- normalized product catalog,
- query table,
- relevance judgments,
- train/validation/test splits,
- dataset fingerprint,
- catalog statistics,
- label distribution.

### Leakage rules

Do not allow exact test query-product labels into training features.

Do not randomly split rows if that causes the same query to appear across train and test in a way that invalidates evaluation.

Prefer query-group-aware splitting.

---

## 8.2 Real logged feedback: Open Bandit Dataset

Use for:

- off-policy evaluation,
- validation of OPE estimators on real logged bandit feedback,
- policy-learning experiments where compatible.

Do not claim it is semantically integrated with ESCI.

Treat it as a separate external-validation track.

---

## 8.3 Auxiliary CTR track: Criteo

Use for:

- large-scale click prediction,
- calibration,
- tabular response-model comparisons.

Do not pretend anonymized Criteo features correspond to ESCI query or product semantics.

The Criteo track is auxiliary and should produce its own experiment report.

Default development should use a manageable subset.

Processing the full largest release is not required for the main project.

---

## 8.4 Optional sequential track: KuaiRand / KuaiRec

Use only after the primary pipeline works.

Potential uses:

- sequential recommendation,
- exposure bias,
- policy evaluation,
- offline RL,
- reward modeling.

Keep this as an external validation or extension track.

---

# 9. Phase 0 — Engineering Foundation

## Goal

Create a reproducible project skeleton before model development.

## Required tasks

- initialize package structure,
- define typed configs,
- define artifact directories,
- implement deterministic seeding utilities,
- implement structured logging,
- configure linting and tests,
- add smoke-test CI,
- create `docs/DECISIONS.md`,
- create experiment metadata schema.

## Required command surface

At minimum:

```bash
make setup
make lint
make typecheck
make test
make smoke
```

## Exit criteria

- clean environment install works,
- tests pass,
- one smoke experiment writes a valid run directory,
- run metadata includes config, seed, git commit, and metrics.

---

# 10. Phase 1 — ESCI Data Pipeline

## Goal

Build a robust retrieval/ranking dataset pipeline.

## Tasks

1. Download or ingest ESCI data.
2. Normalize product text.
3. Construct query and relevance tables.
4. Implement query-group-aware splits.
5. Build catalog-level product representation.
6. Add tiny fixture dataset for tests.
7. Add full-data and sampled-data configs.

## Required validation

- no duplicate primary keys,
- no missing required IDs,
- valid ESCI label values,
- train/validation/test isolation checks,
- report label distributions,
- report catalog coverage.

## Exit criteria

A command such as:

```bash
python scripts/download_data.py --config configs/data/esci.yaml
```

followed by:

```bash
python scripts/build_catalog.py --config configs/data/esci.yaml
```

must produce validated artifacts and a dataset report.

---

# 11. Phase 2 — Candidate Retrieval

## Goal

Compare lexical, dense, and hybrid retrieval over a large product catalog.

## 11.1 BM25 baseline

Implement:

```text
query -> BM25 index -> top N products
```

Required metrics:

- Recall@10
- Recall@50
- Recall@100
- Recall@500
- MRR
- NDCG@10 where valid
- p50 latency
- p95 latency
- index size
- build time

## 11.2 Dense bi-encoder

Implement:

```text
query encoder -> q
product encoder -> d
score = similarity(q, d)
```

Use a pretrained encoder first.

Fine-tuning is a separate experiment, not a prerequisite for a baseline.

Required:

- offline embedding generation,
- persistent FAISS index,
- batched retrieval,
- CPU and GPU-compatible paths where practical.

## 11.3 Hybrid retrieval

Implement configurable fusion.

Minimum:

```text
score = alpha * normalized_bm25
      + (1 - alpha) * normalized_dense
```

Also consider reciprocal rank fusion as an additional baseline.

Tune fusion only on validation data.

## Experiment E1

Compare:

- BM25
- Dense
- Hybrid weighted fusion
- Reciprocal rank fusion

Do not assert hybrid wins until measured.

## Exit criteria

A single experiment command produces a comparison table and latency report for all retrieval methods.

---

# 12. Phase 3 — Learning to Rank

## Goal

Rerank retrieved candidates using increasingly expressive models.

## 12.1 Heuristic baseline

Use configurable weighted features such as:

- BM25 score,
- dense similarity,
- token overlap,
- title match,
- brand match,
- category match.

## 12.2 Pointwise baseline

Predict graded or binary relevance from query-product features.

## 12.3 Pairwise ranker

Construct preference pairs from ESCI grades.

Example:

```text
Exact > Substitute > Complement > Irrelevant
```

Document pair-sampling strategy.

## 12.4 LambdaMART

Use grouped ranking data.

Candidate features may include:

- BM25 score,
- dense similarity,
- cross-encoder score,
- title overlap,
- category consistency,
- brand consistency,
- query length,
- product-text length,
- retrieval rank,
- source indicators.

Never include target leakage.

## 12.5 Cross-encoder reranker

Rerank only a limited top-M candidate set.

Do not run a cross-encoder over the full catalog.

Measure quality/latency tradeoff.

## 12.6 Cascade

Canonical cascade:

```text
catalog
  -> retrieve top 500
  -> LambdaMART top 50
  -> cross-encoder top 10
```

Make N/M/K configurable.

## Experiment E2

Compare:

- retrieval score only,
- heuristic ranker,
- pointwise,
- pairwise,
- LambdaMART,
- cross-encoder,
- cascade.

Required metrics:

- NDCG@5
- NDCG@10
- MRR
- MAP where appropriate
- Recall@K
- p50/p95 latency
- throughput

## Exit criteria

Produce a quality-latency Pareto plot and a reproducible ranking report.

---

# 13. Phase 4 — Response Prediction and Calibration

## Goal

Separate relevance from response probability.

A relevant item is not necessarily equally likely to be clicked or converted.

## 13.1 Simulator response model

The simulator must expose latent click and conversion probabilities.

## 13.2 Optional Criteo benchmark

Compare at least:

- logistic regression,
- GBDT,
- one neural CTR model such as DeepFM or DCN.

## Metrics

- Log Loss
- ROC-AUC
- PR-AUC where appropriate
- Brier Score
- Expected Calibration Error
- reliability diagrams

## Calibration methods

Implement at least:

- Platt/logistic calibration,
- isotonic calibration.

Temperature scaling may be added for neural logits.

## Experiment E3A

Compare raw vs calibrated predictions.

## Exit criteria

A report must show discrimination and calibration separately.

Do not treat AUC as sufficient.

---

# 14. Phase 5 — Controlled User and Feedback Simulator

## Goal

Create a configurable environment that produces logged interaction feedback while preserving hidden ground truth for scientific evaluation.

This simulator is the bridge between offline ranking and online learning.

## 14.1 User representation

Each user or user cohort should have latent preferences over dimensions such as:

- categories,
- brands,
- price sensitivity,
- quality sensitivity,
- novelty preference.

Example:

```text
user_preference = [
    sports=0.9,
    electronics=0.5,
    fashion=0.2,
    price_sensitivity=0.8,
]
```

## 14.2 Product representation

Products should expose:

- semantic relevance,
- category,
- brand,
- quality,
- price or price bucket,
- novelty,
- advertiser/campaign ownership.

## 14.3 Latent utility

Default configurable form:

```text
utility(user, query, product, t)
    = w_rel * relevance
    + w_aff * user_affinity
    + w_quality * quality
    + w_price * price_match
    + w_novelty * novelty
    + temporal_effect
    + noise
```

Then:

```text
p(click | examined) = sigmoid(utility)
```

## 14.4 Position bias

Model examination separately:

```text
p(click)
    = p(examine | position)
    * p(click | examined)
```

Position examination probabilities must be configurable.

## 14.5 Conversion

Default:

```text
p(conversion | click)
    = sigmoid(conversion_utility)
```

Conversion value may vary by advertiser.

## 14.6 Logged feedback

Every exposure log must include:

- context,
- candidate set,
- chosen action,
- action probability/propensity,
- displayed position,
- click,
- conversion,
- reward,
- timestamp,
- policy name,
- policy version.

## 14.7 Oracle

The simulator may expose latent ground-truth policy value only to evaluation code.

Training policies must not access oracle values.

## Exit criteria

Tests verify:

- click probabilities are bounded,
- stronger relevance increases expected click probability under controlled settings,
- worse position reduces expected observed clicks under default position-bias config,
- identical seeds reproduce trajectories,
- drift schedules trigger at expected times.

---

# 15. Phase 6 — Bias Correction

## Goal

Demonstrate why observed clicks are not equivalent to relevance.

## Experiment E3B

Generate logs with position-biased exposure.

Train/evaluate:

1. naive click-based model,
2. inverse propensity weighting,
3. clipped IPS,
4. self-normalized IPS where appropriate,
5. doubly robust method.

Evaluate against hidden simulator ground truth.

Required outcomes:

- ranking quality against true relevance,
- policy value error,
- estimator bias,
- estimator variance,
- sensitivity to propensity clipping.

Do not design the simulator so that a preferred method is guaranteed to win.

---

# 16. Phase 7 — Contextual Bandit Layer

## Goal

Add online exploration over the top-K ranked candidate set.

The bandit is not responsible for catalog retrieval.

Canonical flow:

```text
retriever -> ranker -> top K -> bandit action -> exposure
```

## 16.1 Context

Candidate context may include:

- query embedding,
- user/cohort vector,
- time features,
- device,
- candidate relevance,
- pCTR,
- pCVR,
- quality,
- uncertainty,
- cold-start flag.

## 16.2 Action

Initial action:

```text
choose one item from top K
```

Later extension:

```text
choose an ordered slate
```

Slate optimization is optional until single-action bandits are correct.

## 16.3 Reward

Default configurable reward:

```text
reward =
    click_weight * click
  + conversion_weight * conversion
  + value_weight * conversion_value
  - low_quality_penalty
```

## Algorithms

Required:

- Random
- Greedy
- Epsilon-Greedy
- UCB
- LinUCB
- Thompson Sampling

Advanced:

- NeuralUCB or another neural contextual-bandit method.

## Metrics

- cumulative reward,
- cumulative regret against simulator oracle,
- CTR,
- conversion rate,
- cold-start performance,
- exploration cost,
- time to discover strong new items.

## Experiment E4

Compare all required policies under:

- stationary preferences,
- cold-start products,
- sparse rewards.

Use multiple seeds.

## Exit criteria

Generate learning curves with confidence intervals.

Single-seed RL/bandit plots are not acceptable as final evidence.

---

# 17. Phase 8 — Non-Stationarity and Preference Drift

## Goal

Evaluate adaptation when the environment changes.

## Drift types

Implement configurable:

1. global category trend shift,
2. cohort preference shift,
3. individual preference drift,
4. advertiser entry/exit,
5. reward-scale change,
6. competition shift.

Example:

```text
rounds 0-100k:
    running products favored

rounds 100k-200k:
    hiking products favored

rounds 200k+:
    winter products favored
```

## Algorithms to compare

- Static ranker
- Greedy online policy
- LinUCB
- Thompson Sampling
- Sliding-Window UCB
- Discounted UCB or discounted Thompson variant

## Metrics

- pre-shift reward,
- post-shift reward,
- immediate degradation,
- recovery time,
- cumulative regret after shift,
- stability under recurring shifts.

## Experiment E5

Primary question:

> Which policy recovers fastest after preference drift, and what exploration cost does it pay before and after the shift?

---

# 18. Phase 9 — Off-Policy Evaluation

## Goal

Estimate new policy value from logged feedback.

This phase is mandatory.

## 18.1 Logging policy requirement

For OPE experiments, the logging policy must provide valid propensities.

Do not use a deterministic policy and then pretend unsupported counterfactual actions can be evaluated with IPS.

Enforce overlap/support checks.

## 18.2 Estimators

Required:

- Replay where applicable
- IPS
- SNIPS
- Direct Method
- Doubly Robust

Advanced:

- Switch-DR

## 18.3 Simulator validation

Because the simulator can execute target policies directly, use it to compare:

```text
OPE estimated value
vs
true online policy value
```

Measure:

- bias,
- variance,
- MSE,
- rank correlation across candidate policies,
- confidence interval coverage where implemented.

## 18.4 Real-data validation

Run compatible estimators through Open Bandit Dataset / Pipeline.

Keep this separate from simulator conclusions.

## Experiment E6

Questions:

1. Which OPE estimator best estimates policy value under adequate overlap?
2. How do estimators degrade when overlap worsens?
3. How does propensity clipping trade bias for variance?
4. Do simulator conclusions transfer qualitatively to real logged feedback?

## Exit criteria

Produce at least:

- estimator error table,
- policy-ranking correlation plot,
- overlap stress test,
- real-data OPE report.

---

# 19. Phase 10 — Auction Simulator

## Goal

Introduce repeated competition, bids, budgets, and advertiser value.

## 19.1 Advertisers

Each advertiser has:

- budget,
- base bid,
- value per conversion,
- quality score,
- campaign targets,
- optional ROI target.

## 19.2 Eligibility

An advertiser/campaign is eligible only if:

- relevant to the opportunity,
- budget remains,
- targeting constraints pass.

## 19.3 Auction score

Default configurable mechanism:

```text
ad_score =
    bid
    * predicted_ctr
    * quality_score
```

Do not claim this reproduces a proprietary company auction.

The mechanism is a research simulator assumption.

## 19.4 Payment mechanisms

Support at least one documented mechanism.

Possible:

- first-price,
- simplified second-price-style.

If multiple mechanisms are implemented, compare them explicitly.

## 19.5 Opportunity stream

An episode consists of a time-ordered stream of impression opportunities.

Demand intensity should vary by time of day.

## Exit criteria

Simulator tests verify:

- no advertiser spends beyond allowed tolerance,
- ineligible campaigns cannot win,
- budget updates are correct,
- same seed reproduces auctions,
- counterfactual baselines can be evaluated consistently.

---

# 20. Phase 11 — RL Bidding and Budget Pacing

## Goal

Optimize long-horizon advertiser value under finite budgets.

This is the main RL component.

## 20.1 Episode

Default:

```text
one simulated day
```

Configurable horizon.

## 20.2 State

Initial state vector:

```text
remaining_budget_fraction
time_remaining_fraction
current_spend_rate
target_spend_rate
predicted_ctr
predicted_cvr
relevance_score
quality_score
estimated_competition
recent_win_rate
recent_cpa
recent_reward
```

Normalize all continuous state features.

## 20.3 Action spaces

### Discrete action space

Bid multiplier:

```text
0.00x
0.50x
0.75x
1.00x
1.25x
1.50x
```

Use for DQN and PPO experiments.

### Continuous action space

```text
bid_multiplier in [0, 2]
```

Use for SAC and PPO-continuous experiments.

Do not compare algorithms across different action spaces without clearly stating the difference.

## 20.4 Reward

Default:

```text
reward_t
    = conversion_value_t
    - spend_t
    - lambda_pace * pacing_error_t
    - lambda_risk * budget_violation_risk_t
```

Optional terminal reward:

```text
terminal_bonus
    = final_profit
    - lambda_unused * abs(target_spend - actual_spend)
```

Every coefficient must be configurable.

## 20.5 Non-RL baselines

Required:

- Fixed bid
- Greedy pCTR/pCVR bid policy
- Uniform pacing
- PID pacing controller

## 20.6 RL algorithms

Required:

- DQN for discrete action space
- PPO
- SAC for continuous action space

Do not implement custom RL algorithms before validated library baselines work.

## 20.7 Evaluation

Required metrics:

- total advertiser value,
- profit,
- ROAS,
- conversions,
- spend,
- budget utilization,
- pacing error,
- CPA,
- overspend violations,
- reward,
- end-of-day unused budget,
- stability across seeds.

Use at least 5 seeds for final RL claims unless compute constraints are explicitly documented.

## Experiment E7

Compare:

- fixed bid,
- greedy,
- uniform pacing,
- PID,
- DQN,
- PPO,
- SAC.

Primary question:

> Does RL improve long-horizon advertiser value relative to strong pacing baselines, and under which demand/competition regimes?

Do not expect RL to always win.

A finding that PID or a simple controller is more stable is valid and interesting.

---

# 21. Phase 12 — Multi-Agent Auction Learning

## Goal

Study multiple adaptive advertisers competing in repeated auctions.

This phase should begin only after the single-agent auction/RL environment is validated.

## 21.1 Scenarios

Required:

### Scenario A

```text
one learning advertiser
vs
fixed competitors
```

### Scenario B

```text
one learning advertiser
vs
heuristic adaptive competitors
```

### Scenario C

```text
multiple independent RL advertisers
```

## 21.2 Heterogeneity

Vary:

- budgets,
- conversion values,
- base bids,
- quality scores,
- learning rates,
- risk tolerance.

## 21.3 Research questions

- Do independent learners converge?
- Does simultaneous adaptation destabilize policies?
- Do bidding wars emerge?
- How does budget asymmetry affect learned strategy?
- Can one adaptive advertiser exploit slower competitors?
- Does a policy trained against fixed competitors fail against adaptive competitors?
- How sensitive are learned strategies to auction mechanism?

## Metrics

In addition to single-agent metrics:

- market share,
- advertiser utility,
- price inflation,
- bid volatility,
- strategy entropy,
- policy stability,
- regret against fixed competitors,
- fairness/market concentration diagnostics where meaningful.

## Experiment E8

Compare:

- all fixed,
- one RL learner,
- all independent RL learners.

Use multiple seeds.

---

# 22. Phase 13 — End-to-End Integration

## Goal

Connect the complete stack without collapsing module boundaries.

Canonical online request path:

```text
query
 -> retrieve top 500
 -> LTR top 50
 -> neural rerank top 10
 -> pCTR/pCVR
 -> bandit exposure decision
 -> campaign eligibility
 -> RL bid multiplier
 -> auction
 -> interaction
 -> logging
 -> policy updates
```

## Requirements

- every stage can be disabled by config,
- deterministic baseline path exists,
- latency is measured per stage,
- online logs contain enough information for replay and OPE,
- model and policy versions are recorded.

## Experiment E9

Ablate:

1. offline ranking only,
2. ranking + pCTR,
3. ranking + bandit,
4. ranking + bandit + debiasing,
5. full ranking + bandit + RL pacing.

Do not compare unlike objectives without a clear metric definition.

---

# 23. Serving Layer

This is secondary to research correctness but should demonstrate systems competence.

Implement a FastAPI service with endpoints such as:

```text
POST /retrieve
POST /rank
POST /decide
POST /feedback
GET  /health
```

Possible request:

```json
{
  "query": "waterproof trail running shoes",
  "user_context": {
    "cohort": "outdoor_enthusiast",
    "device": "mobile"
  }
}
```

Possible response:

```json
{
  "request_id": "...",
  "candidates": [...],
  "selected_ad": "...",
  "policy": "linucb_v3",
  "model_versions": {
    "retriever": "...",
    "ranker": "...",
    "ctr": "..."
  }
}
```

Do not expose private simulator oracle values through serving APIs.

Measure:

- end-to-end p50 latency,
- p95 latency,
- stage latency breakdown,
- throughput under a simple load test.

---

# 24. Experiment Matrix

The final repository should contain the following major experiments.

## E1 — Retrieval

Question:

> Does dense or hybrid retrieval improve candidate recall over BM25?

Compare:

- BM25
- Dense
- Hybrid
- RRF

Metrics:

- Recall@K
- MRR
- NDCG
- latency

---

## E2 — Ranking

Question:

> How much does learned ranking improve over retrieval scores, and at what latency cost?

Compare:

- retrieval score
- heuristic
- pointwise
- pairwise
- LambdaMART
- cross-encoder
- cascade

Metrics:

- NDCG@5
- NDCG@10
- MRR
- latency

---

## E3 — Calibration and biased feedback

Question:

> What breaks when raw clicks are treated as unbiased relevance labels?

Compare:

- raw response model
- calibrated model
- naive click training
- IPS
- clipped IPS
- DR

Metrics:

- Log Loss
- AUC
- Brier
- ECE
- true NDCG
- policy value error

---

## E4 — Online exploration

Question:

> Which bandit policy best discovers strong uncertain or cold-start items?

Compare:

- random
- greedy
- epsilon-greedy
- UCB
- LinUCB
- Thompson Sampling
- optional NeuralUCB

Metrics:

- cumulative reward
- regret
- discovery time
- CTR
- conversion rate

---

## E5 — Drift

Question:

> Which online policy recovers fastest after preferences shift?

Compare:

- static ranker
- greedy
- LinUCB
- Thompson
- sliding-window UCB
- discounted policy

Metrics:

- recovery time
- post-shift regret
- reward

---

## E6 — OPE

Question:

> Which estimator most accurately ranks candidate policies from logged data?

Compare:

- Replay
- IPS
- SNIPS
- DM
- DR
- Switch-DR

Metrics:

- bias
- variance
- MSE
- policy rank correlation
- overlap sensitivity

Run on:

- simulator logs with known ground truth,
- Open Bandit Dataset as external validation.

---

## E7 — RL pacing

Question:

> Can learned sequential policies improve long-horizon advertiser value under finite budgets?

Compare:

- fixed bid
- greedy
- uniform pacing
- PID
- DQN
- PPO
- SAC

Metrics:

- profit
- ROAS
- conversions
- pacing error
- utilization
- CPA
- violations

---

## E8 — Multi-agent competition

Question:

> What changes when multiple adaptive advertisers compete?

Compare:

- fixed competitors
- one learner
- multiple learners

Metrics:

- advertiser utility
- market share
- bid volatility
- stability
- concentration

---

## E9 — End-to-end ablation

Question:

> Which layers provide incremental value, and when?

Compare:

- ranking only
- ranking + pCTR
- ranking + bandit
- ranking + bandit + OPE-informed selection
- full stack with RL pacing

---

# 25. Evaluation Standards

## 25.1 Seeds

For final stochastic experiments:

- use multiple seeds,
- report mean and uncertainty,
- preserve per-seed raw metrics.

Default:

```text
5 seeds
```

Use more if cheap.

## 25.2 Data splits

Never tune on test data.

Maintain:

- train
- validation
- test

For temporal experiments, use time-aware splits where appropriate.

## 25.3 Statistical reporting

Prefer:

- bootstrap confidence intervals,
- mean ± standard error,
- paired comparisons when runs share conditions.

Do not overclaim significance.

## 25.4 Failure analysis

Every major experiment report should include at least one failure slice.

Examples:

- long-tail queries,
- cold-start products,
- ambiguous queries,
- low-overlap OPE regions,
- high-competition auctions,
- low-budget advertisers,
- sudden drift.

---

# 26. Compute and Cost Guardrails

The default target is a strong research-grade project that can be completed with local development plus rented single-GPU instances.

## Principles

1. Do not train large language models.
2. Do not train text encoders from scratch.
3. Cache embeddings.
4. Cache FAISS indexes.
5. Use sampled configs for development.
6. Use full configs only for final runs.
7. Separate CPU-heavy and GPU-heavy stages.
8. Run one-seed smoke tests before multi-seed sweeps.
9. Record GPU-hours and wall-clock time.
10. Never process the largest Criteo release end-to-end merely for prestige.

## Target planning envelope

Default project target:

```text
$150-$400 total external compute
```

This is a planning guardrail, not a promise.

A more aggressive run budget may be allowed if justified in `docs/DECISIONS.md`.

## Before any sweep

Run:

1. tiny fixture,
2. sampled smoke test,
3. one-seed full-shape test,
4. only then multi-seed sweep.

---

# 27. Experiment Artifact Contract

Every run must create:

```text
artifacts/runs/<run_id>/
├── config.yaml
├── metadata.json
├── metrics.json
├── stdout.log
├── stderr.log
├── git_commit.txt
├── environment.txt
├── plots/
└── checkpoints/
```

`metadata.json` should include:

```json
{
  "run_id": "...",
  "experiment": "...",
  "seed": 42,
  "git_commit": "...",
  "dataset_fingerprint": "...",
  "start_time": "...",
  "end_time": "...",
  "status": "SUCCESS"
}
```

Allowed statuses:

- `SUCCESS`
- `FAILED`
- `PARTIAL`
- `BLOCKED`
- `NOT_RUN`

---

# 28. Testing Requirements

## Unit tests

Required for:

- metric calculations,
- propensity weighting,
- reward functions,
- budget accounting,
- auction winner selection,
- calibration utilities,
- drift schedule,
- state normalization.

## Integration tests

Required for:

- query -> retrieval,
- retrieval -> ranking,
- ranking -> bandit,
- bandit -> feedback log,
- auction -> budget update,
- Gymnasium env step/reset.

## Simulator invariants

Test:

- no negative remaining budget beyond tolerance,
- click probability in [0, 1],
- conversion implies valid configured event flow,
- deterministic seeds reproduce trajectories,
- OPE propensity is positive for logged action,
- oracle information is not exposed to training policy.

## Smoke tests

Every major training command must have a tiny smoke config that finishes quickly.

---

# 29. Documentation Requirements

## README.md

The README should eventually contain:

1. project motivation,
2. architecture diagram,
3. dataset strategy,
4. quickstart,
5. experiment table,
6. final results,
7. key findings,
8. limitations,
9. reproducibility instructions.

Do not fill final result tables until experiments run.

## docs/ARCHITECTURE.md

Explain module interactions.

## docs/DATA.md

For each dataset:

- purpose,
- source,
- license considerations,
- preprocessing,
- split strategy,
- what conclusions it can and cannot support.

## docs/SIMULATOR.md

Document every major simulator assumption.

## docs/EXPERIMENTS.md

Define experiment hypotheses before results.

## docs/DECISIONS.md

Record architectural decisions and rejected alternatives.

## docs/LIMITATIONS.md

Explicitly discuss:

- public-data mismatch with real proprietary ads systems,
- simulator assumptions,
- reward misspecification,
- unobserved confounding,
- OPE overlap,
- auction simplification,
- external validity.

---

# 30. Codex Operating Instructions

These instructions govern agent behavior while implementing this repository.

## 30.1 Work phase by phase, but preserve the full architecture

Do not remove later phases merely because the current phase is earlier.

Implement clean interfaces so later modules can attach without major rewrites.

## 30.2 Before coding a phase

1. inspect existing code,
2. read relevant configs,
3. read `docs/DECISIONS.md`,
4. identify interfaces affected,
5. state the implementation plan in the task log,
6. then edit.

## 30.3 After coding a phase

Always:

1. run formatter/linter,
2. run targeted tests,
3. run relevant smoke test,
4. inspect generated artifacts,
5. update docs,
6. report exact commands and outcomes.

## 30.4 Do not silently weaken requirements

If a requested feature cannot be completed:

- implement the maximum honest subset,
- mark the rest `PARTIAL` or `BLOCKED`,
- explain the blocker,
- do not substitute a toy implementation and call it complete.

## 30.5 No placeholder science

Forbidden:

- fake result tables,
- hard-coded “improvement” metrics,
- random numbers presented as model output,
- unrun experiment conclusions,
- claims that a method is “better” without results.

## 30.6 Prefer simple correct code over premature distributed systems

Do not add:

- Kubernetes,
- Spark,
- Ray clusters,
- feature stores,
- Kafka,

unless profiling or project requirements justify them.

The project may later add such systems, but they are not substitutes for experimental correctness.

## 30.7 Keep dependencies intentional

Before adding a dependency:

- verify existing libraries do not already solve the need,
- prefer maintained packages,
- avoid overlapping libraries for the same job.

## 30.8 Preserve modularity

The following must remain independently replaceable:

- retriever,
- ranker,
- response model,
- simulator,
- bandit policy,
- logging policy,
- OPE estimator,
- auction mechanism,
- RL policy.

## 30.9 Maintain a task ledger

Create and update:

```text
docs/TASKS.md
```

Use:

```text
[ ] not started
[~] in progress
[x] complete
[!] blocked
```

A task is complete only after tests and smoke runs pass.

## 30.10 Maintain a results ledger

Create:

```text
docs/RESULTS.md
```

Every result entry must point to an artifact run directory.

No orphan numbers.

---

# 31. Milestone Gates

## Milestone M0 — Foundation

Complete when:

- package installs,
- CI passes,
- run metadata works.

## Milestone M1 — Data

Complete when:

- ESCI ingestion works,
- splits validate,
- dataset report exists.

## Milestone M2 — Retrieval

Complete when:

- BM25,
- dense,
- hybrid

all run through one evaluation harness.

## Milestone M3 — Ranking

Complete when:

- LambdaMART and cross-encoder can rerank retrieved candidates,
- cascade metrics and latency are reported.

## Milestone M4 — Simulator

Complete when:

- user response,
- position bias,
- conversion,
- logged propensities,
- oracle evaluation

are tested.

## Milestone M5 — Bandits

Complete when:

- required bandit baselines run,
- regret/reward curves are reproducible,
- cold-start experiment works.

## Milestone M6 — OPE

Complete when:

- IPS/SNIPS/DM/DR run,
- simulator estimated-vs-true value comparison exists,
- Open Bandit external validation exists.

## Milestone M7 — Drift

Complete when:

- at least three drift policies are compared,
- recovery-time metric is implemented.

## Milestone M8 — Auction

Complete when:

- repeated auctions,
- budgets,
- eligibility,
- payments,
- opportunity stream

are validated.

## Milestone M9 — RL Pacing

Complete when:

- strong non-RL baselines,
- PPO,
- DQN,
- SAC

are evaluated with multiple seeds.

## Milestone M10 — Multi-Agent

Complete when:

- fixed-vs-learning and multi-learning scenarios run.

## Milestone M11 — End-to-End

Complete when:

- complete configurable request path runs,
- stage latency is measured,
- final ablation report exists.

---

# 32. Definition of Done

The project is complete only when all of the following are true:

## Offline ML

- BM25 retrieval works.
- Dense retrieval works.
- Hybrid retrieval works.
- LTR baseline works.
- LambdaMART works.
- Cross-encoder reranking works.
- Ranking metrics and latency are reported.

## Feedback and prediction

- response model exists,
- calibration is evaluated,
- position bias is modeled,
- logged feedback includes propensities.

## Online learning

- required bandit policies work,
- regret/reward curves exist,
- cold-start experiment exists,
- drift experiment exists.

## Counterfactual evaluation

- IPS works,
- SNIPS works,
- DM works,
- DR works,
- simulator OPE validation exists,
- Open Bandit external validation exists.

## RL

- auction simulator works,
- budgets are enforced,
- non-RL pacing baselines work,
- DQN works,
- PPO works,
- SAC works,
- final RL evaluation uses multiple seeds.

## Multi-agent

- at least one multi-agent competition experiment runs.

## Engineering

- tests pass,
- configs are versioned,
- runs are reproducible,
- no fabricated results,
- final README links claims to run artifacts.

---

# 33. Preferred Final Research Narrative

The final repository and report should tell a coherent story:

### Part I — Offline relevance

> How well can a multi-stage retrieval and ranking cascade identify relevant sponsored products?

### Part II — Learning from interaction

> What happens when the system learns from clicks distorted by exposure and position bias?

### Part III — Exploration

> Can contextual bandits discover strong uncertain items and adapt to preference drift better than static rankers?

### Part IV — Counterfactual evaluation

> Can we estimate new policy value from logged feedback without deploying every policy online?

### Part V — Sequential optimization

> Can RL improve bidding and budget pacing when spending now changes future opportunity?

### Part VI — Strategic interaction

> What changes when multiple adaptive advertisers compete in repeated auctions?

The project should emphasize that these are different but connected decision problems.

---

# 34. First Implementation Order

Unless the repository already contains work that changes dependencies, implement in this order:

```text
1. project skeleton + configs + run tracking
2. ESCI ingestion + validation
3. BM25 retrieval
4. dense retrieval + FAISS
5. hybrid retrieval
6. retrieval benchmark
7. LambdaMART reranking
8. cross-encoder reranking
9. ranking cascade
10. response simulator
11. position bias + logging propensities
12. contextual-bandit baselines
13. drift experiments
14. OPE estimators
15. Open Bandit validation
16. auction simulator
17. non-RL pacing baselines
18. DQN
19. PPO
20. SAC
21. multi-agent competition
22. end-to-end integration
23. optional Criteo CTR track
24. optional KuaiRand/KuaiRec extension
25. serving + latency report
26. final ablations + README
```

Do not skip baseline implementation to jump directly to PPO.

---

# 35. Immediate First Task for Codex

Begin with Milestone M0 and M1.

Specifically:

1. create the repository skeleton,
2. add `pyproject.toml`,
3. set up linting, typing, and tests,
4. create typed configuration support,
5. create experiment run metadata utilities,
6. implement ESCI ingestion adapter,
7. create a tiny deterministic ESCI-like test fixture,
8. implement query-group-aware split validation,
9. generate a dataset summary report,
10. add exact setup and smoke-test commands to README.

Do not begin dense retrieval or RL until the data and run-artifact foundations pass tests.

At the end of the first task, report:

- files created,
- architectural decisions,
- exact commands run,
- test outcomes,
- smoke-run artifact path,
- remaining M1 tasks.

---

# 36. Final Rule

The objective is not to maximize the number of algorithms in the repository.

The objective is to build a system where each algorithm answers a clear question:

```text
Retrieval:
    What could be relevant?

Ranking:
    What is most relevant?

Response prediction:
    What is likely to happen immediately?

Bandits:
    What should we try while learning?

OPE:
    Can we evaluate a new policy from logged data?

RL:
    What action maximizes long-term value under future consequences?

Multi-agent learning:
    What changes when other decision-makers adapt too?
```

Preserve that conceptual clarity throughout the implementation.
