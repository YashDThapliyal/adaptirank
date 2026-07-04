# Experiments

M0 foundation and M1 data validation runs are engineering verification, not scientific
experiments. E1–E9 are `NOT_RUN` and retain the hypotheses specified in `AGENTS.md`.

The 300-source-train/100-source-test official sample has purpose `integration_verification`.
Its diagnostics may validate plumbing but may not appear in final research result tables.

## E2 ranking protocol

- Candidate source: the separately versioned `M3 three-split retrieval handoff`; canonical M2
  evidence is read-only and remains the M2 benchmark.
- Split roles are locked: train fits model parameters; validation selects hyperparameters,
  features, and early stopping; official test is final evaluation after freezing.
- Primary pointwise and LambdaMART targets use judged train rows only with `E/S/C/I -> 3/2/1/0`.
  Unjudged candidates retain null labels/grades and are scored at inference, never relabeled as
  negatives. Any sampled-unjudged negative experiment must be a separately named ablation.
- Primary label-free features: BM25/dense/weighted-hybrid/RRF scores and ranks, query length,
  title length, query-title lexical overlap, exact-token overlap, and brand match. Category is
  excluded because source coverage is 0%. Cross-encoder score is not a primary LambdaMART feature.
- CE-A is `Hybrid top-100 -> CE`. CE-B is `Hybrid top-500 -> LambdaMART top-50 -> CE`. One A100
  run scores the deduplicated union, and coverage of every LambdaMART-top-50 pair is a hard gate.
- Hardware-mixed latency is labeled by stage and is not presented as directly comparable.

### CE A100 Colab protocol

- Canonical entry point: `notebooks/m3_cross_encoder_a100_runall.ipynb`.
- The notebook must clone `YashDThapliyal/adaptirank` at scoring commit
  `eb67d8f1d8bbba14a58e9a0a12fd787b5efaa01d` (`eb67d8f`). Drive-staged CE inputs were built at `4f327ff86c5a50b11e850620e8b2f8d74311721c`
  (`4f327ff`). Verify the clean checkout, install from the
  lockfile, and use `adaptirank.ranking.ce_workflow` for all gates.
- Inputs are Drive-staged: `MyDrive/adaptirank/m3_ce_a100_input.tar.gz` with SHA-256
  `a79bb8ad98b2cdbfb56b6f6680c95ce87ef1dd792a16ac91d95fec563ee67f5f`, plus
  `MyDrive/adaptirank/adaptirank_dataset.tar`.
- The CE union must verify as exactly 3,156,056 rows with parquet SHA-256
  `16a43b01f0ba159e5950c1fe7d4363b6c05d7b0c9ffe6c581272379ef9c8488d` before scoring.
- The notebook must run the pinned cross-encoder probe, benchmark deterministic validation pairs,
  write Drive checkpoint blocks, consolidate final scores, and verify one finite score for every
  union pair before any cascade evaluation consumes the scores.

## E1 retrieval protocol

- Primary relevance for recall: Exact + Substitute.
- Sensitivity relevance: Exact + Substitute + Complement.
- Missing judgments: unknown, never relabeled Irrelevant or grade 0.
- BM25 field selection: validation Recall@100, then validation NDCG@10.
- Weighted hybrid alpha selection: validation Recall@100, then validation NDCG@10.
- Test metrics are read only after selection is fixed.
- Dense baseline: pinned pretrained encoder with no fine-tuning.
- Required output: overall and quality-latency comparison, raw per-query metrics, query-length,
  lexical-overlap, and label-structure slices, plus representative failures.

### Exact metric semantics (audited)

These definitions are pinned by hand-computable tests in `tests/unit/test_retrieval_metrics.py`
and `tests/unit/test_retrieval_evaluation.py`. They are descriptive of the implementation in
`src/adaptirank/retrieval/evaluate.py`; they were audited, not changed to alter numbers.

- **Relevance labels.** Primary relevance = `{E, S}`. Sensitivity relevance = `{E, S, C}`.
  Numeric grades: `E=3, S=2, C=1, I=0`.
- **Recall@k (primary and sensitivity).** Numerator = number of retrieved items within the raw
  top-k positions whose label is in the relevance set. Denominator = total number of *known*
  judgments in that relevance set for the query (independent of how many were retrieved).
  Unjudged retrieved items are neither hits nor negatives, but they occupy raw rank slots and can
  push judged-relevant items out of the top-k window. A missed known-relevant item that is never
  retrieved still counts in the denominator.
- **Top-k truncation.** Recall@k truncates by *raw* rank (`rank <= k`, unjudged included in the
  positional count). NDCG@k truncates the *condensed judged* list to k judged items on both the
  retrieved and ideal sides. These are deliberately different truncation semantics.
- **MRR.** Condensed reciprocal rank: unjudged items are skipped (do not advance the rank);
  judged non-primary items (`C`, `I`) advance the condensed rank but are not hits. The reciprocal
  rank is `1 / (condensed rank of the first primary-relevant item)`, or `0` if none appears.
- **MRR query eligibility.** MRR is macro-averaged over *all* evaluated queries in the split; a
  query with no retrieved primary-relevant judged item (including a query with zero primary
  judgments) contributes `0.0`. This differs from recall, whose denominator is undefined for a
  zero-relevant query, so such queries are `None` and excluded from the recall mean.
- **NDCG (graded).** Gain = `2**grade - 1` (exponential, so `E=7, S=3, C=1, I=0`); discount =
  `log2(position + 2)` over the condensed judged list. Ideal DCG uses all known judgment grades
  for the query sorted descending (so `C`/`I` grades participate in the denominator). Unjudged
  retrieved items are removed before scoring. NDCG is `0.0` when the ideal DCG is `0`.
- **Zero-relevant query.** Recall → `None` (excluded from means); MRR → `0.0` (included);
  NDCG → `0.0` when ideal DCG is `0` (included).
