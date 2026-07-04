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
