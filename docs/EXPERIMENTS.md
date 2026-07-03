# Experiments

M0 foundation and M1 data validation runs are engineering verification, not scientific
experiments. E1–E9 are `NOT_RUN` and retain the hypotheses specified in `AGENTS.md`.

The 300-source-train/100-source-test official sample has purpose `integration_verification`.
Its diagnostics may validate plumbing but may not appear in final research result tables.

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
