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
