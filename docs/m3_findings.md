# M3 Cross-Encoder Findings

## 1 M3 Research Questions

Observed result: M3 evaluates offline ranking baselines, learned rerankers, imported CE cascades, score behavior, slices, examples, and quality-latency from fixed artifacts.

Interpretation: This does not answer online ads, bandit, OPE, auction, or RL questions.

## 2 Dataset And Evaluation Policy

Observed result: Aggregate values and query slices use the test split. Representative examples are test-only, with query text joined from `queries.parquet` and product title/brand from `catalog.parquet`. UNJUDGED is separate from I.

Interpretation: Judged ranking metrics and raw-depth recall answer different questions.

## 3 Canonical Artifact Identifiers/Fingerprints Relevant To M3

Observed result: dataset fingerprint `dda38161938e829f2c2fc9b73d40d6cf922a5470c3b45bf176f742ee0ca7c667`. CE root `/Users/yash/Documents/AdaptiRank/artifacts/ranking/dda38161938e829f2c2fc9b73d40d6cf922a5470c3b45bf176f742ee0ca7c667/m3_three_split/cross_encoder`. Analysis root `/Users/yash/Documents/AdaptiRank/artifacts/ranking/dda38161938e829f2c2fc9b73d40d6cf922a5470c3b45bf176f742ee0ca7c667/m3_three_split/cross_encoder/analysis`. Pair union SHA `16a43b01f0ba159e5950c1fe7d4363b6c05d7b0c9ffe6c581272379ef9c8488d`; scores SHA `923960c5caeef63b33738cb5b4b9ea6cf2163a3a51676359f77dc68a291dd442`.

Interpretation: Imported CE score artifacts were not modified and CE inference was not rerun.

## 4 Baseline Results

Observed result:

BM25: NDCG@5 0.641576, NDCG@10 0.609906, MRR 0.848967, Recall@50 0.298437, Recall@500 0.527779.
Dense: NDCG@5 0.617645, NDCG@10 0.579609, MRR 0.837741, Recall@50 0.248115, Recall@500 0.486403.
Weighted Hybrid: NDCG@5 0.680388, NDCG@10 0.656273, MRR 0.880465, Recall@50 0.325851, Recall@500 0.589773.
RRF: NDCG@5 0.678732, NDCG@10 0.654843, MRR 0.879373, Recall@50 0.328005, Recall@500 0.588507.

Interpretation: Weighted Hybrid is the strongest retrieval/fusion baseline by test NDCG@10.

## 5 Pointwise Results

Observed result: Pointwise: NDCG@5 0.683448, NDCG@10 0.658381, MRR 0.883380, Recall@50 0.291320, Recall@500 0.589773. Pointwise vs LambdaMART dNDCG@10 = -0.000812.

Interpretation: Pointwise is competitive but does not beat LambdaMART on aggregate NDCG@10.

## 6 LambdaMART Results

Observed result: LambdaMART: NDCG@5 0.685001, NDCG@10 0.659193, MRR 0.881553, Recall@50 0.311810, Recall@500 0.589773. LambdaMART vs Weighted Hybrid dNDCG@10 = 0.002920.

Interpretation: LambdaMART is the best aggregate M3 method by test NDCG@10.

## 7 CE-A Results

Observed result: Hybrid->CE: NDCG@5 0.613957, NDCG@10 0.553624, MRR 0.835182, Recall@50 0.353951, Recall@500 0.408111. Hybrid->CE vs Weighted Hybrid dNDCG@10 = -0.102650. CE-A reranks Hybrid top 100.

Interpretation: CE-A regresses in this cascade, consistent with objective/domain mismatch.

## 8 CE-B Results

Observed result: Hybrid->LambdaMART->CE: NDCG@5 0.559085, NDCG@10 0.481169, MRR 0.795038, Recall@50 0.311810, Recall@500 0.311810. Hybrid->LambdaMART->CE vs LambdaMART dNDCG@10 = -0.178024. CE-B reranks LambdaMART top 50.

Interpretation: CE-B is the largest aggregate regression.

## 9 Validated CE Correctness-Audit Summary

Observed result: audit `PASS`; missing scores 0; invalid scores 0; ordering violations A/B 0/0; membership violations A/B 0/0.

Interpretation: Negative CE results are not explained by audited import, ordering, or membership bugs.

## 10 CE Score-Distribution Findings

Observed result: global count 3156056; mean -1.160727; median -0.606986; p25 -5.366263; p75 3.257844. By label:

| Label | Count | Mean | Median | p25 | p75 |
|---|---:|---:|---:|---:|---:|
| E | 120908 | 4.516645 | 5.427917 | 2.765439 | 7.083426 |
| S | 74048 | 2.779884 | 3.479633 | 0.547910 | 5.644616 |
| C | 11490 | 3.746844 | 4.662795 | 1.734993 | 6.451965 |
| I | 21050 | 2.055546 | 2.884169 | -0.522941 | 5.292221 |
| UNJUDGED | 2928560 | -1.537132 | -1.053817 | -5.752994 | 2.795658 |

Interpretation: CE score magnitude is not ESCI-grade alignment. CE product input was title+description+brand, not title-only.

## 11 Rank-Displacement Findings

Observed result: Displacement is CE rank minus predecessor rank; negative means promotion.

| Cascade | Label | Count | Mean | Median | p25 | p75 | Promoted | Demoted |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ce_a_vs_hybrid | E | 119113 | -5.991 | -1.000 | -15.000 | 4.000 | 0.545 | 0.376 |
| ce_a_vs_hybrid | S | 73192 | -5.024 | -1.000 | -17.000 | 8.000 | 0.532 | 0.427 |
| ce_a_vs_hybrid | C | 11372 | -6.300 | -2.000 | -16.000 | 5.000 | 0.556 | 0.389 |
| ce_a_vs_hybrid | I | 20825 | -5.305 | -2.000 | -18.000 | 8.000 | 0.537 | 0.420 |
| ce_a_vs_hybrid | UNJUDGED | 2759898 | 0.458 | 1.000 | -20.000 | 22.000 | 0.474 | 0.510 |
| ce_b_vs_lambdamart | E | 94003 | -3.051 | -1.000 | -8.000 | 3.000 | 0.516 | 0.374 |
| ce_b_vs_lambdamart | S | 50662 | -3.094 | -1.000 | -10.000 | 4.000 | 0.536 | 0.403 |
| ce_b_vs_lambdamart | C | 8097 | -3.733 | -1.000 | -10.000 | 3.000 | 0.544 | 0.380 |
| ce_b_vs_lambdamart | I | 13396 | -3.658 | -1.000 | -11.000 | 4.000 | 0.545 | 0.386 |
| ce_b_vs_lambdamart | UNJUDGED | 1326042 | 0.394 | 1.000 | -10.000 | 11.000 | 0.466 | 0.504 |

Interpretation: Promotions alone do not imply improved aggregate ordering.

## 12 Query-Slice Findings

Observed result: low lexical overlap H->CE dNDCG@10 = -0.140717 and H->L->CE dNDCG@10 = -0.195133; high disagreement H->CE dNDCG@10 = -0.138625 and H->L->CE dNDCG@10 = -0.224976. Navigational slice is NOT_IMPLEMENTED.

- query_length: worst H->CE short_1_2 (-0.107428); worst H->L->CE short_1_2 (-0.186257).
- lexical_overlap: worst H->CE low (-0.140717); worst H->L->CE low (-0.195133).
- bm25_dense_disagreement: worst H->CE high_disagreement (-0.138625); worst H->L->CE high_disagreement (-0.224976).
- navigational: NOT_IMPLEMENTED (No defensible navigational intent label exists in saved M3 artifacts.)

Interpretation: These patterns are consistent with objective/domain mismatch, without causal overclaiming.

## 13 Quality-Latency Findings

Observed result: CE batch throughput is 181042.723 pairs/sec on NVIDIA A100-SXM4-40GB; online CE p50/p95 latency is NA. Hardware and stage scope differ; CE has imported A100 batch throughput but no measured online request latency.

Interpretation: This is hardware-mixed and stage-mixed, not a serving benchmark.

## 14 Representative Wins

Observed result: see `/Users/yash/Documents/AdaptiRank/artifacts/ranking/dda38161938e829f2c2fc9b73d40d6cf922a5470c3b45bf176f742ee0ca7c667/m3_three_split/cross_encoder/analysis/representative_examples.md` sections `ce_a_largest_wins_vs_hybrid` and `ce_b_largest_wins_vs_lambdamart`.

Interpretation: These are illustrative examples, not aggregate proof.

## 15 Representative Failures

Observed result: see `/Users/yash/Documents/AdaptiRank/artifacts/ranking/dda38161938e829f2c2fc9b73d40d6cf922a5470c3b45bf176f742ee0ca7c667/m3_three_split/cross_encoder/analysis/representative_examples.md` sections `ce_a_largest_losses_vs_hybrid`, `ce_b_largest_losses_vs_lambdamart`, `high_scoring_I`, `low_scoring_E`, `S_C_inversion`, and `keyword_compatibility_or_product_type_mismatch`.

Interpretation: Failures are consistent with objective/domain mismatch and product-type confusion, not causal proof.

## 16 Negative Results

Observed result: both CE cascades reduce test NDCG@10 versus immediate predecessors. CE-A -0.102650; CE-B -0.178024. CE-score feature ablation is `DOCUMENTED_ONLY`.

Interpretation: The pretrained CE is not a drop-in M3 improvement.

## 17 Caveats And Limitations

Observed result: CE inference was imported, fixed, and not rerun; the CE is not ESCI/e-commerce tuned; no M4+ online components are included.

Interpretation: Conclusions are offline M3 ranking conclusions only.

## 18 Open Questions / Optional Follow-Up Experiments

Observed result: Open options include ESCI fine-tuning, CE calibration, CE-score feature ablation, depth sweeps, and focused low-overlap/high-disagreement analysis.

Interpretation: These are follow-ups, not requirements for this closeout.

## 19 Exact Paths To Supporting Artifacts/Reports

Observed result: final table `/Users/yash/Documents/AdaptiRank/artifacts/ranking/dda38161938e829f2c2fc9b73d40d6cf922a5470c3b45bf176f742ee0ca7c667/m3_three_split/cross_encoder/analysis/final_comparison_table.md`; audit `/Users/yash/Documents/AdaptiRank/artifacts/ranking/dda38161938e829f2c2fc9b73d40d6cf922a5470c3b45bf176f742ee0ca7c667/m3_three_split/cross_encoder/analysis/ce_correctness_audit.json`; score by label `/Users/yash/Documents/AdaptiRank/artifacts/ranking/dda38161938e829f2c2fc9b73d40d6cf922a5470c3b45bf176f742ee0ca7c667/m3_three_split/cross_encoder/analysis/score_distribution_by_label.md`; displacement `/Users/yash/Documents/AdaptiRank/artifacts/ranking/dda38161938e829f2c2fc9b73d40d6cf922a5470c3b45bf176f742ee0ca7c667/m3_three_split/cross_encoder/analysis/rank_displacement_summary.md`; slices `/Users/yash/Documents/AdaptiRank/artifacts/ranking/dda38161938e829f2c2fc9b73d40d6cf922a5470c3b45bf176f742ee0ca7c667/m3_three_split/cross_encoder/analysis/query_slice_summary.md`; quality latency `/Users/yash/Documents/AdaptiRank/artifacts/ranking/dda38161938e829f2c2fc9b73d40d6cf922a5470c3b45bf176f742ee0ca7c667/m3_three_split/cross_encoder/analysis/quality_latency_summary.md`; representative examples `/Users/yash/Documents/AdaptiRank/artifacts/ranking/dda38161938e829f2c2fc9b73d40d6cf922a5470c3b45bf176f742ee0ca7c667/m3_three_split/cross_encoder/analysis/representative_examples.md`; source JSON `/Users/yash/Documents/AdaptiRank/artifacts/ranking/dda38161938e829f2c2fc9b73d40d6cf922a5470c3b45bf176f742ee0ca7c667/m3_three_split/cross_encoder/analysis/m3_findings_source.json`.

Full final comparison:

| Method | Role | Queries | NDCG@5 | NDCG@10 | MRR | Recall@50 | Recall@500 | Depth |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 | retrieval baseline | 8956 | 0.641576 | 0.609906 | 0.848967 | 0.298437 | 0.527779 | 500 |
| Dense | retrieval baseline | 8956 | 0.617645 | 0.579609 | 0.837741 | 0.248115 | 0.486403 | 500 |
| Weighted Hybrid | retrieval fusion | 8956 | 0.680388 | 0.656273 | 0.880465 | 0.325851 | 0.589773 | 500 |
| RRF | retrieval fusion | 8956 | 0.678732 | 0.654843 | 0.879373 | 0.328005 | 0.588507 | 500 |
| Pointwise | learned reranker | 8956 | 0.683448 | 0.658381 | 0.883380 | 0.291320 | 0.589773 | 500 |
| LambdaMART | learned reranker | 8956 | 0.685001 | 0.659193 | 0.881553 | 0.311810 | 0.589773 | 500 |
| Hybrid->CE | cross-encoder reranker | 8956 | 0.613957 | 0.553624 | 0.835182 | 0.353951 | 0.408111 | 100 |
| Hybrid->LambdaMART->CE | cascade reranker | 8956 | 0.559085 | 0.481169 | 0.795038 | 0.311810 | 0.311810 | 50 |

Interpretation: These generated artifacts support the values in this report.
