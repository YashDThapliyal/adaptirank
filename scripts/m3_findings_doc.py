"""Expanded M3 findings document renderer."""
# ruff: noqa: E501

from __future__ import annotations

from pathlib import Path
from typing import Any

LABELS = ("E", "S", "C", "I", "UNJUDGED")


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def _repo_path(paths: dict[str, Path], key: str, *parts: str) -> str:
    value = paths[key].joinpath(*parts)
    root = paths.get("root")
    if root is not None:
        try:
            return str(value.relative_to(root))
        except ValueError:
            pass
    return str(value)


def _row_delta(
    rows: dict[str, dict[str, Any]], method: str, base: str, metric: str
) -> float | None:
    method_value = rows[method].get(metric)
    base_value = rows[base].get(metric)
    if method_value is None or base_value is None:
        return None
    return float(method_value) - float(base_value)


def _delta_for_min(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    return float("inf") if value is None else float(value)


def _score_table(summary: dict[str, dict[str, Any]]) -> str:
    lines = [
        "| Label | Count | Mean | Median | p25 | p75 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        row = summary[label]
        lines.append(
            f"| {label} | {row['count']} | {fmt(row['mean'])} | "
            f"{fmt(row['median'])} | {fmt(row['p25'])} | {fmt(row['p75'])} |"
        )
    return "\n".join(lines)


def _disp_table(summary: dict[str, Any]) -> str:
    lines = [
        "| Cascade | Label | Count | Mean | Median | p25 | p75 | Promoted | Demoted |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cascade, payload in summary.items():
        for label in LABELS:
            row = payload["by_label"][label]
            lines.append(
                f"| {cascade} | {label} | {row['count']} | {fmt(row['mean'], 3)} | "
                f"{fmt(row['median'], 3)} | {fmt(row['p25'], 3)} | "
                f"{fmt(row['p75'], 3)} | {fmt(row['pct_promoted'], 3)} | "
                f"{fmt(row['pct_demoted'], 3)} |"
            )
    return "\n".join(lines)


def _metric(row: dict[str, Any]) -> str:
    return (
        f"{row['method']}: NDCG@5 {fmt(row['ndcg_5'])}, "
        f"NDCG@10 {fmt(row['ndcg_10'])}, MRR {fmt(row['mrr'])}, "
        f"Recall@10 {fmt(row['recall_10'])}, Recall@50 {fmt(row['recall_50'])}, "
        f"Recall@100 {fmt(row['recall_100'])}, Recall@500 {fmt(row['recall_500'])}."
    )


def _slice_notes(slices: dict[str, Any]) -> str:
    notes = []
    ce_a_key = "hybrid_to_cross_encoder_vs_weighted_hybrid__ndcg_10_delta"
    ce_b_key = "hybrid_to_lambdamart_to_cross_encoder_vs_lambdamart__ndcg_10_delta"
    for family, payload in slices.items():
        if payload.get("status"):
            notes.append(f"- {family}: {payload['status']} ({payload['reason']})")
            continue
        if not payload.get("rows"):
            notes.append(f"- {family}: no slice rows available.")
            continue
        worst_a = min(payload["rows"], key=lambda row: _delta_for_min(row, ce_a_key))
        worst_b = min(payload["rows"], key=lambda row: _delta_for_min(row, ce_b_key))
        notes.append(
            f"- {family}: worst H->CE {worst_a['slice']} ({fmt(worst_a[ce_a_key])}); "
            f"worst H->L->CE {worst_b['slice']} ({fmt(worst_b[ce_b_key])})."
        )
    return "\n".join(notes)


def _final_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Method | Role | Queries | NDCG@5 | NDCG@10 | MRR | Recall@10 | Recall@50 | "
        "Recall@100 | Recall@500 | Depth |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['model_role']} | {row['queries']} | "
            f"{fmt(row['ndcg_5'])} | {fmt(row['ndcg_10'])} | {fmt(row['mrr'])} | "
            f"{fmt(row['recall_10'])} | {fmt(row['recall_50'])} | "
            f"{fmt(row['recall_100'])} | {fmt(row['recall_500'])} | "
            f"{row['candidate_depth']} |"
        )
    return "\n".join(lines)


def expanded_findings_doc(
    paths: dict[str, Path], source: dict[str, Any], rows: list[dict[str, Any]]
) -> str:
    mapped = {row["method_key"]: row for row in rows}
    audit = source["audit"]
    score_dist = source["score_distribution"]["global"]
    by_label = source["score_distribution_by_label"]
    disp = source["displacement_summary"]
    slices = source["query_slice_summary"]
    latency = source["quality_latency"]
    ablate = source["feature_ablation"]
    ce_a_delta = _row_delta(mapped, "hybrid_to_cross_encoder", "weighted_hybrid", "ndcg_10")
    ce_b_delta = _row_delta(
        mapped, "hybrid_to_lambdamart_to_cross_encoder", "lambdamart", "ndcg_10"
    )
    lm_delta = _row_delta(mapped, "lambdamart", "weighted_hybrid", "ndcg_10")
    pw_delta = _row_delta(mapped, "pointwise", "lambdamart", "ndcg_10")
    sec = []
    sec.append(
        (
            "1 M3 Research Questions",
            "Observed result: M3 evaluates offline ranking baselines, learned rerankers, imported CE cascades, score behavior, slices, examples, and quality-latency from fixed artifacts.\n\nInterpretation: This does not answer online ads, bandit, OPE, auction, or RL questions.",
        )
    )
    sec.append(
        (
            "2 Dataset And Evaluation Policy",
            "Observed result: Aggregate values and query slices use the test split. Representative examples are test-only, with query text joined from `queries.parquet` and product title/brand from `catalog.parquet`. UNJUDGED is separate from I.\n\nInterpretation: Judged ranking metrics and raw-depth recall answer different questions.",
        )
    )
    sec.append(
        (
            "3 Canonical Artifact Identifiers/Fingerprints Relevant To M3",
            f"Observed result: dataset fingerprint `{source['dataset_fingerprint']}`. CE root `{_repo_path(paths, 'ce')}`. Analysis root `{_repo_path(paths, 'analysis')}`. Pair union SHA `{source['coverage']['pair_union_sha256']}`; scores SHA `{source['coverage']['scores_sha256']}`.\n\nInterpretation: Imported CE score artifacts were not modified and CE inference was not rerun.",
        )
    )
    sec.append(
        (
            "4 Baseline Results",
            "Observed result:\n\n"
            + "\n".join(_metric(mapped[k]) for k in ("bm25", "dense", "weighted_hybrid", "rrf"))
            + "\n\nInterpretation: Weighted Hybrid is the strongest retrieval/fusion baseline by test NDCG@10.",
        )
    )
    sec.append(
        (
            "5 Pointwise Results",
            f"Observed result: {_metric(mapped['pointwise'])} Pointwise vs LambdaMART dNDCG@10 = {fmt(pw_delta)}.\n\nInterpretation: Pointwise is competitive but does not beat LambdaMART on aggregate NDCG@10.",
        )
    )
    sec.append(
        (
            "6 LambdaMART Results",
            f"Observed result: {_metric(mapped['lambdamart'])} LambdaMART vs Weighted Hybrid dNDCG@10 = {fmt(lm_delta)}.\n\nInterpretation: LambdaMART is the best aggregate M3 method by test NDCG@10.",
        )
    )
    sec.append(
        (
            "7 CE-A Results",
            f"Observed result: {_metric(mapped['hybrid_to_cross_encoder'])} Hybrid->CE vs Weighted Hybrid dNDCG@10 = {fmt(ce_a_delta)}. CE-A reranks Hybrid top 100.\n\nInterpretation: CE-A regresses in this cascade, consistent with objective/domain mismatch.",
        )
    )
    sec.append(
        (
            "8 CE-B Results",
            f"Observed result: {_metric(mapped['hybrid_to_lambdamart_to_cross_encoder'])} Hybrid->LambdaMART->CE vs LambdaMART dNDCG@10 = {fmt(ce_b_delta)}. CE-B reranks LambdaMART top 50.\n\nInterpretation: CE-B is the largest aggregate regression.",
        )
    )
    sec.append(
        (
            "9 Validated CE Correctness-Audit Summary",
            f"Observed result: audit `{audit['status']}`; missing scores {audit['pair_equality']['missing_scores_for_union_pairs']}; invalid scores {audit['pair_equality']['invalid_score_values']}; ordering violations A/B {audit['ordering']['ce_a_score_desc_product_key_asc_violations']}/{audit['ordering']['ce_b_score_desc_product_key_asc_violations']}; membership violations A/B {audit['membership']['ce_a_non_hybrid_top_100_rows']}/{audit['membership']['ce_b_non_lambdamart_top_50_rows']}.\n\nInterpretation: Negative CE results are not explained by audited import, ordering, or membership bugs.",
        )
    )
    sec.append(
        (
            "10 CE Score-Distribution Findings",
            f"Observed result: global count {score_dist['count']}; mean {score_dist['mean']:.6f}; median {score_dist['quantiles']['p50']:.6f}; p25 {score_dist['quantiles']['p25']:.6f}; p75 {score_dist['quantiles']['p75']:.6f}. By label:\n\n{_score_table(by_label)}\n\nInterpretation: CE score magnitude is not ESCI-grade alignment. CE product input was title+description+brand, not title-only.",
        )
    )
    sec.append(
        (
            "11 Rank-Displacement Findings",
            "Observed result: Displacement is CE rank minus predecessor rank; negative means promotion.\n\n"
            + _disp_table(disp)
            + "\n\nInterpretation: Promotions alone do not imply improved aggregate ordering.",
        )
    )
    sec.append(
        (
            "12 Query-Slice Findings",
            f"Observed result: see slice-by-slice deltas below. Navigational slice is NOT_IMPLEMENTED.\n\n{_slice_notes(slices)}\n\nInterpretation: These patterns are consistent with objective/domain mismatch, without causal overclaiming.",
        )
    )
    sec.append(
        (
            "13 Quality-Latency Findings",
            f"Observed result: CE batch throughput is {source['scoring_stats']['pairs_per_second']:.3f} pairs/sec on {latency['methods']['hybrid_to_cross_encoder']['hardware']}; online CE p50/p95 latency is NA. {latency['caveat']}\n\nInterpretation: This is hardware-mixed and stage-mixed, not a serving benchmark.",
        )
    )
    sec.append(
        (
            "14 Representative Wins",
            f"Observed result: see `{_repo_path(paths, 'analysis', 'representative_examples.md')}` sections `ce_a_largest_wins_vs_hybrid` and `ce_b_largest_wins_vs_lambdamart`.\n\nInterpretation: These are illustrative examples, not aggregate proof.",
        )
    )
    sec.append(
        (
            "15 Representative Failures",
            f"Observed result: see `{_repo_path(paths, 'analysis', 'representative_examples.md')}` sections `ce_a_largest_losses_vs_hybrid`, `ce_b_largest_losses_vs_lambdamart`, `high_scoring_I`, `low_scoring_E`, `S_C_inversion`, and `keyword_compatibility_or_product_type_mismatch`.\n\nInterpretation: Failures are consistent with objective/domain mismatch and product-type confusion, not causal proof.",
        )
    )
    sec.append(
        (
            "16 Negative Results",
            f"Observed result: both CE cascades reduce test NDCG@10 versus immediate predecessors. CE-A {fmt(ce_a_delta)}; CE-B {fmt(ce_b_delta)}. CE-score feature ablation is `{ablate['status']}`.\n\nInterpretation: The pretrained CE is not a drop-in M3 improvement.",
        )
    )
    sec.append(
        (
            "17 Caveats And Limitations",
            "Observed result: CE inference was imported, fixed, and not rerun; the CE is not ESCI/e-commerce tuned; no M4+ online components are included.\n\nInterpretation: Conclusions are offline M3 ranking conclusions only.",
        )
    )
    sec.append(
        (
            "18 Open Questions / Optional Follow-Up Experiments",
            "Observed result: Open options include ESCI fine-tuning, CE calibration, CE-score feature ablation, depth sweeps, and focused low-overlap/high-disagreement analysis.\n\nInterpretation: These are follow-ups, not requirements for this closeout.",
        )
    )
    sec.append(
        (
            "19 Exact Paths To Supporting Artifacts/Reports",
            f"Observed result: final table `{_repo_path(paths, 'analysis', 'final_comparison_table.md')}`; audit `{_repo_path(paths, 'analysis', 'ce_correctness_audit.json')}`; score by label `{_repo_path(paths, 'analysis', 'score_distribution_by_label.md')}`; displacement `{_repo_path(paths, 'analysis', 'rank_displacement_summary.md')}`; slices `{_repo_path(paths, 'analysis', 'query_slice_summary.md')}`; quality latency `{_repo_path(paths, 'analysis', 'quality_latency_summary.md')}`; representative examples `{_repo_path(paths, 'analysis', 'representative_examples.md')}`; source JSON `{_repo_path(paths, 'analysis', 'm3_findings_source.json')}`.\n\nFull final comparison:\n\n{_final_table(rows)}\n\nInterpretation: These generated artifacts support the values in this report.",
        )
    )
    return "# M3 Cross-Encoder Findings\n\n" + "\n\n".join(
        f"## {heading}\n\n{text}" for heading, text in sec
    )
