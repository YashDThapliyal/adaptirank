# Limitations

- Public relevance data does not contain production auctions, budgets, or counterfactuals.
- Fixture and sampled integration statistics are not research findings.
- Hosted CI cannot be claimed until the workflow runs on a connected remote.
- Full BM25 evaluation uses incomplete ESCI judgments. Recall uses known relevant judgments;
  condensed MRR/NDCG ignore unjudged results, so these metrics should not be read as exhaustive
  relevance assessments of the broad catalog.
- Current dense and hybrid benchmark evidence is blocked on downloading the pinned pretrained
  model; no substitute synthetic result is reported.
- Simulator, OPE overlap, auction simplification, reward misspecification, and external validity
  remain future concerns rather than completed work.
