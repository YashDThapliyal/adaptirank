# Architecture

The canonical project name is **AdaptiRank** and the Python namespace is `adaptirank`.

M0–M1 implement shared configuration, provenance, run artifacts, canonical domain types, and
ESCI ingestion. Later components remain independently replaceable namespaces: retrieval,
ranking, response prediction, simulator, bandit policy, logging policy, OPE estimator, auction
mechanism, RL policy, multi-agent environment, evaluation, and serving.

No later-stage algorithm is implemented in the current milestone.

## M2 retrieval boundary

`adaptirank.retrieval.Retriever` separates index construction from top-K retrieval. Tantivy
provides persistent BM25 indexing over independently configurable title, description, and brand
fields. Sentence Transformers provides the pinned, unfine-tuned dense encoder; normalized
embeddings and a FAISS index are persisted separately. Weighted score fusion and reciprocal rank
fusion consume method candidate artifacts rather than reaching into either index.

The shared evaluator attaches judgments only after retrieval. Missing judgments remain
`unjudged`; condensed MRR/NDCG ignore unknown candidates rather than converting them to negative
labels. Hybrid output is designed as the direct M3 handoff, but M3 itself is not implemented.
