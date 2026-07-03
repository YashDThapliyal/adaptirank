# Data

The primary retrieval/ranking backbone is Amazon Shopping Queries (ESCI). The canonical config
uses `small_version == 1` and `product_locale == "us"`, matching Task 1. The larger
`large_version == 1` path is available only through its explicit config.

Source identity is preserved with raw `example_id`, `query_id`, `product_id`, locale, split,
small/large flags, and `esci_label`. `(product_locale, product_id)` is the source product key;
query grouping and leakage checks use `(product_locale, query_id)`. SHA-256 internal keys are
derived conveniences and never replace raw IDs.

Numeric grades are derived from the configurable mapping `E/S/C/I -> 3/2/1/0`. Missing
query-product judgments remain null with `judgment_status = "unjudged"`; absence is unknown, not
an irrelevant judgment. Deterministic background catalog products therefore never receive
fabricated negative labels.

Official provenance records the pinned Amazon commit, exact immutable URL, observed size, and
locally observed SHA-256. Local hashes are integrity fingerprints, not authoritative checksums
unless independently published expected values are available and matched.

The verified official source revision is
`7916cdf6ab75a462e77f20ab40428a10923998d5`. The `official-sample` integration downloads all
source bytes, then processes 300 source-train and 100 source-test query groups. This configuration
is integration evidence only; full benchmark claims require `configs/data/esci_small_us.yaml`.

Open Bandit, Criteo, and KuaiRand/KuaiRec remain `NOT_RUN` external tracks and are never merged
into a fabricated log.
