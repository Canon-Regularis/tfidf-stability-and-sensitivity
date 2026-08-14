#!/usr/bin/env python3
"""The whole pipeline of README sections 2 and 3, on seven documents, in one file.

A ranking is a list of documents together with the margins that say how far it
stands from being a different list, and those margins are what sections 4.4 and
4.5 reason about; a top-k printed without them is the artefact this repository
argues is misleading.

Seven inline documents are enough to contain both regimes the paper separates,
side by side in a single ranking:

* a boundary with a small but strictly positive margin, which certifies a
  tolerance: any uniform perturbation below ``m_k / 2`` leaves the top-k set
  alone (section 4.4, research question A1); and
* a boundary with a margin of exactly zero, where the scores certify nothing and
  membership is settled entirely by the deterministic tie-break, with no
  numerical error involved (section 4.5, research question A2).

The two co-occurring at this scale is why they are studied separately rather than
as one notion of "stability", and why no dataset is needed to see the phenomenon.

The run also prints every intermediate quantity section 1.2 requires (tf, df,
idf, weight, norm) for one document.

Nothing is loaded from disk. Run it with::

    python examples/minimal_corpus_demo.py
"""

from __future__ import annotations

import math
import sys
import textwrap
from pathlib import Path
from typing import Any, Final

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tfidf_stability.preprocessing.ngrams import JOINER  # noqa: E402
from tfidf_stability.preprocessing.pipeline import (  # noqa: E402
    PreprocessedDocument,
    PreprocessingPipeline,
)
from tfidf_stability.preprocessing.tokenise import GAP  # noqa: E402
from tfidf_stability.ranking.attributes import AttributeTable  # noqa: E402
from tfidf_stability.ranking.margins import boundary_margin, min_adjacent_margin_top  # noqa: E402
from tfidf_stability.ranking.ranker import Ranking, rank_all_operators  # noqa: E402
from tfidf_stability.similarity.cosine import cosine_against_corpus  # noqa: E402
from tfidf_stability.vectorisation.idf import LogImpl, smoothed_idf_one  # noqa: E402
from tfidf_stability.vectorisation.sparse import SparseVector, l2_norm  # noqa: E402
from tfidf_stability.vectorisation.tfidf import TfidfModel, TfidfVectoriser  # noqa: E402

# Written out rather than loaded, so every number below traces back to text
# visible on this page. The three secondary attributes are section 2.3.1's tuple;
# `rating` is carried as G8's exact integer pair (2 * sum of ratings, count), so
# no float enters the tie-break.
CORPUS: Final[tuple[dict[str, Any], ...]] = (
    {
        "doc_id": "d1",
        "text": "Numerical stability of sparse matrix computations, and stability under rounding",
        "popularity": 100,
        "rating_sum2": 9,
        "rating_count": 2,
        "engagement": 4,
    },
    {
        "doc_id": "d2",
        "text": "Sparse matrix algorithms and their numerical conditioning",
        "popularity": 93,
        "rating_sum2": 17,
        "rating_count": 4,
        "engagement": 11,
    },
    {
        "doc_id": "d3",
        "text": "Perturbation analysis for numerical linear algebra",
        "popularity": 86,
        "rating_sum2": 8,
        "rating_count": 2,
        "engagement": 6,
    },
    {
        "doc_id": "d4",
        "text": "Cosine similarity between sparse document vectors",
        "popularity": 79,
        "rating_sum2": 15,
        "rating_count": 4,
        "engagement": 2,
    },
    # The last three share no term with the query and score exactly 0, the source
    # of the exact-tie block; short documents with no in-vocabulary overlap are
    # the common case in real catalogues. Their attributes disagree: popularity,
    # identifier and engagement each induce a different order on the three, so the
    # block exhibits section 4.5 rather than merely permitting it.
    {
        "doc_id": "d5",
        "text": "Ranking documents by cosine similarity scores",
        "popularity": 58,
        "rating_sum2": 12,
        "rating_count": 3,
        "engagement": 3,
    },
    {
        "doc_id": "d6",
        "text": "A gentle introduction to baking sourdough bread",
        "popularity": 65,
        "rating_sum2": 19,
        "rating_count": 4,
        "engagement": 9,
    },
    {
        "doc_id": "d7",
        "text": "Sourdough bread baking, a gentle introduction",
        "popularity": 72,
        "rating_sum2": 7,
        "rating_count": 2,
        "engagement": 5,
    },
)

QUERY: Final = "numerical stability of sparse matrices"

#: The rank whose margins are read in detail. Small enough that the whole
#: ranking is on screen beside it.
K: Final = 3

#: n-grams are stored joined by U+001F, which is unprintable; swapped for this
#: only when displaying. See the injectivity argument in `preprocessing/ngrams.py`.
SHOWN_JOINER: Final = "+"

WIDTH: Final = 78


def show(token: str) -> str:
    """Render a stored feature legibly, keeping n-gram structure visible."""
    return token.replace(JOINER, SHOWN_JOINER).replace(GAP, "<gap>")


def rule(title: str) -> None:
    print(f"\n{'-' * WIDTH}\n {title}\n{'-' * WIDTH}")


def para(text: str) -> None:
    """Print a commentary paragraph, rewrapped.

    Several embed measured values whose width is unknown until the run, so a
    hand-wrapped string goes ragged the moment a number changes.
    """
    print("\n" + textwrap.fill(" ".join(text.split()), width=WIDTH))


def print_corpus() -> None:
    rule("0. The corpus (written inline; nothing is loaded)")
    print(f"{len(CORPUS)} documents, with the section 2.3.1 tie-break attributes:\n")
    print(f"  {'id':<4} {'popularity':>10} {'rating':>8} {'engagement':>10}  text")
    for r in CORPUS:
        mean = r["rating_sum2"] / (2 * r["rating_count"])
        print(
            f"  {r['doc_id']:<4} {r['popularity']:>10} {mean:>8.2f} "
            f"{r['engagement']:>10}  {r['text']}"
        )
    print(f'\nQuery: "{QUERY}"')


def print_preprocessing(doc: PreprocessedDocument) -> None:
    rule("1. The fixed preprocessing map (section 2)")
    print("normalise -> tokenise -> drop stopwords -> lemmatise -> n-grams, applied to d1:\n")
    # A stopword leaves a sentinel behind, so the lemma stream is punctured
    # rather than shortened; hence the separate gap count.
    n_gaps = sum(1 for t in doc.lemmas if t == GAP)
    rows = (
        ("raw tokens", f"({len(doc.raw_tokens)})", doc.raw_tokens),
        ("lemmas", f"({len(doc.lemmas) - n_gaps} + {n_gaps} gaps)", doc.lemmas),
        ("features", f"({len(doc.features)})", doc.features),
    )
    width = max(len(count) for _, count, _ in rows) + 2
    for label, count, stream in rows:
        print(f"  {label:<12}{count:<{width}}{' '.join(show(t) for t in stream)}")
    para(
        "Stopwords leave a <gap> that no n-gram may cross, so 'stability under "
        "rounding' yields no bigram. Bigrams are shown joined by '+'; the stored "
        "joiner is U+001F, which cannot occur inside a token, so the bigram "
        "spars+matrix stays distinct from any single token spelt 'spars matrix'."
    )


def print_vocabulary(model: TfidfModel) -> None:
    rule("2. Vocabulary, document frequency and IDF (section 2.1)")
    N = model.vocabulary.n_documents
    print(f"N = {N} documents, |V| = {model.n_features} features (unigrams and bigrams).")
    print("\n  idf(t) = log((1 + N) / (1 + df(t))) + 1\n")
    print(f"  {'df':>3}  {'idf':<20} tokens with this df")
    for df in sorted(set(model.vocabulary.df)):
        n_tokens = sum(1 for d in model.vocabulary.df if d == df)
        print(f"  {df:>3}  {model.idf[model.vocabulary.df.index(df)]:<20.17g} {n_tokens}")
    para(
        "Monotone decay in df, and idf >= 1 everywhere thanks to the additive "
        "constant, so no term is ever annihilated -- not even at df = N."
    )

    # log is the one transcendental in the pipeline and IEEE-754 leaves it free to
    # round however the libm likes, so the disagreement is measured here.
    disagree = [
        df
        for df in range(N + 1)
        if smoothed_idf_one(df, N, LogImpl.CORRECTLY_ROUNDED)
        != smoothed_idf_one(df, N, LogImpl.PLATFORM)
    ]
    present = sorted(set(model.vocabulary.df))
    overlap = sorted(set(disagree) & set(present))
    reach = (
        f"including df {overlap}, which this corpus does contain"
        if overlap
        else f"none of them among the df this corpus contains ({present})"
    )
    para(
        f"idf is computed with the correctly-rounded logarithm ({model.idf.log_impl}), "
        f"because IEEE-754 mandates correct rounding for + - * / sqrt but not for log, "
        f"and platform libms therefore disagree. Measured here over df = 0 .. N, the "
        f"platform log differs at {len(disagree)} of {N + 1} values ({disagree}), "
        f"{reach}. At corpus scale the disagreement reaches ~15% of idf entries, and it "
        f"takes every weight, norm and score with it (spec_addenda.md#g13)."
    )


def print_intermediates(model: TfidfModel, i: int) -> None:
    rule("3. Every intermediate for one document (section 1.2)")
    inter = model.intermediates(i)
    L = inter["in_vocabulary_length"]
    terms = inter["terms"]
    print(
        f"{inter['doc_id']}: L = {L} in-vocabulary feature occurrences over "
        f"{len(terms)} distinct features.\n"
        f"tf_i(t) = count_i(t) / L,   w_i(t) = tf_i(t) * idf(t)\n"
    )
    print(f"  {'feature':<16} {'df':>3} {'idf':>10} {'tf':>10} {'w = tf*idf':>12}")
    for t in terms:
        print(
            f"  {show(t['token']):<16} {t['df']:>3} {t['idf']:>10.6f} "
            f"{t['tf']:>10.6f} {t['weight']:>12.6f}"
        )

    # L is an exact occurrence count, so the tf column sums to 1 in the reals.
    # Shown under both the left-to-right fold of section 2.3 and exact summation;
    # binary64 addition is non-associative, so the two may disagree.
    naive_tf = sum(t["tf"] for t in terms)
    exact_tf = math.fsum(t["tf"] for t in terms)
    print(f"\n  sum of tf   {naive_tf!r} left to right, {exact_tf!r} exact  (1 in the reals)")
    recomputed = l2_norm(model.document(i), model.reduction)
    print(f"  ||w||_2     {inter['norm']!r}")
    print(
        f"  recomputed from the weights above under the same reduction "
        f"({model.reduction}): {recomputed == inter['norm']}"
    )
    para(
        "Note the one feature whose tf differs: it occurs twice in the document, so "
        "it takes twice the tf of its neighbours and -- sharing their idf -- the "
        "largest weight in the row."
    )


def print_query(doc: PreprocessedDocument, model: TfidfModel, q: SparseVector) -> None:
    rule("4. The query, embedded in the same space (section 3)")
    kept = [f for f in doc.features if f in model.vocabulary]
    dropped = [f for f in doc.features if f not in model.vocabulary]
    print(f"  features  {' '.join(show(f) for f in doc.features)}")
    print(f"  kept      {' '.join(show(f) for f in kept)}")
    print(f"  dropped   {' '.join(show(f) for f in dropped)}  (out of vocabulary)")
    para(
        "The query reuses the corpus vocabulary and the corpus IDF unchanged, so "
        "query and documents live in one space. Nothing is refitted and the "
        "vocabulary is not extended: 'matrices' stems to a form the corpus never "
        "produced, so it is simply dropped, exactly as it would be for a document."
    )
    print(
        f"\n  nnz(q) = {q.nnz} of |V| = {model.n_features},  "
        f"||q||_2 = {l2_norm(q, model.reduction):.12f}"
    )


def print_ranking(model: TfidfModel, ranking: Ranking, scores: list[float]) -> None:
    rule("5. Similarity, ranking and margins (sections 2.3.1, 2.3.2)")
    print(f"s_i = cos(q, w_i);  order under {ranking.operator}, ties broken on")
    print("(popularity, rating, engagement, identifier).\n")
    print(f"  {'rank':>4} {'doc':<4} {'score':<16} {'m_k':>14} {'flip radius m_k/2':>19}")
    for j, d in enumerate(ranking.order, start=1):
        margin = boundary_margin(ranking.sorted_scores, j, mode=ranking.strict_mode)
        # An undefined margin prints its stated reason rather than a number: G3
        # forbids coercing it to 0, which reads as an exact tie, or to infinity,
        # which reads as perfect stability.
        cells = (
            f"{margin.value:>14.6e} {margin.flip_radius:>19.6e}"
            if margin.defined
            else f"{'--':>14} {'--':>19}    ({margin.reason})"
        )
        print(f"  {j:>4} {model.doc_ids[d]:<4} {scores[d]:<16.12f} {cells}")


def print_reading(model: TfidfModel, rankings: dict[str, Ranking]) -> None:
    rule("6. What the margins say")
    ranking = rankings["pi"]
    m_k = boundary_margin(ranking.sorted_scores, K, mode=ranking.strict_mode)
    m_top = min_adjacent_margin_top(ranking.sorted_scores, K, mode=ranking.strict_mode)
    at_k, past_k = ranking.order[K - 1], ranking.order[K]

    joint = min(m_k.flip_radius, m_top.flip_radius)
    print(f"At k = {K}, over the top-k set {[model.doc_ids[d] for d in ranking.top_k(K)]}:\n")
    for label, value, gloss in (
        (f"m_{K}", m_k.value, f"boundary gap: {model.doc_ids[at_k]} over {model.doc_ids[past_k]}"),
        ("m_min^top", m_top.value, f"tightest gap strictly inside the top-{K}"),
        ("set radius", m_k.flip_radius, f"the top-{K} set survives any |ds_i| below this"),
        ("order radius", m_top.flip_radius, f"the order within the top-{K} survives this"),
        ("joint radius", joint, "both invariants at once"),
    ):
        print(f"  {label:<13} {value:.6e}   {gloss}")
    para(
        "The two radii bound disjoint sets of gaps -- m_min^top the gaps strictly "
        "inside the top-k, m_k the one at its boundary -- so neither implies the "
        "other, and a certificate quoted without saying which invariant it certifies "
        "is ambiguous."
    )

    # The other regime, in the same ranking: an exact tie is the absence of a
    # margin, and no amount of numerical care recovers one.
    lowest = ranking.sorted_scores[-1]
    tied = [d for d in range(model.n_documents) if ranking.scores[d] == lowest]
    if len(tied) > 1:
        first = ranking.n_documents - len(tied) + 1
        para(
            f"From rank {first} down, {len(tied)} documents score exactly {lowest:.1f}: "
            f"they share no feature with the query. So m_k = 0 for k = {first} .. "
            f"{ranking.n_documents - 1}, and there the margins certify nothing at all. "
            f"The order of those {len(tied)} is not a similarity result -- it is whatever "
            f"the tie-break says, and the operators of section 4.5 do not agree:"
        )
        print()
        for name, other in rankings.items():
            print(f"  {name:<9} {[model.doc_ids[d] for d in other.order_within(tied)]}")
        # Counted rather than asserted, so the sentence stays true if the set of
        # operators or the attribute values change.
        distinct = len({other.order_within(tied) for other in rankings.values()})
        para(
            f"Identical scores, to the last bit; {distinct} distinct orders from "
            f"{len(rankings)} operators. That discontinuity is research question A2, and "
            f"it is independent of every numerical concern above it -- ds = 0 and the "
            f"outcome still moves. examples/tie_break_discontinuity_demo.py takes it "
            f"further."
        )


def main() -> int:
    pipeline = PreprocessingPipeline()
    documents = [pipeline.preprocess_document(str(r["doc_id"]), str(r["text"])) for r in CORPUS]
    features = [list(d.features) for d in documents]
    doc_ids = [str(r["doc_id"]) for r in CORPUS]

    model = TfidfVectoriser().fit(features, doc_ids)
    table = AttributeTable.from_records(CORPUS)

    query = pipeline.preprocess_document("query", QUERY)
    q = TfidfVectoriser.transform_query(list(query.features), model)
    scores = cosine_against_corpus(
        q, [model.document(i) for i in range(model.n_documents)], model.norms
    )
    rankings = rank_all_operators(scores, table, n_zero_norm_docs=len(model.zero_norm_documents))

    print(f"\n{'=' * WIDTH}\n A minimal TF-IDF corpus, end to end\n{'=' * WIDTH}")
    print_corpus()
    print_preprocessing(documents[0])
    print_vocabulary(model)
    print_intermediates(model, 0)
    print_query(query, model, q)
    print_ranking(model, rankings["pi"], scores)
    print_reading(model, rankings)
    print(f"\nmodel digest {model.digest()[:16]}...  (every weight, bit-exactly)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
