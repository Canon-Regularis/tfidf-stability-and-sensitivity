"""Every inequality in README section 4, handed to Hypothesis to break.

The distinction that matters here: these are not examples confirming the bounds,
they are adversarial searches for counterexamples. A bound that has survived a
directed search is worth considerably more than one that has been spot-checked,
and for a paper whose whole subject is perturbation behaviour, "we checked the
theorems hold" is the substantive claim.

Four inequalities are under test:

* **section 4.1** ``delta_idf`` is the difference of two exact logarithms, and
  its sign follows the competing effects of ``N`` and ``df``;
* **section 4.2** the three-term decomposition bounds ``||w' - w||``;
* **section 4.3** ``|delta cos| <= C (||du|| + ||dv||)`` with the explicit
  ``C = 1/L`` of ``spec_addenda.md#g4``;
* **section 4.4** ``eps < m_k/2`` guarantees the top-k set -- and, going beyond
  the paper, is *exactly* the radius rather than merely a safe one.
"""

from __future__ import annotations

import math
import random

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from tfidf_stability.perturbation.corpus_edits import (
    EditKind,
    add_document,
    duplicate_document,
    edit_document,
    remove_document,
)
from tfidf_stability.perturbation.idf_perturb import align_models, analyse_idf_shift
from tfidf_stability.perturbation.score_bounds import (
    certified_radius,
    flip_witness,
    is_order_stable,
    is_top_k_stable,
)
from tfidf_stability.perturbation.vector_perturb import analyse_vector_shift
from tfidf_stability.ranking.attributes import AttributeSpec, AttributeTable
from tfidf_stability.ranking.ranker import rank, sorted_scores_desc
from tfidf_stability.ranking.sort_keys import SortKeySpec
from tfidf_stability.similarity.geometry import lipschitz_constant
from tfidf_stability.vectorisation.idf import delta_idf, smoothed_idf_one
from tfidf_stability.vectorisation.sparse import SparseVector, l2_norm
from tfidf_stability.vectorisation.tfidf import TfidfVectoriser

POP = SortKeySpec("pop_only", ("popularity",))
DIM = 20
ALPHA = ["a", "b", "c", "d", "e", "f", "g", "h"]


def table_of(n: int) -> AttributeTable:
    return AttributeTable.from_records(
        [{"doc_id": f"d{i:04d}", "popularity": n - i} for i in range(n)],
        (AttributeSpec("popularity"),),
    )


def corpus_of(rng: random.Random, n: int) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    ids = tuple(f"d{i}" for i in range(n))
    docs = tuple(tuple(rng.choice(ALPHA) for _ in range(rng.randint(1, 6))) for _ in range(n))
    return ids, docs


def fit(corpus: tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]):  # type: ignore[no-untyped-def]
    ids, docs = corpus
    return TfidfVectoriser().fit(list(docs), list(ids))


# ---------------------------------------------------------------------------
# Corpus edits
# ---------------------------------------------------------------------------
def test_edits_do_not_mutate_the_original_corpus() -> None:
    """A perturbation experiment needs both sides at once."""
    original = (("a", "b"), (("x",), ("y",)))
    add_document(original, "c", ["z"])
    remove_document(original, "a")
    edit_document(original, "a", ["q"])
    assert original == (("a", "b"), (("x",), ("y",)))


def test_edit_records_carry_what_the_bounds_need() -> None:
    corpus = (("a", "b"), (("x", "y"), ("z",)))
    _, rec = edit_document(corpus, "a", ["y", "w"])
    assert rec.kind is EditKind.EDIT
    assert rec.removed_features == ("x", "y")
    assert rec.added_features == ("y", "w")
    assert rec.changes_corpus_size is False
    assert rec.touched_features == {"x", "w"}, "y is in both, so its df cannot move"


def test_add_and_remove_change_the_corpus_size() -> None:
    corpus = (("a",), (("x",),))
    assert add_document(corpus, "b", ["y"])[1].changes_corpus_size is True
    assert remove_document(corpus, "a")[1].changes_corpus_size is True


def test_duplicate_produces_an_identical_document() -> None:
    """The perturbation that manufactures an exact tie by construction."""
    corpus = (("a",), (("x", "y"),))
    new_corpus, rec = duplicate_document(corpus, "a", "a_copy")
    assert new_corpus[1][0] == new_corpus[1][1]
    assert rec.kind is EditKind.DUPLICATE


def test_duplicate_ids_are_refused() -> None:
    with pytest.raises(ValueError, match="already exists"):
        add_document((("a",), (("x",),)), "a", ["y"])


def test_editing_an_unknown_document_raises() -> None:
    with pytest.raises(KeyError):
        edit_document((("a",), (("x",),)), "zzz", ["y"])


# ---------------------------------------------------------------------------
# Section 4.1 -- IDF perturbation
# ---------------------------------------------------------------------------
def test_delta_idf_captures_the_two_competing_effects() -> None:
    """Section 4.1's point: N and df push in opposite directions."""
    assert delta_idf(3, 4, 10, 11) < 0, "the token gained a document"
    assert delta_idf(3, 3, 10, 11) > 0, "the corpus grew, the token did not"
    assert delta_idf(3, 3, 10, 10) == 0.0


@given(
    st.integers(min_value=1, max_value=200),
    st.integers(min_value=0, max_value=200),
    st.integers(min_value=0, max_value=200),
)
def test_delta_idf_equals_the_difference_of_idf_values(n: int, df: int, df2: int) -> None:
    """The ``+1`` cancels in the difference, so the two routes must agree."""
    assume(df <= n and df2 <= n)
    direct = delta_idf(df, df2, n, n)
    via = smoothed_idf_one(df2, n) - smoothed_idf_one(df, n)
    assert abs(direct - via) <= 8 * math.ulp(max(abs(direct), abs(via), 1.0))


def test_adding_a_document_moves_every_idf() -> None:
    """Even tokens whose df is unchanged move, because N moved.

    Section 4.1 says this explicitly, and it is the reason a corpus edit cannot
    be treated as a purely local perturbation.
    """
    rng = random.Random(0)
    corpus = corpus_of(rng, 8)
    before = fit(corpus)
    after = fit(add_document(corpus, "new", ["zzz_unseen"])[0])

    shift = analyse_idf_shift(before, after)
    assert shift.n_after == shift.n_before + 1
    deltas = shift.alignment.delta_idf()
    shared = shift.alignment.shared
    assert all(deltas[i] != 0.0 for i in shared), "every shared token's idf moved"


def test_vocabulary_churn_is_visible_in_the_looseness_ratio() -> None:
    """G5: a token that exists on only one side inflates ``||didf||_inf``."""
    rng = random.Random(1)
    corpus = corpus_of(rng, 6)
    stable = analyse_idf_shift(fit(corpus), fit(edit_document(corpus, "d0", ["a"])[0]))
    churned = analyse_idf_shift(
        fit(corpus), fit(add_document(corpus, "new", ["brand_new_token"])[0])
    )
    assert churned.alignment.vocabulary_changed is True
    assert churned.looseness >= stable.looseness


def test_alignment_partitions_the_union_vocabulary() -> None:
    rng = random.Random(2)
    corpus = corpus_of(rng, 6)
    before = fit(corpus)
    after = fit(add_document(corpus, "new", ["brand_new"])[0])
    a = align_models(before, after)

    covered = sorted([*a.shared, *a.gained, *a.lost])
    assert covered == list(range(a.n_tokens)), "shared, gained and lost partition the union"
    assert list(a.tokens) == sorted(a.tokens, key=lambda t: t.encode("utf-8"))


# ---------------------------------------------------------------------------
# Section 4.2 -- the three-term bound
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(12))
def test_three_term_bound_is_never_violated_under_a_real_edit(seed: int) -> None:
    """The inequality, evaluated against actual corpus perturbations.

    Randomised over all four edit kinds. The vocabulary genuinely churns here,
    which is the case section 4.2 does not cover and G5 resolves.
    """
    rng = random.Random(seed)
    corpus = corpus_of(rng, rng.randint(4, 10))
    ids = corpus[0]

    kind = rng.choice(["add", "remove", "edit", "duplicate"])
    if kind == "add":
        perturbed = add_document(corpus, "new", [rng.choice(ALPHA)])[0]
    elif kind == "remove" and len(ids) > 2:
        perturbed = remove_document(corpus, ids[-1])[0]
    elif kind == "duplicate":
        perturbed = duplicate_document(corpus, ids[0], "copy")[0]
    else:
        perturbed = edit_document(
            corpus, ids[0], [rng.choice(ALPHA) for _ in range(rng.randint(1, 5))]
        )[0]

    before, after = fit(corpus), fit(perturbed)
    shared_ids = set(before.doc_ids) & set(after.doc_ids)
    assert shared_ids

    for doc_id in sorted(shared_ids):
        result = analyse_vector_shift(before, after, doc_id)
        assert result.bound.holds, (
            f"{kind} edit, {doc_id}: observed {result.bound.observed!r} exceeded "
            f"bound {result.bound.total!r} "
            f"(local={result.bound.local!r} global={result.bound.glob!r} "
            f"interaction={result.bound.interaction!r})"
        )
        assert result.pythagoras_holds, "the union-vocabulary split must be exact"


def test_an_unperturbed_corpus_moves_nothing() -> None:
    rng = random.Random(3)
    corpus = corpus_of(rng, 6)
    model = fit(corpus)
    for doc_id in model.doc_ids:
        result = analyse_vector_shift(model, model, doc_id)
        assert result.bound.observed == 0.0
        assert result.bound.total == 0.0
        assert result.churn_fraction == 0.0


def test_a_local_edit_leaves_untouched_documents_moved_only_globally() -> None:
    """Editing one document still moves the others -- through ``idf`` alone.

    A clean instance of section 4.2's separation: for every document except the
    edited one, ``dtf`` is zero, so the local and interaction terms vanish and
    the entire bound is the global term.
    """
    rng = random.Random(4)
    corpus = corpus_of(rng, 8)
    before = fit(corpus)
    after = fit(edit_document(corpus, "d0", ["a", "a", "b"])[0])

    for doc_id in before.doc_ids:
        if doc_id == "d0":
            continue
        result = analyse_vector_shift(before, after, doc_id)
        if result.bound.observed > 0.0:
            assert result.bound.dominant_term == "global"
            assert result.bound.local == pytest.approx(0.0, abs=1e-12)


def test_analyse_vector_shift_refuses_a_document_missing_from_one_side() -> None:
    rng = random.Random(5)
    corpus = corpus_of(rng, 4)
    before = fit(corpus)
    after = fit(add_document(corpus, "new", ["a"])[0])
    with pytest.raises(KeyError, match="present in both"):
        analyse_vector_shift(before, after, "new")


# ---------------------------------------------------------------------------
# Section 4.3 -- the Lipschitz bound (G4)
# ---------------------------------------------------------------------------
nonneg = st.floats(min_value=0.0, max_value=1e3, allow_nan=False, allow_infinity=False)
sparse_map = st.dictionaries(st.integers(0, DIM - 1), nonneg, min_size=0, max_size=DIM)


def sv(mapping: dict[int, float]) -> SparseVector:
    return SparseVector.from_mapping(mapping, DIM)


@settings(suppress_health_check=[HealthCheck.filter_too_much])
@given(sparse_map, sparse_map, sparse_map, sparse_map)
def test_lipschitz_bound_survives_an_adversarial_search(
    a: dict[int, float], b: dict[int, float], c: dict[int, float], d: dict[int, float]
) -> None:
    """``C = 1/L`` where ``L`` is the smallest of the four norms (G4).

    Vectors with a tiny norm are excluded, not because the bound fails there but
    because evaluating it in binary64 becomes dominated by its own rounding --
    the same low-norm regime documented as G18.
    """
    u, v, up, vp = sv(a), sv(b), sv(c), sv(d)
    assume(min(l2_norm(x) for x in (u, v, up, vp)) > 1e-6)
    bound = lipschitz_constant(u, v, up, vp)
    assert bound.holds, (
        f"observed {bound.observed!r} exceeded uniform {bound.uniform!r} / "
        f"tight {bound.tight!r} at L={bound.min_norm!r}"
    )


def test_lipschitz_bound_applies_to_a_real_corpus_perturbation() -> None:
    """The bound in the setting it exists for, rather than on random vectors."""
    rng = random.Random(6)
    corpus = corpus_of(rng, 8)
    before = fit(corpus)
    after = fit(edit_document(corpus, "d0", ["a", "b", "c"])[0])
    alignment = align_models(before, after)

    checked = 0
    for doc_id in sorted(set(before.doc_ids) & set(after.doc_ids)):
        i, j = before.doc_ids.index(doc_id), after.doc_ids.index(doc_id)
        w = sv(dict(enumerate(alignment.embed_before(before, i))))
        w_prime = sv(dict(enumerate(alignment.embed_after(after, j))))
        query = sv(dict(enumerate(alignment.embed_before(before, 0))))
        query_after = sv(dict(enumerate(alignment.embed_after(after, 0))))
        if min(l2_norm(x) for x in (w, w_prime, query, query_after)) <= 1e-9:
            continue
        assert lipschitz_constant(query, w, query_after, w_prime).holds
        checked += 1
    assert checked > 0


# ---------------------------------------------------------------------------
# Section 4.4 -- certificates, and that the radius is exact
# ---------------------------------------------------------------------------
def test_certificate_is_half_the_margin() -> None:
    s = (1.0, 0.75, 0.5, 0.25)
    cert = certified_radius(s, 2)
    assert cert.defined is True
    assert cert.set_radius == 0.125
    assert cert.exact_tie is False


def test_certificate_at_an_exact_tie_is_no_radius_at_all() -> None:
    """``0`` here is categorically different from a small radius: membership is
    already decided entirely by the tie-break."""
    cert = certified_radius((0.5, 0.5, 0.1), 1)
    assert cert.exact_tie is True
    assert cert.set_radius == 0.0
    assert is_top_k_stable((0.5, 0.5, 0.1), 1, 0.0) is False


def test_certificate_is_undefined_at_the_end_of_the_ranking() -> None:
    cert = certified_radius((1.0, 0.5), 2)
    assert cert.defined is False
    assert math.isnan(cert.set_radius)
    assert is_top_k_stable((1.0, 0.5), 2, 0.0) is False


def test_neither_radius_dominates_the_other() -> None:
    """The two conditions of section 4.4 constrain **disjoint** sets of gaps.

    ``m_min^top`` minimises over the gaps strictly *inside* the top-k (ranks
    1->2 through (k-1)->k); ``m_k`` is the gap *at the boundary* (k->k+1). So
    neither radius bounds the other, and it is easy to construct rankings where
    each is the tighter one in turn.

    Worth pinning, because "preserving the order is harder than preserving the
    set" is an intuitive-sounding claim that happens to be false, and a
    certificate quoted without saying which invariant it certifies is ambiguous.
    """
    # Tight cluster at the top, wide boundary -> the order radius binds.
    tight_top = (1.00, 0.99, 0.20)
    cert = certified_radius(tight_top, 2)
    assert cert.order_radius < cert.set_radius

    # Well-spread top, near-tied boundary -> the set radius binds.
    tight_boundary = (1.0, 0.5, 0.49)
    cert = certified_radius(tight_boundary, 2)
    assert cert.set_radius < cert.order_radius

    # Both orderings occur on random data, so neither is a theorem.
    rng = random.Random(7)
    seen_order_tighter = seen_set_tighter = False
    for _ in range(200):
        s = sorted_scores_desc([rng.random() for _ in range(rng.randint(3, 20))])
        for k in range(2, len(s)):
            c = certified_radius(s, k)
            if not (c.defined and not math.isnan(c.order_radius)):
                continue
            seen_order_tighter |= c.order_radius < c.set_radius
            seen_set_tighter |= c.set_radius < c.order_radius
    assert seen_order_tighter, "expected a case where the order radius binds"
    assert seen_set_tighter, "expected a case where the set radius binds"


def test_the_joint_radius_is_the_minimum_of_the_two() -> None:
    """Guaranteeing set *and* order needs both conditions to hold."""
    rng = random.Random(8)
    for _ in range(100):
        s = sorted_scores_desc([rng.random() for _ in range(rng.randint(3, 15))])
        for k in range(2, len(s)):
            c = certified_radius(s, k)
            if c.defined and not math.isnan(c.order_radius):
                assert c.joint_radius == min(c.set_radius, c.order_radius)
                assert is_top_k_stable(s, k, c.joint_radius * 0.99)
                assert is_order_stable(s, k, c.joint_radius * 0.99)


@given(
    st.lists(st.floats(0.0, 1.0, allow_nan=False), min_size=4, max_size=25),
    st.integers(1, 5),
    st.floats(0.0, 0.999),
)
def test_a_certified_perturbation_never_changes_the_top_k_set(
    scores: list[float], k: int, fraction: float
) -> None:
    """Section 4.4's sufficiency, as an adversarial search.

    The ``assume`` is on the **realised** deltas rather than the drawn ``eps``:
    the theorem is over the reals, but ``fl(s + d)`` rounds, so the realised
    perturbation can exceed ``|d|`` by up to half an ulp. Assuming on the drawn
    value would make this test flaky for reasons that have nothing to do with
    the mathematics.
    """
    n = len(scores)
    assume(k < n)
    s = sorted_scores_desc(scores)
    cert = certified_radius(s, k)
    assume(cert.defined and cert.set_radius > 0.0)

    eps = cert.set_radius * fraction
    perturbed = [x + (eps if i % 2 else -eps) for i, x in enumerate(scores)]
    realised = max(abs(p - x) for p, x in zip(perturbed, scores, strict=True))
    assume(realised < cert.set_radius)

    table = table_of(n)
    before = set(rank(scores, table, POP).order[:k])
    after = set(rank(perturbed, table, POP).order[:k])
    assert before == after


def test_the_flip_witness_shows_the_radius_is_exact() -> None:
    """Necessity, which section 4.4 does not address.

    Dyadic throughout, so the construction has no rounding of its own: the
    margin is 0.25, the radius 0.125, and the witness sits one ``2^-30`` beyond.
    """
    scores = [1.0, 0.5, 0.25, 0.0]
    table = table_of(4)
    ranking = rank(scores, table, POP)
    k = 2

    cert = certified_radius(ranking.sorted_scores, k)
    assert cert.set_radius == 0.125
    assert is_top_k_stable(ranking.sorted_scores, k, 0.124) is True
    assert is_top_k_stable(ranking.sorted_scores, k, 0.125) is False

    witness = flip_witness(scores, ranking.order, k, delta=2.0**-30)
    assert witness is not None
    perturbed, eps = witness
    assert eps == 0.125 + 2.0**-30
    assert max(abs(p - s) for p, s in zip(perturbed, scores, strict=True)) == eps

    before = set(rank(scores, table, POP).order[:k])
    after = set(rank(perturbed, table, POP).order[:k])
    assert before != after, "the witness must flip the top-k set"


@given(st.lists(st.floats(0.01, 1.0, allow_nan=False), min_size=4, max_size=15), st.integers(1, 3))
def test_the_flip_witness_always_flips_when_it_exists(scores: list[float], k: int) -> None:
    """The same claim, searched rather than constructed."""
    n = len(scores)
    assume(k < n)
    table = table_of(n)
    ranking = rank(scores, table, POP)
    witness = flip_witness(scores, ranking.order, k)
    assume(witness is not None)
    perturbed, _ = witness  # type: ignore[misc]

    before = set(rank(scores, table, POP).order[:k])
    after = set(rank(perturbed, table, POP).order[:k])
    assert before != after


def test_no_witness_exists_at_an_exact_tie() -> None:
    """There is no radius to exceed: the pair is already tied."""
    scores = [1.0, 0.5, 0.5, 0.0]
    ranking = rank(scores, table_of(4), POP)
    assert flip_witness(scores, ranking.order, 2) is None


def test_no_witness_beyond_the_end_of_the_ranking() -> None:
    scores = [1.0, 0.5]
    ranking = rank(scores, table_of(2), POP)
    assert flip_witness(scores, ranking.order, 2) is None


def test_a_perturbation_can_preserve_membership_while_reordering_it() -> None:
    """The two guarantees are genuinely different properties.

    With a tight gap at the top and a wide one at the boundary, there is a band
    of ``eps`` for which the top-k *set* is certified but its *ordering* is not.
    """
    s = (1.0, 0.9, 0.2)
    cert = certified_radius(s, 2)
    assert cert.order_radius < cert.set_radius
    eps = (cert.order_radius + cert.set_radius) / 2
    assert is_top_k_stable(s, 2, eps) is True
    assert is_order_stable(s, 2, eps) is False
