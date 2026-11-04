"""Property and example tests for the inequalities of README section 4.

Section 4.1 covers ``delta_idf`` and its sign, 4.2 the three-term bound on ``||w' - w||``,
4.3 ``|delta cos| <= C (||du|| + ||dv||)`` with ``C = 1/L`` (``spec_addenda.md#g4``),
and 4.4 the top-k certificate ``eps < m_k/2``, which is also checked to be the exact radius.
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
from tfidf_stability.perturbation.idf_perturb import (
    Alignment,
    IdfPerturbation,
    align_models,
    analyse_idf_shift,
)
from tfidf_stability.perturbation.score_bounds import (
    certified_radius,
    flip_witness,
    is_order_stable,
    is_top_k_stable,
)
from tfidf_stability.perturbation.vector_perturb import (
    ThreeTermBound,
    VectorPerturbation,
    analyse_vector_shift,
)
from tfidf_stability.ranking.attributes import AttributeSpec, AttributeTable
from tfidf_stability.ranking.ranker import rank, rank_top_k, sorted_scores_desc
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
    """Adding, removing and editing leave the original corpus unchanged.

    A perturbation experiment needs the corpus before and after an edit at once.
    """
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
    """A duplicated document is identical to its source, so it ties exactly."""
    corpus = (("a",), (("x", "y"),))
    new_corpus, rec = duplicate_document(corpus, "a", "a_copy")
    assert new_corpus[1][0] == new_corpus[1][1]
    assert rec.kind is EditKind.DUPLICATE


def test_duplicate_ids_are_refused() -> None:
    with pytest.raises(ValueError, match="already exists"):
        add_document((("a",), (("x",),)), "a", ["y"])


def test_editing_an_unknown_document_raises() -> None:
    with pytest.raises(KeyError, match="no document with id 'zzz'"):
        edit_document((("a",), (("x",),)), "zzz", ["y"])


# ---------------------------------------------------------------------------
# Section 4.1: IDF perturbation
# ---------------------------------------------------------------------------
def test_delta_idf_captures_the_two_competing_effects() -> None:
    """Section 4.1: N and df move the idf in opposite directions."""
    assert delta_idf(3, 4, 10, 11) < 0, "the token gained a document"
    assert delta_idf(3, 3, 10, 11) > 0, "the corpus grew, the token did not"
    assert delta_idf(3, 3, 10, 10) == 0.0


@pytest.mark.property
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

    Section 4.1 says so explicitly; a corpus edit is never a purely local
    perturbation.
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
# Section 4.2: the three-term bound
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(12))
def test_three_term_bound_is_never_violated_under_a_real_edit(seed: int) -> None:
    """The inequality, evaluated against actual corpus perturbations.

    Randomised over all four edit kinds. The vocabulary churns here, the case
    section 4.2 does not cover and G5 resolves.
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
    """Editing one document still moves the others, through ``idf`` alone.

    Section 4.2's separation: for every document but the edited one ``dtf`` is
    zero, so the local and interaction terms vanish and the bound is the global
    term.
    """
    rng = random.Random(4)
    corpus = corpus_of(rng, 8)
    before = fit(corpus)
    after = fit(edit_document(corpus, "d0", ["a", "a", "b"])[0])

    asserted = 0
    for doc_id in before.doc_ids:
        if doc_id == "d0":
            continue
        result = analyse_vector_shift(before, after, doc_id)
        if result.bound.observed > 0.0:
            assert result.bound.dominant_term == "global"
            assert result.bound.local == pytest.approx(0.0, abs=1e-12)
            asserted += 1
    # At least one non-edited document shows a non-zero shift; otherwise the
    # loop above asserts nothing.
    assert asserted > 0, "no document was actually examined"


def test_analyse_vector_shift_refuses_a_document_missing_from_one_side() -> None:
    rng = random.Random(5)
    corpus = corpus_of(rng, 4)
    before = fit(corpus)
    after = fit(add_document(corpus, "new", ["a"])[0])
    with pytest.raises(KeyError, match="present in both"):
        analyse_vector_shift(before, after, "new")


# ---------------------------------------------------------------------------
# Section 4.3: the Lipschitz bound (G4)
# ---------------------------------------------------------------------------
nonneg = st.floats(min_value=0.0, max_value=1e3, allow_nan=False, allow_infinity=False)
sparse_map = st.dictionaries(st.integers(0, DIM - 1), nonneg, min_size=0, max_size=DIM)


def sv(mapping: dict[int, float]) -> SparseVector:
    return SparseVector.from_mapping(mapping, DIM)


@pytest.mark.property
@settings(suppress_health_check=[HealthCheck.filter_too_much])
@given(sparse_map, sparse_map, sparse_map, sparse_map)
def test_lipschitz_bound_survives_an_adversarial_search(
    a: dict[int, float], b: dict[int, float], c: dict[int, float], d: dict[int, float]
) -> None:
    """``C = 1/L`` where ``L`` is the smallest of the four norms (G4).

    Tiny-norm vectors are excluded because evaluating the bound in binary64 gets
    dominated by its own rounding there (the low-norm regime of G18). The bound
    itself still holds.
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
# Section 4.4: certificates, and that the radius is exact
# ---------------------------------------------------------------------------
def test_certificate_is_half_the_margin() -> None:
    s = (1.0, 0.75, 0.5, 0.25)
    cert = certified_radius(s, 2)
    assert cert.defined is True
    assert cert.set_radius == 0.125
    assert cert.exact_tie is False


def test_certificate_at_an_exact_tie_is_no_radius_at_all() -> None:
    """At an exact tie the set radius is zero and the certificate does not hold.

    The tie-break decides membership, so zero differs in kind from a small radius.
    """
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
    """The two radii of section 4.4 constrain disjoint sets of gaps.

    ``m_min^top`` minimises over the gaps strictly inside the top-k (ranks 1->2
    through (k-1)->k) and ``m_k`` is the boundary gap (k->k+1). Neither radius
    bounds the other, so a certificate quoted without its invariant is ambiguous.
    """
    # Tight cluster at the top, wide boundary -> the order radius binds.
    tight_top = (1.00, 0.99, 0.20)
    cert = certified_radius(tight_top, 2)
    assert cert.order_radius < cert.set_radius

    # Well-spread top, near-tied boundary -> the set radius binds.
    tight_boundary = (1.0, 0.5, 0.49)
    cert = certified_radius(tight_boundary, 2)
    assert cert.set_radius < cert.order_radius

    # Both orderings occur on random data, so neither inequality is general.
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
    asserted = 0
    for _ in range(100):
        s = sorted_scores_desc([rng.random() for _ in range(rng.randint(3, 15))])
        for k in range(2, len(s)):
            c = certified_radius(s, k)
            if c.defined and not math.isnan(c.order_radius):
                assert c.joint_radius == min(c.set_radius, c.order_radius)
                assert is_top_k_stable(s, k, c.joint_radius * 0.99)
                assert is_order_stable(s, k, c.joint_radius * 0.99)
                asserted += 1
    # At least one (scores, k) pair has a defined order radius; every assertion
    # above sits behind that guard.
    assert asserted > 0, "no (scores, k) pair had a defined order radius"


@pytest.mark.property
@given(
    st.lists(st.floats(0.0, 1.0, allow_nan=False), min_size=4, max_size=25),
    st.integers(1, 5),
    st.floats(0.0, 0.999),
)
def test_a_certified_perturbation_never_changes_the_top_k_set(
    scores: list[float], k: int, fraction: float
) -> None:
    """Section 4.4's sufficiency, as an adversarial search.

    The ``assume`` is on the realised deltas rather than the drawn ``eps``: the
    theorem is over the reals, but ``fl(s + d)`` rounds, so the realised
    perturbation can exceed ``|d|`` by up to half an ulp, enough to reach a small radius.
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

    Dyadic throughout, so the construction contributes no rounding: margin 0.25,
    radius 0.125, witness one ``2^-30`` beyond.
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


@pytest.mark.property
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
    """The two guarantees are different properties.

    With a tight gap at the top and a wide one at the boundary there is a band
    of ``eps`` where the top-k set is certified and its ordering is not.
    """
    s = (1.0, 0.9, 0.2)
    cert = certified_radius(s, 2)
    assert cert.order_radius < cert.set_radius
    eps = (cert.order_radius + cert.set_radius) / 2
    assert is_top_k_stable(s, 2, eps) is True
    assert is_order_stable(s, 2, eps) is False


# ---------------------------------------------------------------------------
# Which of the two radii binds, and when neither does
# ---------------------------------------------------------------------------


# Neither radius dominates: m_min^top minimises over the gaps strictly inside the
# top-k, m_k is the boundary gap, and those sets are disjoint. A certificate
# quoted without saying which invariant it certifies is ambiguous, so both are
# carried and both arms of the comparison have to be reachable.
def test_a_tight_top_with_a_wide_boundary_makes_the_order_radius_binding() -> None:
    scores = (1.000, 0.999, 0.100)  # ranks 1->2 nearly tied, 2->3 wide apart
    cert = certified_radius(scores, 2)
    assert cert.defined
    assert cert.order_radius < cert.set_radius
    assert cert.order_radius_is_binding
    assert cert.joint_radius == cert.order_radius


def test_a_spread_top_with_a_near_tied_boundary_makes_the_set_radius_binding() -> None:
    scores = (1.000, 0.500, 0.499)  # ranks 1->2 wide, 2->3 nearly tied
    cert = certified_radius(scores, 2)
    assert cert.defined
    assert cert.set_radius < cert.order_radius
    assert not cert.order_radius_is_binding
    assert cert.joint_radius == cert.set_radius


def test_an_undefined_certificate_binds_nothing_and_has_no_joint_radius() -> None:
    """k == N leaves no r_{k+1}, so there is no boundary gap to certify."""
    cert = certified_radius((1.0, 0.5), 2)
    assert not cert.defined
    assert not cert.order_radius_is_binding, "an absent radius cannot be the binding one"
    assert math.isnan(cert.joint_radius)


def test_a_vacuous_order_radius_leaves_the_set_radius_as_the_joint_one() -> None:
    """At k = 1 the order radius is NaN and the joint radius is the set radius.

    G16 leaves the order radius undefined at k = 1 because the minimum over the
    interior gaps is empty. That NaN must not propagate into the joint radius.
    """
    cert = certified_radius((1.0, 0.5, 0.2), 1)
    assert cert.defined
    assert math.isnan(cert.order_radius), "the premise: no interior gap exists at k = 1"
    assert not cert.order_radius_is_binding
    assert cert.joint_radius == cert.set_radius, "a vacuous radius constrains nothing"


def test_a_perturbation_of_exactly_half_the_margin_ties_rather_than_reversing() -> None:
    """At exactly m/2 the pair ties, so flip_witness returns no witness (A1).

    Dyadic scores make the construction exact, so at ``eps = m/2`` the two
    scores meet rather than cross. No witness is returned there or for an excess
    lost to rounding; the default excess produces one that reverses the pair.
    """
    scores = (0.5, 0.25, 0.1)  # dyadic, so m/2 lands on a representable value
    order = (0, 1, 2)
    assert flip_witness(scores, order, 1, delta=0.0) is None, (
        "exactly m/2 ties the pair; only a strict excess reverses it"
    )
    assert flip_witness(scores, order, 1, delta=1e-30) is None, (
        "an excess too small to survive rounding is no excess at all"
    )

    outcome = flip_witness(scores, order, 1)
    assert outcome is not None, "the default excess must produce a genuine witness"
    perturbed, eps = outcome
    assert perturbed[1] > perturbed[0], "a returned witness must actually reverse the pair"
    assert eps > (scores[0] - scores[1]) / 2.0, "and it must exceed the certified radius"


# ---------------------------------------------------------------------------
# The degenerate readings the diagnostics have to survive
# ---------------------------------------------------------------------------

# Each of these is a ratio whose denominator can be zero: an edit that moved
# nothing, or one that moved only tokens the two vocabularies do not share.
# The records are constructed directly because the accessor's return value is
# under test, not the corpus edit that produces it.
_EMPTY_ALIGNMENT = Alignment(tokens=(), in_before=(), in_after=(), idf_before=(), idf_after=())


def _idf_shift(linf: float, linf_shared: float) -> IdfPerturbation:
    return IdfPerturbation(
        alignment=_EMPTY_ALIGNMENT,
        n_before=0,
        n_after=0,
        linf=linf,
        linf_shared=linf_shared,
        worst_token="",
        worst_delta=linf,
    )


def test_a_shift_confined_to_churned_tokens_reports_infinite_looseness() -> None:
    """A shift confined to churned tokens reports infinite looseness.

    No shared token moved, so the section 4.2 bound is driven by tokens on one
    side only. Reporting 1.0, the value for a maximally tight bound, would
    invert the reading.
    """
    assert _idf_shift(linf=0.25, linf_shared=0.0).looseness == float("inf")


def test_an_idf_vector_that_did_not_move_at_all_is_maximally_tight() -> None:
    """0/0 is the no-op edit, not a degenerate one: the bound is exact."""
    assert _idf_shift(linf=0.0, linf_shared=0.0).looseness == 1.0


def test_a_stable_vocabulary_gives_a_looseness_of_one() -> None:
    """The reference point the two cases above are read against."""
    assert _idf_shift(linf=0.25, linf_shared=0.25).looseness == 1.0


def test_a_bound_of_zero_reports_no_tightness_rather_than_dividing_by_it() -> None:
    """A bound of zero reports a tightness of zero rather than dividing by it.

    All three terms vanish and the observed shift is zero, so 0/0 must resolve
    to a number a percentile summary can hold.
    """
    assert ThreeTermBound(local=0.0, glob=0.0, interaction=0.0, observed=0.0).tightness == 0.0


def test_the_churn_fraction_splits_a_movement_into_shared_and_churned_parts() -> None:
    """The churn fraction is the share of squared movement on churned tokens.

    The three parts are supported on disjoint coordinate sets, so the squared
    movement partitions exactly. The fraction separates tokens changing weight
    from tokens appearing or disappearing.
    """
    # 0.6^2 + 0.6^2 + 0.52^2 = 0.9904 = 0.99518...^2, so Pythagoras holds.
    moved = VectorPerturbation(
        doc_id="d0",
        bound=ThreeTermBound(local=1.0, glob=1.0, interaction=1.0, observed=math.sqrt(0.9904)),
        alignment=_EMPTY_ALIGNMENT,
        shared_shift=0.6,
        gained_mass=0.6,
        lost_mass=0.52,
    )
    assert moved.pythagoras_holds, "the premise: the split is exact"
    assert moved.churn_fraction == pytest.approx((0.6**2 + 0.52**2) / 0.9904)
    assert 0.0 < moved.churn_fraction < 1.0


def test_a_document_that_did_not_move_has_no_churn_fraction() -> None:
    """A document the edit did not move has a churn fraction of zero.

    Churn is measured against the observed shift, so this case divides by zero.
    Zero is returned because none of a zero movement is due to churn.
    """
    still = VectorPerturbation(
        doc_id="d0",
        bound=ThreeTermBound(local=0.0, glob=0.0, interaction=0.0, observed=0.0),
        alignment=_EMPTY_ALIGNMENT,
        shared_shift=0.0,
        gained_mass=0.0,
        lost_mass=0.0,
    )
    assert still.churn_fraction == 0.0
    assert still.bound.holds, "a zero bound still bounds a zero shift"


# ---------------------------------------------------------------------------
# Vocabulary churn: what a token absent from one side is worth
# ---------------------------------------------------------------------------
def test_a_token_missing_from_a_model_contributes_zero_idf_not_one() -> None:
    """A token absent from one model has an idf of 0.0 there, not 1.0.

    `align_models` places both models over the union vocabulary. The smoothed
    idf floor is exactly 1.0, so treating absence as 1.0 would make a vanished
    token indistinguishable from a ubiquitous one and shrink every G5 delta.
    """
    before = TfidfVectoriser().fit([["alpha", "beta"], ["alpha"]])
    after = TfidfVectoriser().fit([["gamma", "delta"], ["gamma"]])

    alignment = align_models(before, after)
    assert alignment.tokens == ("alpha", "beta", "delta", "gamma")
    assert alignment.shared == (), "the premise: the two vocabularies are disjoint"

    # N = 2 either side, so df of 2 gives 1.0 and df of 1 gives log(3/2) + 1.
    common, rare = smoothed_idf_one(2, 2), smoothed_idf_one(1, 2)
    assert common == 1.0, "the smoothed floor, which is why 0.0 and 1.0 differ here"

    # after - before, over the union, in token order.
    assert tuple(alignment.delta_idf()) == (-common, -rare, rare, common)


def test_a_wholly_replaced_vocabulary_has_no_shared_movement_at_all() -> None:
    """With no shared tokens, `linf_shared` is zero and `looseness` is infinite.

    G5 reads an infinite looseness as a bound driven entirely by churn.
    """
    before = TfidfVectoriser().fit([["alpha", "beta"], ["alpha"]])
    after = TfidfVectoriser().fit([["gamma", "delta"], ["gamma"]])

    shift = analyse_idf_shift(before, after)

    assert shift.linf == smoothed_idf_one(1, 2), "the largest move is the rare token's"
    assert shift.linf_shared == 0.0
    assert shift.looseness == float("inf")
    assert shift.worst_token in ("beta", "delta"), "one of the two rare tokens"
    assert abs(shift.worst_delta) == shift.linf


# ---------------------------------------------------------------------------
# The two tolerance comparisons, at the tolerance
# ---------------------------------------------------------------------------


# `holds` and `pythagoras_holds` are checked here at the tolerance itself, from
# both sides. The one-sided assertions elsewhere in this file also pass with a
# tolerance divided by the scale instead of multiplied, or without the
# `max(1.0, ...)` floor.
def _three_term(local: float, glob: float, interaction: float, observed: float) -> ThreeTermBound:
    return ThreeTermBound(local=local, glob=glob, interaction=interaction, observed=observed)


def test_the_three_term_bound_admits_its_slack_exactly_and_no_further() -> None:
    """The slack covers rounding in evaluating the bound and nothing further."""
    total = 4.0
    limit = total * (1.0 + 1e-12) + 1e-15
    at = _three_term(local=1.0, glob=2.0, interaction=1.0, observed=limit)
    assert at.total == total, "the premise: the three terms sum to the total"
    assert at.holds, "an observed value sitting exactly on the slack is inside it"

    assert not _three_term(1.0, 2.0, 1.0, observed=limit + 1e-9).holds


def test_the_three_term_tightness_is_the_ratio_even_below_one() -> None:
    """Tightness is guarded on `total > 0.0`, not on `total > 1.0`.

    A threshold of 1.0 would report zero tightness for every bound below one,
    which covers most norms of small perturbations.
    """
    assert _three_term(0.25, 0.25, 0.0, observed=0.25).tightness == 0.5
    assert _three_term(2.0, 1.0, 1.0, observed=2.0).tightness == 0.5


def _pythagoras(observed: float, shared: float, gained: float, lost: float) -> VectorPerturbation:
    return VectorPerturbation(
        doc_id="d0",
        bound=_three_term(observed, 0.0, 0.0, observed),
        alignment=_EMPTY_ALIGNMENT,
        shared_shift=shared,
        gained_mass=gained,
        lost_mass=lost,
    )


def test_the_pythagorean_tolerance_has_a_floor_below_one() -> None:
    """The Pythagorean tolerance is `1e-9 * max(1.0, lhs)`, floored at 1e-9.

    Without the floor a small movement gets a proportionally tiny tolerance and
    the identity, exact in the reals and approximate in binary64, fails on sound
    data. 2**-15 squares exactly, so the discrepancy here is exactly 2**-30.
    """
    # observed = 0, so lhs = 0 and the floor is doing all the work.
    tiny = _pythagoras(observed=0.0, shared=0.0, gained=2.0**-15, lost=0.0)
    assert 2.0**-30 < 1e-9, "the discrepancy is inside the floored tolerance"
    assert tiny.pythagoras_holds


def test_the_pythagorean_tolerance_scales_with_the_movement_above_one() -> None:
    """Above the floor the tolerance is proportional to the movement.

    Dividing by the scale instead of multiplying inverts that.
    """
    big = _pythagoras(observed=2.0, shared=2.0, gained=2.0**-15, lost=0.0)
    # lhs = 4, rhs = 4 + 2**-30, tolerance 1e-9 * 4.
    assert 2.0**-30 < 1e-9 * 4.0
    assert 2.0**-30 > 1e-9 / 4.0, "and outside the tolerance a division would give"
    assert big.pythagoras_holds


def test_a_broken_pythagorean_split_is_reported_as_broken() -> None:
    """A split that violates the identity reports `pythagoras_holds` as False.

    A misaligned index breaks the identity long before it breaks any inequality.
    """
    assert not _pythagoras(observed=1.0, shared=1.0, gained=1.0, lost=1.0).pythagoras_holds


# ---------------------------------------------------------------------------
# Recovering tf across a churned vocabulary
# ---------------------------------------------------------------------------
def test_a_token_only_one_model_knows_contributes_no_term_frequency() -> None:
    """A token known to only one model contributes no term frequency.

    `w / idf` recovers tf and idf is 0.0 for a token the model never saw, so the
    division is guarded at 0.0. The smoothed idf floor is exactly 1.0, so a
    guard at 1.0 would zero the tf of every token appearing in every document.
    """
    before = TfidfVectoriser().fit([["alpha", "beta"], ["alpha", "gamma"]], ["d0", "d1"])
    after = TfidfVectoriser().fit([["alpha", "delta"], ["alpha", "gamma"]], ["d0", "d1"])

    # "alpha" is in every document on both sides, so its idf is exactly 1.0.
    assert smoothed_idf_one(2, 2) == 1.0
    result = analyse_vector_shift(before, after, "d0")

    assert result.bound.holds
    assert result.pythagoras_holds
    # beta left and delta arrived, so both sides carry mass the other does not.
    assert result.lost_mass > 0.0, "beta was in the before model only"
    assert result.gained_mass > 0.0, "delta is in the after model only"
    assert 0.0 < result.churn_fraction <= 1.0


def test_a_ubiquitous_token_keeps_its_term_frequency() -> None:
    """A token in every document keeps its tf under the `w / idf` guard.

    The smoothed idf floor is exactly 1.0, so guarding at 1.0 instead of 0.0
    would zero the tf of such tokens, and the global term uses ``||tf||_2``. In
    `d0`, ``tf = (1/2, 1/2)`` gives ``sqrt(1/2)``; dropping alpha gives ``1/2``.
    """
    before = TfidfVectoriser().fit([["alpha", "beta"], ["alpha"]], ["d0", "d1"])
    after = TfidfVectoriser().fit([["alpha", "beta"], ["alpha", "beta"]], ["d0", "d1"])

    assert smoothed_idf_one(2, 2) == 1.0, "alpha is in every document on both sides"
    rare = smoothed_idf_one(1, 2)

    result = analyse_vector_shift(before, after, "d0")

    # glob = ||tf||_2 * ||delta_idf||_inf. Only beta's idf moved, by rare - 1.
    expected = math.sqrt(0.5) * (rare - 1.0)
    assert result.bound.glob == pytest.approx(expected, rel=1e-12)
    assert result.bound.glob != pytest.approx(0.5 * (rare - 1.0), rel=1e-6)


def test_a_token_absent_before_starts_from_zero_term_frequency() -> None:
    """A token absent before the edit starts from a term frequency of zero.

    Zero makes ``delta_tf`` the genuine movement; a start of one gives a wrong local
    term ``||delta_tf||_2 * ||idf||_inf``. The edit is asymmetric, gamma arriving and
    nothing leaving, because a paired add and remove cancels the error out.
    """
    before = TfidfVectoriser().fit([["alpha", "beta"], ["alpha"]], ["d0", "d1"])
    after = TfidfVectoriser().fit([["alpha", "beta", "gamma"], ["alpha"]], ["d0", "d1"])

    rare = smoothed_idf_one(1, 2)
    assert smoothed_idf_one(2, 2) == 1.0
    assert rare > 1.0, "so the infinity-norm of idf_before is the rare token's"

    result = analyse_vector_shift(before, after, "d0")

    # d0 goes from (1/2, 1/2, -) to (1/3, 1/3, 1/3), so delta_tf is
    # (-1/6, -1/6, 1/3) and its norm is sqrt(1/6).
    assert result.bound.local == pytest.approx(math.sqrt(1.0 / 6.0) * rare, rel=1e-12)
    # Starting gamma at 1.0 instead would make delta_tf (-1/6, -1/6, -2/3).
    assert result.bound.local != pytest.approx(math.sqrt(0.5) * rare, rel=1e-6)


# ---------------------------------------------------------------------------
# corpus_edits: every edit refuses what would break the ranking
# ---------------------------------------------------------------------------

# A `Corpus` is `(ids, documents)` and each edit returns the new corpus with an
# `EditRecord`. The record is what a manifest carries, so an edit that succeeded
# without describing itself would leave a run unreproducible.
_CORPUS = (("d0", "d1"), (["a", "b"], ["a"]))


@pytest.mark.parametrize(
    ("label", "edit"),
    [
        ("adding onto an existing id", lambda: add_document(_CORPUS, "d0", ["c"])),
        ("duplicating onto an existing id", lambda: duplicate_document(_CORPUS, "d0", "d1")),
    ],
)
def test_an_edit_that_would_duplicate_an_identifier_is_refused(label: str, edit: object) -> None:
    """An edit onto an existing identifier raises rather than succeeding.

    The final tie-break key is the identifier, so a duplicate leaves the sort
    order undetermined.
    """
    with pytest.raises(ValueError, match="already exists"):
        edit()  # type: ignore[operator]


@pytest.mark.parametrize(
    ("label", "edit"),
    [
        ("removing", lambda: remove_document(_CORPUS, "zz")),
        ("editing", lambda: edit_document(_CORPUS, "zz", ["z"])),
        ("duplicating", lambda: duplicate_document(_CORPUS, "zz", "new")),
    ],
)
def test_an_edit_naming_an_absent_document_is_refused(label: str, edit: object) -> None:
    """An edit naming an absent document raises a `KeyError` with the id.

    A sweep over a stale id list then reports which id was not found instead of
    appending a document.
    """
    with pytest.raises(KeyError, match="no document with id 'zz'"):
        edit()  # type: ignore[operator]


def test_adding_and_removing_change_the_corpus_size_in_opposite_directions() -> None:
    """Adding grows and removing shrinks the corpus by one document.

    `N` is in the idf of every term, so a silent size change moves every weight.
    """
    grown, _ = add_document(_CORPUS, "d2", ["c"])
    shrunk, _ = remove_document(_CORPUS, "d0")

    assert grown[0] == ("d0", "d1", "d2")
    assert shrunk[0] == ("d1",)


def test_editing_a_document_leaves_the_corpus_size_alone() -> None:
    """Editing a document leaves the corpus size unchanged.

    This isolates a change in the document-frequency distribution from a change
    in `N`, which section 4.1 separates.
    """
    edited, _ = edit_document(_CORPUS, "d0", ["z"])

    assert edited[0] == _CORPUS[0]
    assert edited[1][0] == ("z",)
    assert edited[1][1] == _CORPUS[1][1], "the untouched document is untouched"


def test_a_duplicate_scores_identically_and_so_manufactures_an_exact_tie() -> None:
    """A duplicate is identical to its source, feature for feature.

    The copy scores identically against every query, so membership at the
    boundary is decided purely by the tie-break.
    """
    copied, _ = duplicate_document(_CORPUS, "d0", "copy")

    assert copied[0] == ("d0", "d1", "copy")
    # The copy is stored as a tuple while the original arrived as a list, so the
    # comparison is on contents.
    assert tuple(copied[1][-1]) == tuple(copied[1][0]), (
        "the copy is the original, feature for feature"
    )


@pytest.mark.parametrize(
    ("edit", "kind"),
    [
        (lambda: add_document(_CORPUS, "d2", ["c"]), EditKind.ADD),
        (lambda: remove_document(_CORPUS, "d0"), EditKind.REMOVE),
        (lambda: edit_document(_CORPUS, "d0", ["z"]), EditKind.EDIT),
        (lambda: duplicate_document(_CORPUS, "d0", "copy"), EditKind.DUPLICATE),
    ],
)
def test_every_edit_describes_itself_for_the_manifest(edit: object, kind: EditKind) -> None:
    """Every edit record carries its kind for the run manifest.

    The manifest then says which perturbation produced a number without
    re-deriving it from the corpora.
    """
    _, record = edit()  # type: ignore[operator]
    assert record.kind is kind


def test_an_edit_leaves_the_original_corpus_untouched() -> None:
    """Edits return a new corpus and leave the baseline unchanged.

    A sweep can then apply many edits to one baseline without accumulation.
    """
    before = (_CORPUS[0], tuple(tuple(d) for d in _CORPUS[1]))
    add_document(_CORPUS, "d2", ["c"])
    remove_document(_CORPUS, "d0")

    assert _CORPUS[0] == before[0]
    assert tuple(tuple(d) for d in _CORPUS[1]) == before[1]


@pytest.mark.parametrize(
    ("label", "edit", "delta"),
    [
        ("add", lambda: add_document(_CORPUS, "d2", ["c"]), 1),
        ("remove", lambda: remove_document(_CORPUS, "d0"), -1),
        ("edit", lambda: edit_document(_CORPUS, "d0", ["z"]), 0),
        ("duplicate", lambda: duplicate_document(_CORPUS, "d0", "copy"), 1),
    ],
)
def test_the_record_reports_the_corpus_it_came_with(label: str, edit: object, delta: int) -> None:
    """The record carries `n_before` and `n_after` with the correct direction.

    `N` sits inside the idf of every term, so a misreported direction would
    invert the section 4.2 bound. `changes_corpus_size` only asks whether the
    two differ, so it cannot catch the direction on its own.
    """
    (ids, _), record = edit()  # type: ignore[operator]

    assert record.n_before == len(_CORPUS[0]), "the corpus the edit was handed"
    assert record.n_after == len(ids), "and the one it produced"
    assert record.n_after - record.n_before == delta


def test_an_edit_that_leaves_the_size_alone_says_so_rather_than_omitting_it() -> None:
    """An edit that keeps the size reports equal `n_before` and `n_after`.

    With `N` fixed, any idf movement comes from `df`, the case section 4.1
    isolates.
    """
    _, record = edit_document(_CORPUS, "d0", ["z"])

    assert record.n_before == record.n_after == len(_CORPUS[0])
    assert record.changes_corpus_size is False


def test_a_discrepancy_exactly_on_the_tolerance_is_accepted() -> None:
    """A discrepancy exactly on the tolerance is accepted: `<=`, not `<`.

    `sqrt(1e-9)` squares back to exactly `1e-9` in binary64, so the boundary is
    reachable: with `observed = 0` the left side is zero, the tolerance is the
    floor `1e-9`, and the right side equals it to the last bit.
    """
    at_the_edge = _pythagoras(observed=0.0, shared=math.sqrt(1e-9), gained=0.0, lost=0.0)

    assert math.sqrt(1e-9) ** 2 == 1e-9, "the discrepancy is exactly the tolerance"
    assert at_the_edge.pythagoras_holds


def test_a_discrepancy_one_ulp_past_the_tolerance_is_refused() -> None:
    """A discrepancy one representable step past the tolerance is refused."""
    just_over = math.nextafter(math.sqrt(1e-9), math.inf)
    outside = _pythagoras(observed=0.0, shared=just_over, gained=0.0, lost=0.0)

    assert just_over**2 > 1e-9
    assert not outside.pythagoras_holds


def test_a_truncated_ranking_returns_no_witness_rather_than_raising() -> None:
    """A `k` past the end of a truncated `order` yields None, not IndexError.

    `rank_top_k` returns `n_selected = min(k + 1, n)`, so a truncated `Ranking` can
    clear `k < len(scores)` while `order[k]` is out of range. The signature documents
    `None` "when no witness exists: `k` out of range".
    """
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    table = table_of(6)
    truncated = rank_top_k(scores, table, POP, k=2)
    assert len(truncated.order) == 3 < len(scores), "the premise: order is shorter"

    for k in (3, 4, 5):
        assert flip_witness(scores, truncated.order, k) is None

    # A `k` the truncated `order` covers still produces a witness, so the bound refuses
    # only the unanswerable case.
    assert flip_witness(scores, truncated.order, 1) is not None
    # A complete `order` answers every `k` in range.
    complete = rank(scores, table, POP)
    assert flip_witness(scores, complete.order, 4) is not None
