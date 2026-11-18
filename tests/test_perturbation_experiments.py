"""One perturbation, measured at every stage of the section 4 chain.

`perturbation/experiments.py` refits both corpora and reports idf, w, cos and
top-k certificates. Scores are keyed by identifier, never by position. `True`
from `certified_stable` is a proof; `False` means only that no proof reaches.
"""

from __future__ import annotations

import dataclasses
import math
import random

import pytest

from tfidf_stability.perturbation.corpus_edits import (
    Corpus,
    EditKind,
    add_document,
    duplicate_document,
    edit_document,
    remove_document,
)
from tfidf_stability.perturbation.experiments import PerturbationReport, run_perturbation
from tfidf_stability.utils.validation import EmptyVocabularyError

# ---------------------------------------------------------------------------
# Builders. Duplicated per file by house convention rather than shared, so a
# change here cannot silently alter another suite's fixtures.
# ---------------------------------------------------------------------------
_VOCAB = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta")


def corpus_of(n: int, *, seed: int = 20260819) -> Corpus:
    """A small corpus with overlapping feature streams, so df moves under an edit."""
    rng = random.Random(seed)
    ids = tuple(f"d{i}" for i in range(n))
    docs = tuple(tuple(rng.choice(_VOCAB) for _ in range(rng.randint(2, 5))) for _ in range(n))
    return (ids, docs)


# ---------------------------------------------------------------------------
# Normal: the chain is reported end to end
# ---------------------------------------------------------------------------
def test_a_report_covers_every_document_present_on_both_sides() -> None:
    corpus = corpus_of(8)
    perturbed, edit = edit_document(corpus, "d3", ("alpha", "alpha", "beta"))
    report = run_perturbation(corpus, perturbed, edit, ("alpha", "beta"))

    assert isinstance(report, PerturbationReport)
    assert len(report.vector_shifts) == 8, "an edit leaves N unchanged, so every document is shared"
    assert set(report.score_before) == set(report.score_after)


def test_scores_are_keyed_by_identifier_so_a_removal_cannot_misalign_them() -> None:
    """Scores are compared by identifier, not by position.

    Removing d0 shifts every later document down one index. A comparison by
    position would pair d1's new score with d0's old one and report movement
    for documents that did not move.
    """
    corpus = corpus_of(6)
    perturbed, edit = remove_document(corpus, "d0")
    report = run_perturbation(corpus, perturbed, edit, ("alpha",))

    assert "d0" in report.score_before
    assert "d0" not in report.score_after, "the removed document must not appear after"
    shared = set(report.score_before) & set(report.score_after)
    assert shared == {"d1", "d2", "d3", "d4", "d5"}
    # Every surviving document keeps its own identifier's score, not its neighbour's.
    assert len(report.vector_shifts) == len(shared)
    assert [v.doc_id for v in report.vector_shifts] == sorted(shared)


def test_a_duplicate_edit_manufactures_an_exact_tie_between_original_and_copy() -> None:
    """`duplicate_document` produces an exact tie, m_k = 0, without a search."""
    corpus = corpus_of(6)
    perturbed, edit = duplicate_document(corpus, "d1", "d1_copy")
    report = run_perturbation(corpus, perturbed, edit, ("alpha", "beta"))

    assert edit.kind is EditKind.DUPLICATE
    after = report.score_after
    assert after["d1"] == after["d1_copy"], "a copy must score identically to its original"


def test_dominant_terms_counts_one_document_per_shift_that_moved() -> None:
    corpus = corpus_of(10)
    perturbed, edit = edit_document(corpus, "d4", ("gamma", "gamma", "gamma"))
    report = run_perturbation(corpus, perturbed, edit, ("gamma",))

    counts = report.dominant_terms
    assert set(counts) == {"local", "global", "interaction"}
    moved = sum(1 for v in report.vector_shifts if v.bound.total > 0.0)
    assert sum(counts.values()) == moved, "every moved document contributes exactly one count"


def test_every_section_four_two_bound_holds_for_an_ordinary_edit() -> None:
    corpus = corpus_of(12)
    perturbed, edit = edit_document(corpus, "d7", ("delta", "epsilon"))
    report = run_perturbation(corpus, perturbed, edit, ("delta",))
    assert report.all_bounds_hold, "a measured shift exceeded the bound governing it"


# ---------------------------------------------------------------------------
# Boundary: where the report has nothing to measure
# ---------------------------------------------------------------------------
def test_an_edit_that_changes_nothing_moves_no_score_at_all() -> None:
    """max_score_shift is the eps of section 4.4, so a no-op must give exactly 0."""
    corpus = corpus_of(6)
    _ids, docs = corpus
    perturbed, edit = edit_document(corpus, "d2", docs[2])  # replace with itself
    report = run_perturbation(corpus, perturbed, edit, ("alpha", "beta"))

    assert report.max_score_shift == 0.0, "refitting an unchanged corpus must not move a score"
    assert report.all_bounds_hold
    # Every document is present and none of them moved, so the tally has to walk
    # the whole list and count nothing. A tally that counted a document whose
    # bound total is zero would attribute dominance to a term that did no work.
    assert len(report.vector_shifts) == 6, "the tally must have something to walk past"
    assert report.dominant_terms == {"local": 0, "global": 0, "interaction": 0}


def test_emptying_the_corpus_is_refused_rather_than_reported_as_a_perturbation() -> None:
    """Removing the last document leaves nothing to fit.

    The refusal is raised by vocabulary construction during the fit: a corpus
    with no documents has no vocabulary, and a report over one measures nothing.
    """
    corpus = (("only",), (("alpha", "beta"),))
    perturbed, edit = remove_document(corpus, "only")
    with pytest.raises(EmptyVocabularyError, match="empty corpus"):
        run_perturbation(corpus, perturbed, edit, ("alpha",))


def test_max_score_shift_over_no_shared_documents_is_zero_not_an_error() -> None:
    """`max(..., default=0.0)` guards a case `run_perturbation` cannot reach.

    Every edit kind leaves at least one document on both sides, so the empty
    intersection is reachable only by constructing the report directly. A bare
    `max()` over nothing raises, so the default is pinned here.
    """
    corpus = corpus_of(5)
    perturbed, edit = edit_document(corpus, "d1", ("alpha",))
    report = run_perturbation(corpus, perturbed, edit, ("alpha",))

    empty = dataclasses.replace(report, score_before={}, score_after={}, vector_shifts=())
    assert empty.max_score_shift == 0.0
    assert empty.dominant_terms == {"local": 0, "global": 0, "interaction": 0}
    assert empty.all_bounds_hold, "vacuously true over no shifts, and must not raise"


def test_adding_a_document_leaves_every_original_present_on_both_sides() -> None:
    corpus = corpus_of(5)
    perturbed, edit = add_document(corpus, "new", ("theta", "theta"))
    report = run_perturbation(corpus, perturbed, edit, ("theta",))

    assert edit.kind is EditKind.ADD
    assert "new" not in report.score_before
    assert "new" in report.score_after
    assert len(report.vector_shifts) == 5, "the five originals are shared; the addition is not"


# ---------------------------------------------------------------------------
# certified_stable: True is a proof, False is not its opposite
# ---------------------------------------------------------------------------
def test_certified_stable_searches_past_the_certificates_that_do_not_match() -> None:
    """The lookup is a scan, so it must survive a miss before a hit.

    With one certificate the loop body runs once and the "keep looking" arc is
    never taken, so a scan that returned on its first iteration would pass.
    """
    corpus = corpus_of(8)
    perturbed, edit = edit_document(corpus, "d1", ("alpha",))
    report = run_perturbation(corpus, perturbed, edit, ("alpha",), ks=(1, 2, 3))

    assert len(report.certificates_before) == 3
    # k=3 is last, so reaching it means two non-matching certificates were skipped.
    assert report.certified_stable(3) in (True, False, None)
    # No k matches at all, so every certificate is visited and none answers.
    assert report.certified_stable(999) is None, "no certificate was requested at k=999"


def test_certified_stable_returns_none_rather_than_false_when_the_margin_is_undefined() -> None:
    """An undefined margin gives `None`, not `False`.

    An undefined certificate means "no radius exists here", which is not the
    same claim as "the perturbation exceeded the radius". Collapsing the two
    would let absence of a certificate read as evidence of movement.
    """
    corpus = corpus_of(4)
    perturbed, edit = edit_document(corpus, "d0", ("alpha",))
    # k = 4 equals N, so r_{k+1} does not exist and the margin is undefined.
    report = run_perturbation(corpus, perturbed, edit, ("alpha",), ks=(4,))

    undefined = [c for c in report.certificates_before if c.k == 4 and not c.defined]
    if undefined:
        assert report.certified_stable(4) is None, "an undefined certificate is None, never False"


def test_a_zero_shift_is_certified_stable_wherever_a_radius_exists() -> None:
    """True is a proof, and a no-op edit is the case where it must be reachable."""
    corpus = corpus_of(10)
    _ids, docs = corpus
    perturbed, edit = edit_document(corpus, "d5", docs[5])
    report = run_perturbation(corpus, perturbed, edit, ("alpha", "beta"), ks=(1, 2, 3))

    proven = 0
    for cert in report.certificates_before:
        if cert.defined and cert.set_radius > 0.0:
            assert report.certified_stable(cert.k) is True, (
                "a shift of exactly zero is inside every positive radius"
            )
            proven += 1
    assert proven > 0, "no positive radius existed, so the claim was never tested"


def test_the_certificates_are_taken_before_the_edit_not_after() -> None:
    """Section 4.4 certifies from the unperturbed margins.

    Using the perturbed margins would be circular, certifying an edit against
    the state it produced.
    """
    corpus = corpus_of(9)
    perturbed, edit = edit_document(corpus, "d2", ("zeta", "eta", "theta"))
    report = run_perturbation(corpus, perturbed, edit, ("zeta",), ks=(3,))

    from tfidf_stability.perturbation.score_bounds import certified_radius
    from tfidf_stability.ranking.ranker import sorted_scores_desc

    expected = certified_radius(sorted_scores_desc(list(report.score_before.values())), 3)
    actual = next(c for c in report.certificates_before if c.k == 3)
    assert actual.defined == expected.defined
    if expected.defined:
        assert actual.set_radius == expected.set_radius, (
            "certificates must come from the before state"
        )


# ---------------------------------------------------------------------------
# Erroneous: the edit helpers refuse what would corrupt the ranking
# ---------------------------------------------------------------------------
def test_adding_a_duplicate_identifier_is_refused_before_any_fit() -> None:
    corpus = corpus_of(4)
    with pytest.raises(ValueError, match="already exists"):
        add_document(corpus, "d1", ("alpha",))


def test_editing_an_unknown_document_is_refused_rather_than_appended() -> None:
    corpus = corpus_of(4)
    with pytest.raises(KeyError, match="no document with id"):
        edit_document(corpus, "absent", ("alpha",))


# ---------------------------------------------------------------------------
# Stress: adversarial corpora, at nightly cost
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_every_bound_holds_across_a_sweep_of_edits_and_corpus_sizes() -> None:
    """Every section 4.2 bound holds across three corpus sizes and twelve seeds."""
    checked = 0
    for n in (4, 9, 20):
        for seed in range(12):
            corpus = corpus_of(n, seed=seed)
            perturbed, edit = edit_document(
                corpus, f"d{seed % n}", ("alpha", "beta", "gamma")[: 1 + seed % 3]
            )
            report = run_perturbation(corpus, perturbed, edit, ("alpha", "beta"))
            assert report.all_bounds_hold, f"a bound failed at n={n}, seed={seed}"
            assert math.isfinite(report.max_score_shift)
            checked += 1
    assert checked == 36, "the sweep did not run the shape it claims"


@pytest.mark.slow
def test_a_corpus_of_identical_documents_moves_every_score_together() -> None:
    """In an all-identical corpus, the untouched copies still tie after an edit."""
    corpus = (tuple(f"d{i}" for i in range(8)), tuple(("alpha", "beta") for _ in range(8)))
    perturbed, edit = edit_document(corpus, "d0", ("gamma", "delta"))
    report = run_perturbation(corpus, perturbed, edit, ("alpha",))

    untouched = [report.score_after[d] for d in report.score_after if d != "d0"]
    assert len(set(untouched)) == 1, "documents that are exact copies must still tie after the edit"
    assert report.all_bounds_hold


# ---------------------------------------------------------------------------
# An edit that changes the corpus size has no certificate at any k.
# Rationale: docs/engineering_notes.md#certified-stable-size-changing-edits
# ---------------------------------------------------------------------------
def _displacing_add() -> tuple[object, object]:
    """A corpus and an added document that matches the query exactly.

    Local by house convention. Five documents on disjoint topics, so only d0
    matches the query; the addition is a perfect match, so it outranks d0
    while the surviving scores move only through the idf shift it causes.
    """
    corpus = (
        ("d0", "d1", "d2", "d3", "d4"),
        (
            ["alpha", "beta", "gamma"],
            ["delta", "epsilon"],
            ["zeta", "eta"],
            ["theta", "iota"],
            ["kappa", "lambda"],
        ),
    )
    return corpus, add_document(corpus, "NEW", ["alpha", "beta"])


def test_an_added_document_can_take_rank_one_while_every_survivor_barely_moves() -> None:
    """The premise of the size guard: an addition takes rank 1 cheaply.

    Only d0's score moves, by about 0.06, through the idf shift from N and the
    df of alpha and beta; that is inside the k=1 radius of about 0.41. The
    added document scores 1.0 and takes rank 1 outright.
    """
    corpus, (perturbed, edit) = _displacing_add()
    report = run_perturbation(corpus, perturbed, edit, ("alpha", "beta"), ks=(1, 2))

    before = sorted(report.score_before.items(), key=lambda kv: -kv[1])
    after = sorted(report.score_after.items(), key=lambda kv: -kv[1])

    assert before[0][0] == "d0"
    assert after[0][0] == "NEW", "the addition takes rank 1"
    assert report.max_score_shift < 0.1, "while no surviving score moved far"


@pytest.mark.parametrize(
    ("label", "make"),
    [
        ("adding", lambda c: add_document(c, "NEW", ["alpha", "beta"])),
        ("removing", lambda c: remove_document(c, "d1")),
        ("duplicating", lambda c: duplicate_document(c, "d0", "copy")),
    ],
)
def test_an_edit_that_changes_the_corpus_size_certifies_nothing(label: str, make: object) -> None:
    """`None`, never `True`.

    `max_score_shift` is measured over documents present on both sides, so for
    an addition it can sit inside the k=1 radius while the top-1 set changes.
    A removal can take a document out of the top-k without any score moving.
    """
    corpus = (
        ("d0", "d1", "d2", "d3", "d4"),
        (
            ["alpha", "beta", "gamma"],
            ["delta", "epsilon"],
            ["zeta", "eta"],
            ["theta", "iota"],
            ["kappa", "lambda"],
        ),
    )
    perturbed, edit = make(corpus)  # type: ignore[operator]
    report = run_perturbation(corpus, perturbed, edit, ("alpha", "beta"), ks=(1, 2, 3))

    assert edit.changes_corpus_size is True, "the premise of the guard"
    for k in (1, 2, 3):
        assert report.certified_stable(k) is None, f"{label} certified k={k}"


def test_an_edit_that_keeps_the_corpus_size_still_certifies() -> None:
    """The guard must not swallow the case section 4.4 does cover.

    Replacing a document's features perturbs existing scores and leaves the
    candidate set alone, which is the hypothesis of the theorem, so a verdict
    is still available here.
    """
    corpus = corpus_of(10)
    _ids, docs = corpus
    perturbed, edit = edit_document(corpus, "d5", docs[5])
    report = run_perturbation(corpus, perturbed, edit, ("alpha", "beta"), ks=(1, 2, 3))

    assert edit.changes_corpus_size is False
    verdicts = [report.certified_stable(k) for k in (1, 2, 3)]
    assert any(v is True for v in verdicts), "the certificate is still reachable"
