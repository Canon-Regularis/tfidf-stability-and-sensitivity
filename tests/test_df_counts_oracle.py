"""``df_counts`` as a checked oracle for the counts the vocabulary builds.

This module was found at **0% coverage**: nothing in the package imports it. It
is a second, independent implementation of document and collection frequency,
and an unexercised duplicate of a core quantity is a bug waiting to happen --
`vocabulary.py` computes `df` during vocabulary construction, and if the two ever
disagreed, every idf and therefore every score would depend on which one a caller
happened to reach for.

Rather than delete it, these tests turn it into a **differential oracle**, in
exactly the sense the reference-versus-native suite uses: two implementations
written differently, required to agree. That makes the duplication a check rather
than a liability.

`df_after_edit` earns its place separately. It is the incremental update §4.1
needs, and the property that matters -- an edit changes `df` by at most one per
term -- is what makes corpus perturbation tractable at scale. It is asserted
against a from-scratch recomputation here.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tfidf_stability.utils.validation import EmptyVocabularyError
from tfidf_stability.vectorisation.df_counts import (
    collection_frequencies,
    df_after_edit,
    document_frequencies,
)
from tfidf_stability.vectorisation.vocabulary import build_vocabulary

_TOKENS = st.text(alphabet="abcde", min_size=1, max_size=3)
_DOCUMENT = st.lists(_TOKENS, max_size=8)
_CORPUS = st.lists(_DOCUMENT, min_size=1, max_size=10)


# ---------------------------------------------------------------------------
# Agreement with the implementation that is actually used
# ---------------------------------------------------------------------------
def test_document_frequency_agrees_with_the_vocabulary(mini_features) -> None:
    """The oracle and the real path must produce the same counts."""
    vocabulary = build_vocabulary(mini_features)
    oracle = document_frequencies(mini_features)
    for term_id, token in enumerate(vocabulary.tokens):
        assert vocabulary.df[term_id] == oracle[token], token


def test_collection_frequency_agrees_with_the_vocabulary(mini_features) -> None:
    vocabulary = build_vocabulary(mini_features)
    oracle = collection_frequencies(mini_features)
    for term_id, token in enumerate(vocabulary.tokens):
        assert vocabulary.cf[term_id] == oracle[token], token


@settings(max_examples=200, deadline=None)
@given(corpus=_CORPUS)
def test_the_two_implementations_never_disagree(corpus: list[list[str]]) -> None:
    """Searched adversarially rather than checked on one fixture."""
    if not any(corpus):
        # A corpus with no features at all has no vocabulary to compare against;
        # that case is covered by its own test below.
        return
    vocabulary = build_vocabulary(corpus)
    df = document_frequencies(corpus)
    cf = collection_frequencies(corpus)
    for term_id, token in enumerate(vocabulary.tokens):
        assert vocabulary.df[term_id] == df[token]
        assert vocabulary.cf[term_id] == cf[token]


# ---------------------------------------------------------------------------
# The counting rules themselves
# ---------------------------------------------------------------------------
def test_a_repeated_term_counts_once_for_df_and_many_for_cf() -> None:
    """The distinction the two functions exist to make."""
    corpus = [["a", "a", "a", "b"]]
    assert document_frequencies(corpus) == {"a": 1, "b": 1}
    assert collection_frequencies(corpus) == {"a": 3, "b": 1}


def test_an_empty_corpus_and_empty_documents_are_distinguished() -> None:
    assert document_frequencies([]) == {}
    assert document_frequencies([[], []]) == {}
    assert collection_frequencies([[]]) == {}


def test_a_featureless_corpus_has_counts_but_no_vocabulary() -> None:
    """The two disagree here, and correctly so.

    Counting is total -- every corpus has document frequencies, even if they are
    empty -- whereas vocabulary construction treats an empty result as a
    configuration error rather than a valid state, because it almost always means
    ``min_df`` is too high for the corpus.
    """
    assert document_frequencies([[], []]) == {}
    with pytest.raises(EmptyVocabularyError):
        build_vocabulary([[], []])


# ---------------------------------------------------------------------------
# The incremental update (section 4.1)
# ---------------------------------------------------------------------------
@settings(max_examples=200, deadline=None)
@given(corpus=_CORPUS, replacement=_DOCUMENT, index=st.integers(0, 9))
def test_an_incremental_update_matches_recomputing_from_scratch(
    corpus: list[list[str]], replacement: list[str], index: int
) -> None:
    """The whole point of the incremental path: it must not cut a corner.

    An O(nnz of one document) update that disagreed with the O(nnz) recomputation
    would make section 4.1's perturbation experiments measure something other than
    what they claim.
    """
    position = index % len(corpus)
    before = document_frequencies(corpus)

    edited = list(corpus)
    edited[position] = replacement
    expected = document_frequencies(edited)

    actual = df_after_edit(before, removed=corpus[position], added=replacement)
    assert actual == expected


@settings(max_examples=100, deadline=None)
@given(corpus=_CORPUS, replacement=_DOCUMENT, index=st.integers(0, 9))
def test_an_edit_moves_each_term_by_at_most_one(
    corpus: list[list[str]], replacement: list[str], index: int
) -> None:
    """Section 4.1's premise, and what bounds the induced idf shift.

    A single-document edit can add or remove a term for *that* document only, so
    no term's document frequency can move by more than one. If it could, the
    perturbation bounds of section 4.2 would be computed against the wrong
    baseline.
    """
    position = index % len(corpus)
    before = document_frequencies(corpus)
    after = df_after_edit(before, removed=corpus[position], added=replacement)

    for term in set(before) | set(after):
        assert abs(after.get(term, 0) - before.get(term, 0)) <= 1, term


def test_a_term_falling_to_zero_is_dropped_not_kept_at_zero() -> None:
    """A zero entry would make `t in df` true for a term no document contains,
    and `idf` would then be computed for a term outside the vocabulary."""
    before = document_frequencies([["a"], ["b"]])
    after = df_after_edit(before, removed=["a"], added=[])
    assert "a" not in after
    assert after == {"b": 1}


def test_the_input_mapping_is_not_mutated() -> None:
    """Callers hold the pre-edit counts to compute the *shift*; mutating them in
    place would silently make every measured shift zero."""
    before = document_frequencies([["a"], ["a", "b"]])
    snapshot = dict(before)
    df_after_edit(before, removed=["a"], added=["c"])
    assert before == snapshot
