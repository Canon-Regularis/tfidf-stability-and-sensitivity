"""``df_counts`` as a checked oracle for the counts the vocabulary builds.

Nothing in the package imports this module; it sat at 0% coverage. It is a second
implementation of document and collection frequency, `vocabulary.py` computes
`df` during vocabulary construction, and a disagreement would make every idf, and
every score, depend on which one a caller reached for. These tests turn the
duplication into a differential oracle in the sense the reference-versus-native
suite uses.

`df_after_edit` is the incremental update section 4.1 needs. Its property, that an
edit changes `df` by at most one per term, is asserted here against a
from-scratch recomputation.
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


@pytest.mark.property
@settings(max_examples=200, deadline=None)
@given(corpus=_CORPUS)
def test_the_two_implementations_never_disagree(corpus: list[list[str]]) -> None:
    """Searched adversarially rather than checked on one fixture."""
    if not any(corpus):
        # No features means no vocabulary to compare against; covered below.
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
    """The two disagree here, correctly. Counting is total: every corpus has
    document frequencies, possibly empty. Vocabulary construction treats an empty
    result as a configuration error, since it almost always means ``min_df`` is
    too high for the corpus.
    """
    assert document_frequencies([[], []]) == {}
    with pytest.raises(EmptyVocabularyError, match="0 distinct features"):
        build_vocabulary([[], []])


# ---------------------------------------------------------------------------
# The incremental update (section 4.1)
# ---------------------------------------------------------------------------
@pytest.mark.property
@settings(max_examples=200, deadline=None)
@given(corpus=_CORPUS, replacement=_DOCUMENT, index=st.integers(0, 9))
def test_an_incremental_update_matches_recomputing_from_scratch(
    corpus: list[list[str]], replacement: list[str], index: int
) -> None:
    """An O(nnz of one document) update disagreeing with the O(nnz) recomputation
    would make section 4.1's perturbation experiments measure something other
    than what they claim."""
    position = index % len(corpus)
    before = document_frequencies(corpus)

    edited = list(corpus)
    edited[position] = replacement
    expected = document_frequencies(edited)

    actual = df_after_edit(before, removed=corpus[position], added=replacement)
    assert actual == expected


@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(corpus=_CORPUS, replacement=_DOCUMENT, index=st.integers(0, 9))
def test_an_edit_moves_each_term_by_at_most_one(
    corpus: list[list[str]], replacement: list[str], index: int
) -> None:
    """Section 4.1's premise, which bounds the induced idf shift: a single-document
    edit adds or removes a term for that document alone, so no term's document
    frequency moves by more than one. Otherwise section 4.2's perturbation bounds
    sit on the wrong baseline.
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
    """Callers hold the pre-edit counts to compute the shift; mutating them in
    place makes every measured shift zero."""
    before = document_frequencies([["a"], ["a", "b"]])
    snapshot = dict(before)
    df_after_edit(before, removed=["a"], added=["c"])
    assert before == snapshot


# ---------------------------------------------------------------------------
# The two counts differ exactly where a document repeats a token
# ---------------------------------------------------------------------------
def test_a_repeated_token_counts_once_per_document_and_once_per_occurrence() -> None:
    """The whole distinction between the two functions. `df` is set-valued per
    document, `cf` is not, and G6's truncation policy ranks on both -- so a
    collapse of one into the other would silently reorder the vocabulary at the
    `max_features` boundary.
    """
    corpus = [["a", "a", "b"]]

    assert document_frequencies(corpus) == {"a": 1, "b": 1}
    assert collection_frequencies(corpus) == {"a": 2, "b": 1}


@pytest.mark.parametrize("counter", [document_frequencies, collection_frequencies])
def test_a_corpus_of_no_documents_has_no_counts(counter: object) -> None:
    """An empty mapping rather than an error: counting is separable from
    deciding whether the result can build a vocabulary, and only the second is
    an error (G17)."""
    assert counter([]) == {}  # type: ignore[operator]


@pytest.mark.parametrize("counter", [document_frequencies, collection_frequencies])
def test_documents_with_no_features_contribute_nothing_but_still_count(
    counter: object,
) -> None:
    """An all-stopword document survives preprocessing as an empty feature
    stream. It contributes no counts and must not be dropped from the corpus
    size -- which is why `build_vocabulary` tracks `n_docs` separately from the
    counters."""
    assert counter([[], ["a"], []]) == {"a": 1}  # type: ignore[operator]


def test_the_two_counts_agree_exactly_when_no_document_repeats_a_token() -> None:
    """The condition under which the distinction collapses, stated so the tests
    above are read as being about repetition rather than about the corpus."""
    corpus = [["a", "b"], ["b", "c"]]
    assert document_frequencies(corpus) == collection_frequencies(corpus)
