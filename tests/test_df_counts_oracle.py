"""``df_counts`` as a checked oracle for the counts the vocabulary builds.

``df_counts`` is a second implementation of document and collection frequency,
so it is held to agreement with ``vocabulary.py``, and ``df_after_edit`` is checked
against a from-scratch recomputation. Rationale: docs/engineering_notes.md#df-counts-oracle
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
# Agreement with the implementation the vocabulary uses
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
    """The two implementations agree on every generated corpus."""
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
    """A repeated term counts once for ``df`` and once per occurrence for ``cf``."""
    corpus = [["a", "a", "a", "b"]]
    assert document_frequencies(corpus) == {"a": 1, "b": 1}
    assert collection_frequencies(corpus) == {"a": 3, "b": 1}


def test_an_empty_corpus_and_empty_documents_are_distinguished() -> None:
    assert document_frequencies([]) == {}
    assert document_frequencies([[], []]) == {}
    assert collection_frequencies([[]]) == {}


def test_a_featureless_corpus_has_counts_but_no_vocabulary() -> None:
    """A featureless corpus has empty counts but no vocabulary.

    Counting is total, so every corpus has document frequencies. Vocabulary
    construction treats an empty result as a configuration error, since it
    usually means ``min_df`` is too high for the corpus.
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
    """The incremental update equals a from-scratch recomputation.

    ``df_after_edit`` is the O(nnz of one document) update for section 4.1's
    perturbation analysis; it must equal ``document_frequencies`` of the
    edited corpus, the O(nnz) recomputation.
    """
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
    """A single-document edit moves no term's document frequency by more than one.

    The edit adds or removes a term for that document alone. This bound is the
    premise of section 4.1, which bounds the induced idf shift, and section
    4.2's perturbation bounds rest on it.
    """
    position = index % len(corpus)
    before = document_frequencies(corpus)
    after = df_after_edit(before, removed=corpus[position], added=replacement)

    for term in set(before) | set(after):
        assert abs(after.get(term, 0) - before.get(term, 0)) <= 1, term


def test_a_term_falling_to_zero_is_dropped_not_kept_at_zero() -> None:
    """A term whose count falls to zero is removed from the mapping.

    A zero entry would make ``t in df`` true for a term no document contains,
    and idf would then be computed for a term outside the vocabulary.
    """
    before = document_frequencies([["a"], ["b"]])
    after = df_after_edit(before, removed=["a"], added=[])
    assert "a" not in after
    assert after == {"b": 1}


def test_the_input_mapping_is_not_mutated() -> None:
    """The pre-edit mapping is left unchanged by ``df_after_edit``.

    Callers hold the pre-edit counts to compute the shift, so in-place
    mutation would make every measured shift zero.
    """
    before = document_frequencies([["a"], ["a", "b"]])
    snapshot = dict(before)
    df_after_edit(before, removed=["a"], added=["c"])
    assert before == snapshot


# ---------------------------------------------------------------------------
# The two counts differ exactly where a document repeats a token
# ---------------------------------------------------------------------------
def test_a_repeated_token_counts_once_per_document_and_once_per_occurrence() -> None:
    """``df`` counts a repeated token once per document; ``cf`` once per occurrence.

    G6's truncation policy ranks on both, so a collapse of one into the other
    would reorder the vocabulary at the ``max_features`` boundary.
    """
    corpus = [["a", "a", "b"]]

    assert document_frequencies(corpus) == {"a": 1, "b": 1}
    assert collection_frequencies(corpus) == {"a": 2, "b": 1}


@pytest.mark.parametrize("counter", [document_frequencies, collection_frequencies])
def test_a_corpus_of_no_documents_has_no_counts(counter: object) -> None:
    """A corpus of no documents yields an empty mapping, not an error.

    Counting is separable from deciding whether the result can build a
    vocabulary, and only the second is an error (G17).
    """
    assert counter([]) == {}  # type: ignore[operator]


@pytest.mark.parametrize("counter", [document_frequencies, collection_frequencies])
def test_documents_with_no_features_contribute_nothing_but_still_count(
    counter: object,
) -> None:
    """A document with no features contributes no counts.

    An all-stopword document survives preprocessing as an empty feature
    stream. It must not be dropped from the corpus size, which is why
    ``build_vocabulary`` tracks ``n_docs`` separately from the counters.
    """
    assert counter([[], ["a"], []]) == {"a": 1}  # type: ignore[operator]


def test_the_two_counts_agree_exactly_when_no_document_repeats_a_token() -> None:
    """The two counts coincide when no document repeats a token."""
    corpus = [["a", "b"], ["b", "c"]]
    assert document_frequencies(corpus) == collection_frequencies(corpus)
