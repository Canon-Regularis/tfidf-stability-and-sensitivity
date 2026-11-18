"""Term frequency over the frozen vocabulary (README section 2.2).

``tf_i(t) = count_i(t) / L_i`` with ``L_i`` the exact in-vocabulary token count; ``L_i = 0``
gives the zero vector. The ``1 / L_i`` scale changes no cosine, only the section 4.2-4.3 bounds.
Rationale: docs/engineering_notes.md#tf-exact-integer-denominator
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from tfidf_stability.vectorisation.sparse import SparseVector
from tfidf_stability.vectorisation.vocabulary import Vocabulary

__all__ = ["in_vocabulary_counts", "in_vocabulary_length", "term_frequencies"]


def in_vocabulary_counts(features: Sequence[str], vocab: Vocabulary) -> dict[int, int]:
    """Count occurrences of each in-vocabulary feature, keyed by term identifier.

    Out-of-vocabulary features are dropped here, where section 2.2 says they
    stop contributing.
    """
    counts: Counter[int] = Counter()
    for f in features:
        term_id = vocab.id_of(f)
        if term_id is not None:
            counts[term_id] += 1
    return dict(counts)


def in_vocabulary_length(counts: dict[int, int]) -> int:
    """``L_i = sum_{s in V} count_i(s)``, exactly, as an integer."""
    return sum(counts.values())


def term_frequencies(
    features: Sequence[str],
    vocab: Vocabulary,
) -> tuple[SparseVector, int]:
    """Compute the term-frequency vector of one document.

    Takes the preprocessed ``features`` and the frozen ``vocab``. Returns ``(tf, L)``: the sparse
    vector and the exact in-vocabulary token count, which the section 4.2 bounds and the scale
    relation to scikit-learn's vectors need. ``L == 0`` returns the zero vector (section 2.2).
    """
    counts = in_vocabulary_counts(features, vocab)
    length = in_vocabulary_length(counts)
    if length == 0:
        return SparseVector.zero(len(vocab)), 0

    # One correctly-rounded division per term; the denominator is exact.
    return (
        SparseVector.from_mapping({t: c / length for t, c in counts.items()}, len(vocab)),
        length,
    )
