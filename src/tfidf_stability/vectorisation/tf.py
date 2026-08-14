"""Term frequency (README section 2.2).

    tf_i(t) = count_i(t) / sum_{s in V} count_i(s)

The denominator counts in-vocabulary tokens only. Out-of-vocabulary tokens are
discarded after vocabulary construction, so a mostly-filtered document still has
term frequencies summing to 1 over what remains, and a document with no
in-vocabulary tokens maps to the zero vector, as section 2.2 states.

The denominator is an exact integer, accumulated in Python's arbitrary-precision
ints, so ``tf`` is a single correctly-rounded division. The metamorphic test
"concatenating a document with itself leaves tf unchanged" therefore holds
exactly: ``(2c) / (2L)`` and ``c / L`` give identical binary64 values because the
exact rationals are equal and the division is correctly rounded.

The paper does not say so, but ``tf`` rescales each document by the positive
scalar ``1 / L_i``, and cosine similarity is invariant under positive per-vector
scaling, so this normalisation changes no similarity score and no ranking. It
moves only the vector norms, and hence only the perturbation bounds of sections
4.2 and 4.3.
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

    Args:
        features: The document's preprocessed feature stream.
        vocab: The frozen vocabulary.

    Returns:
        A pair ``(tf, L)``: the sparse term-frequency vector and the exact
        in-vocabulary token count. ``L`` is returned because section 4.2's bounds
        need ``||tf||`` and because it is the scalar relating these vectors to
        scikit-learn's (see the module docstring).

        If ``L == 0`` the zero vector is returned, per section 2.2.
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
