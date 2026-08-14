"""Document- and collection-frequency counting (README section 2.1).

    df(t) = |{ i : t appears at least once in d_i }|

``df`` counts documents; ``cf`` counts occurrences. The paper uses only ``df``,
but ``cf`` is the secondary key of the ``max_features`` truncation rule
(``spec_addenda.md#g6``) and accumulates in the same pass.

:func:`~tfidf_stability.vectorisation.vocabulary.build_vocabulary` computes both,
so these functions exist for inspecting intermediates (README section 1.2) and
for the perturbation analysis of section 4.1, which needs ``df`` before and after
a corpus edit.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

__all__ = [
    "collection_frequencies",
    "df_after_edit",
    "document_frequencies",
]


def document_frequencies(documents: Iterable[Sequence[str]]) -> dict[str, int]:
    """Document frequency of every feature.

    ``set(features)`` caps each document's contribution at 1 per distinct
    feature.
    """
    df: Counter[str] = Counter()
    for features in documents:
        df.update(set(features))
    return dict(df)


def collection_frequencies(documents: Iterable[Sequence[str]]) -> dict[str, int]:
    """Total occurrence count of every feature across the corpus."""
    cf: Counter[str] = Counter()
    for features in documents:
        cf.update(features)
    return dict(cf)


def df_after_edit(
    df: dict[str, int],
    removed: Sequence[str] | None = None,
    added: Sequence[str] | None = None,
) -> dict[str, int]:
    """Document frequency after replacing one document's features.

    Only the edited document's features can change, so this costs
    ``O(nnz of the edited document)`` rather than ``O(nnz)``, which is what makes
    the corpus-perturbation experiments of section 4.1 tractable at scale.

    ``N`` is unchanged by an edit but changes under an addition or removal, and
    ``idf`` depends on both. Callers pass ``N`` to
    :func:`~tfidf_stability.vectorisation.idf.smoothed_idf` themselves; this
    function does not guess.

    Args:
        df: Document frequencies before the edit. Not mutated.
        removed: Features of the document being removed or replaced.
        added: Features of the document being added or substituted in.

    Returns:
        A new mapping. Features whose frequency falls to zero are dropped.
    """
    out = dict(df)
    for f in set(removed or ()):
        remaining = out.get(f, 0) - 1
        if remaining <= 0:
            out.pop(f, None)
        else:
            out[f] = remaining
    for f in set(added or ()):
        out[f] = out.get(f, 0) + 1
    return out
