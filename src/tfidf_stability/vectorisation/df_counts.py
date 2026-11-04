"""Document- and collection-frequency counting (README section 2.1).

``df(t) = |{ i : t appears at least once in d_i }|``; ``cf`` counts occurrences, the secondary
``max_features`` key (``spec_addenda.md#g6``). For README 1.2 intermediates and 4.1 corpus edits;
:func:`~tfidf_stability.vectorisation.vocabulary.build_vocabulary` computes both itself.
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
    """Document frequency after one document is removed, added or replaced.

    Returns a copy of ``df`` with each distinct ``removed`` feature down 1 and ``added`` one up 1,
    dropping zeros, in ``O(nnz of the edited document)`` (section 4.1). Callers track ``N``, which
    an addition or removal changes, for :func:`~tfidf_stability.vectorisation.idf.smoothed_idf`.
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
