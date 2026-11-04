"""Add, remove, edit and duplicate documents (README section 4.1).

Each edit records what changed, because the bounds of sections 4.2 and 4.3 are stated
in terms of the difference and cannot be evaluated from the perturbed corpus alone.
Every edit returns a new corpus, so the baseline stays available for comparison.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "Corpus",
    "EditKind",
    "EditRecord",
    "add_document",
    "duplicate_document",
    "edit_document",
    "remove_document",
]

#: A corpus as this module sees it: parallel identifiers and feature streams.
#: Kept apart from ``TfidfModel`` because perturbation happens before
#: vectorisation, which lets the same edit be replayed under different
#: vectoriser configurations.
Corpus = tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]


class EditKind(str, Enum):
    """Which of section 4.1's perturbations was applied."""

    ADD = "add"
    REMOVE = "remove"
    EDIT = "edit"
    DUPLICATE = "duplicate"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class EditRecord:
    """What an edit did, in the terms sections 4.1-4.2 need.

    Holds the ``kind``, the ``doc_id`` affected, the ``removed_features`` and
    ``added_features`` streams, and the sizes ``n_before`` and ``n_after``. Both sizes
    matter: ``idf`` depends on ``N`` as well as ``df`` (section 4.1's competing effects).
    """

    kind: EditKind
    doc_id: str
    removed_features: tuple[str, ...]
    added_features: tuple[str, ...]
    n_before: int
    n_after: int

    def __post_init__(self) -> None:
        """Normalise ``kind`` to its member; the readers select on ``is``."""
        object.__setattr__(self, "kind", EditKind(self.kind))

    @property
    def changes_corpus_size(self) -> bool:
        """Whether ``N`` changed.

        Section 2.2's ``idf(t) = log((1 + N) / (1 + df(t))) + 1`` moves with ``N`` for each term
        whose ``df`` does not move with it; ``df == N`` keeps ``idf`` at exactly ``1.0``. Section
        4.4 certifies a fixed document set: an edit changing ``N`` has no certificate at any ``k``.
        """
        return self.n_before != self.n_after

    @property
    def touched_features(self) -> frozenset[str]:
        """Features whose document frequency can have changed.

        Only these need a ``df`` recomputation, which makes an incremental corpus
        sweep ``O(nnz of the edited document)`` rather than ``O(nnz)``.
        """
        return frozenset(self.removed_features) ^ frozenset(self.added_features)


def _index_of(doc_ids: Sequence[str], doc_id: str) -> int:
    try:
        return list(doc_ids).index(doc_id)
    except ValueError:
        raise KeyError(f"no document with id {doc_id!r}") from None


def add_document(corpus: Corpus, doc_id: str, features: Sequence[str]) -> tuple[Corpus, EditRecord]:
    """Append a document.

    Raises:
        ValueError: If the identifier already exists; duplicate identifiers break
            the strict total order the ranking operator depends on.
    """
    ids, docs = corpus
    if doc_id in ids:
        raise ValueError(f"document id {doc_id!r} already exists")
    return (
        ((*ids, doc_id), (*docs, tuple(features))),
        EditRecord(EditKind.ADD, doc_id, (), tuple(features), len(ids), len(ids) + 1),
    )


def remove_document(corpus: Corpus, doc_id: str) -> tuple[Corpus, EditRecord]:
    """Drop a document."""
    ids, docs = corpus
    i = _index_of(ids, doc_id)
    return (
        ((*ids[:i], *ids[i + 1 :]), (*docs[:i], *docs[i + 1 :])),
        EditRecord(EditKind.REMOVE, doc_id, docs[i], (), len(ids), len(ids) - 1),
    )


def edit_document(
    corpus: Corpus, doc_id: str, features: Sequence[str]
) -> tuple[Corpus, EditRecord]:
    """Replace a document's feature stream, leaving ``N`` unchanged.

    For section 4.2 this isolates the ``df`` effect: with ``N`` fixed, any IDF
    movement comes from ``df`` alone.
    """
    ids, docs = corpus
    i = _index_of(ids, doc_id)
    return (
        (ids, (*docs[:i], tuple(features), *docs[i + 1 :])),
        EditRecord(EditKind.EDIT, doc_id, docs[i], tuple(features), len(ids), len(ids)),
    )


def duplicate_document(corpus: Corpus, doc_id: str, new_id: str) -> tuple[Corpus, EditRecord]:
    """Append an exact copy of an existing document.

    The copy scores identically to the original against every query, so it creates
    an exact tie for the tie-break analysis without a search. It raises ``df`` by one
    for the original's features and ``N`` by one: section 4.1's two competing effects.
    """
    ids, docs = corpus
    i = _index_of(ids, doc_id)
    return add_document(corpus, new_id, docs[i])[0], EditRecord(
        EditKind.DUPLICATE, new_id, (), docs[i], len(ids), len(ids) + 1
    )
