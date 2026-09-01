"""Corpus perturbations (README section 4.1).

Section 4.1 considers "a perturbation of the corpus induced by adding or removing
a document, or by modifying the token content of an existing document". This
module makes those three edits concrete and records what changed, since the
bounds of sections 4.2 and 4.3 are stated in terms of the difference and cannot
be evaluated from the perturbed corpus alone.

Every edit returns a new corpus. A perturbation experiment needs both sides
simultaneously, and an in-place edit would destroy the baseline it is compared
against.
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

    Attributes:
        kind: Which perturbation.
        doc_id: The document added, removed or modified.
        removed_features: Feature stream that left the corpus, if any.
        added_features: Feature stream that entered, if any.
        n_before, n_after: Corpus sizes. Both matter: ``idf`` depends on ``N`` as
            well as on ``df``, so an edit leaving every ``df`` unchanged can
            still move every ``idf`` (section 4.1's "competing effects").
    """

    kind: EditKind
    doc_id: str
    removed_features: tuple[str, ...]
    added_features: tuple[str, ...]
    n_before: int
    n_after: int

    @property
    def changes_corpus_size(self) -> bool:
        """Whether ``N`` changed.

        Not "and hence every IDF moved", which this said and the arithmetic does
        not support. Section 2.2 fixes ``idf(t) = log((1 + N) / (1 + df(t))) + 1``,
        so a token appearing in every document has ``df == N``, the ratio is
        exactly 1, and ``idf`` is exactly ``1.0`` whatever ``N`` is -- measured
        at N = 5, 6, 100 and 101. An edit that changes ``N`` moves the IDF of
        every term whose ``df`` did not move with it, which is most of them but
        not all.

        What the property is for is unaffected: section 4.4's certificate bounds
        the movement of scores over a *fixed* document set, so an edit changing
        ``N`` has no certificate at any ``k`` regardless of which IDFs moved.
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
        ValueError: If the identifier already exists. Duplicate identifiers break
            the strict total order the ranking operator depends on, so they are
            refused here rather than surfacing later as a non-deterministic
            ranking.
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

    The cleanest of the three for section 4.2: with ``N`` fixed, any IDF movement
    comes from ``df`` alone.
    """
    ids, docs = corpus
    i = _index_of(ids, doc_id)
    return (
        (ids, (*docs[:i], tuple(features), *docs[i + 1 :])),
        EditRecord(EditKind.EDIT, doc_id, docs[i], tuple(features), len(ids), len(ids)),
    )


def duplicate_document(corpus: Corpus, doc_id: str, new_id: str) -> tuple[Corpus, EditRecord]:
    """Append an exact copy of an existing document.

    The most useful perturbation for the tie-break analysis: the copy scores
    identically to the original against every query, so it manufactures an exact
    tie without a search. It also raises ``df`` by one for the original's
    features alone while raising ``N`` by one, section 4.1's two competing
    effects in their purest form.
    """
    ids, docs = corpus
    i = _index_of(ids, doc_id)
    return add_document(corpus, new_id, docs[i])[0], EditRecord(
        EditKind.DUPLICATE, new_id, (), docs[i], len(ids), len(ids) + 1
    )
