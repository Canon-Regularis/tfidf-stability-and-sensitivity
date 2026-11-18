"""Identifier orders that decide score ties when a corpus carries no attributes.

``ID_ASC`` is the order :meth:`AttributeTable.from_records` builds; ``ID_DESC`` reverses it,
as trec_eval does; ``SHUFFLE`` orders by a seeded SHA-256 of each identifier, the same anywhere.
Rationale: docs/engineering_notes.md#identifier-tie-orders
"""

from __future__ import annotations

import dataclasses
import hashlib
from enum import Enum

from tfidf_stability.ranking.attributes import AttributeTable

__all__ = ["TieOrder", "tie_order_label", "with_tie_order"]


class TieOrder(str, Enum):
    """How documents with equal scores and attributes are ordered."""

    ID_ASC = "id_asc"
    ID_DESC = "id_desc"
    SHUFFLE = "shuffle"

    def __str__(self) -> str:
        return self.value


def tie_order_label(order: TieOrder, seed: int | None = None) -> str:
    """``id_asc``, ``id_desc`` or ``shuffle_<seed>``, as reports name a variant."""
    return f"{order}_{seed}" if TieOrder(order) is TieOrder.SHUFFLE else str(order)


def with_tie_order(
    table: AttributeTable, order: TieOrder, *, seed: int | None = None
) -> AttributeTable:
    """``table`` with its identifier ranks set by ``order``; columns are unchanged.

    ``seed`` is required for ``SHUFFLE`` and refused otherwise. The result is a bijection
    onto ``0..N-1`` by construction, which :class:`AttributeTable` does not check itself.
    """
    order = TieOrder(order)
    if (order is TieOrder.SHUFFLE) != (seed is not None):
        raise ValueError(f"a seed is required for {TieOrder.SHUFFLE} and only for it: {order}")

    def utf8(i: int) -> bytes:
        return table.doc_ids[i].encode("utf-8")

    def shuffled(i: int) -> tuple[bytes, bytes]:
        digest = hashlib.sha256(f"{seed}\x00{table.doc_ids[i]}".encode()).digest()
        return digest, utf8(i)

    n = table.n_documents
    ranked = sorted(range(n), key=shuffled if order is TieOrder.SHUFFLE else utf8)
    if order is TieOrder.ID_DESC:
        ranked.reverse()
    id_ranks = [0] * n
    for position, i in enumerate(ranked):
        id_ranks[i] = position
    return dataclasses.replace(table, id_ranks=tuple(id_ranks))
