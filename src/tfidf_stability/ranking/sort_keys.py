"""The three ranking operators of README sections 2.3.1 and 4.5.

Keys ``(-score, rank_1, ..., rank_m, id_rank)`` sort ascending; unique identifiers end every
key, so it is injective (:func:`assert_strict_total_order`). ``pi_score`` is ``pi`` with an
empty priority (README section 4.5). Rationale: docs/engineering_notes.md#sort-key-ordering
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from tfidf_stability.ranking.attributes import AttributeTable

__all__ = [
    "OPERATORS",
    "PI",
    "PI_ALT",
    "PI_SCORE",
    "SortKeySpec",
    "assert_strict_total_order",
    "build_keys",
]

SortKey = tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SortKeySpec:
    """Which attribute columns, in which order, resolve a score tie.

    Attributes:
        name: Stable identity, recorded in the run manifest.
        priority: Attribute names, highest first; the identifier is implicit and must not appear.
    """

    name: str
    priority: tuple[str, ...] = ()

    def digest(self, table: AttributeTable) -> str:
        """Identity of this operator as applied to this table.

        The table's digest is folded in: the same priority over a different
        attribute table is a different ranking function, and a manifest carrying
        only the priority could not distinguish them.
        """
        h = hashlib.sha256()
        h.update(f"{self.name}|{'>'.join(self.priority)}\n".encode())
        h.update(table.digest().encode())
        return h.hexdigest()


#: The operator of README section 2.3.1.
PI: Final = SortKeySpec("pi", ("popularity", "rating", "engagement"))

#: Section 4.5's score-only baseline: ties fall straight through to the
#: identifier, so any difference from PI is attributable to the attributes.
PI_SCORE: Final = SortKeySpec("pi_score", ())

#: Section 4.5's alternate priority. The paper says "reordered" without naming
#: which of the 3! orderings; pinned here to the reversal, the antipode that
#: maximises distance from PI's priority. Proposed as addendum G15; the full
#: permutation sweep is available as an ablation.
PI_ALT: Final = SortKeySpec("pi_alt", ("engagement", "rating", "popularity"))

OPERATORS: Final[tuple[SortKeySpec, ...]] = (PI, PI_SCORE, PI_ALT)


def build_keys(
    scores: Sequence[float],
    table: AttributeTable,
    spec: SortKeySpec = PI,
) -> list[SortKey]:
    """Build one sort key per document, in document order.

    ``scores`` must be finite (the caller checks), index-aligned to the rank-encoded ``table``;
    ``spec`` selects the operator. Sorting the keys ascending yields the ranking. Raises
    ValueError if the score count differs from the table or the priority names the identifier.
    """
    if len(scores) != table.n_documents:
        raise ValueError(
            f"{len(scores)} scores but the attribute table has {table.n_documents} documents"
        )
    if "identifier" in spec.priority or "doc_id" in spec.priority:
        raise ValueError(
            "the identifier terminates every key implicitly and must not appear in "
            "a priority: moving it would make the ordering non-total"
        )

    rank_rows = table.rank_matrix(spec.priority)
    id_ranks = table.id_ranks
    return [
        (-scores[i], *(row[i] for row in rank_rows), id_ranks[i]) for i in range(table.n_documents)
    ]


def assert_strict_total_order(keys: Sequence[SortKey]) -> None:
    """Verify the comparator axioms and key injectivity.

    A self-test of the comparator, not the data: a score-only key is not injective under a
    score tie. For tests and an opt-in debug mode, never the published path: the transitivity
    check is O(n^3). Raises AssertionError if two keys are equal or an axiom fails.
    """
    n = len(keys)
    if len(set(keys)) != n:
        raise AssertionError(
            f"the sort key is not injective: {n - len(set(keys))} duplicate key(s). "
            f"The ranking operator is then only a weak order, and the sorted "
            f"permutation is not unique."
        )
    for a in range(n):
        if keys[a] < keys[a]:
            raise AssertionError(f"not irreflexive at {a}")
        for b in range(a + 1, n):
            ab, ba = keys[a] < keys[b], keys[b] < keys[a]
            if ab and ba:
                raise AssertionError(f"not asymmetric at ({a}, {b})")
            if not ab and not ba:
                raise AssertionError(f"not trichotomous at ({a}, {b}): keys compare equal")
    if n <= 40:  # O(n^3), so only at the sizes tests use
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    if keys[a] < keys[b] and keys[b] < keys[c] and not keys[a] < keys[c]:
                        raise AssertionError(f"not transitive at ({a}, {b}, {c})")
