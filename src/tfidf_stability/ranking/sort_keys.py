"""The three ranking operators (README sections 2.3.1 and 4.5).

    pi       = Sort(s_i, a_i)            the full attribute tuple, lexicographic
    pi_score = Sort(s_i, id_i)           attribute-independent baseline
    pi_alt   = Sort(s_i, a_i reordered)  alternate priority

The key is ``(-score, rank_1, ..., rank_m, id_rank)``, compared ascending and
lexicographically.

Negating a binary64 flips the sign bit and never rounds, so ``-s`` is exact and
order-reversing, and Python's tuple ``<`` and C++'s ``operator<`` are then the
same relation: one comparator to reason about instead of two.

Identifiers are unique and terminate every key, so the key is injective and the
comparator is a strict total order. Move the identifier or drop it and the order
stops being total, at which point the sorted permutation is no longer unique.
See :func:`assert_strict_total_order`.

``pi_score`` is the empty-priority case of ``pi`` rather than a separate code
path (``spec_addenda.md`` G15). It therefore agrees with ``pi`` whenever the
attributes fail to discriminate, which weakens the ablation slightly and is the
correct reading of section 4.5.
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
        priority: Attribute names, highest priority first. The identifier is
            appended implicitly and must not appear here.
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
    """Build one sort key per document.

    Args:
        scores: Similarity scores, index-aligned to the table's documents.
            Must be finite; the caller checks that once, before reaching here.
        table: The rank-encoded attribute table.
        spec: Which operator.

    Returns:
        Keys in document order. Sorting them ascending yields the ranking.

    Raises:
        ValueError: If the score count does not match the table, or if the
            priority names the identifier.
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

    A self-test of the comparator rather than of the data: it catches the classic
    "sorted by score only" bug, where the key stops being injective and the
    sorted permutation stops being unique.

    For tests and an opt-in debug mode; never on the published path, since the
    transitivity check is O(n^3).

    Raises:
        AssertionError: If any axiom fails.
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
