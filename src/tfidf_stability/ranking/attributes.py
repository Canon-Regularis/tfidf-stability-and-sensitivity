"""Secondary sort attributes (README section 2.3.1, ``spec_addenda.md#g8``).

Score ties are resolved lexicographically on a fixed attribute tuple: popularity,
rating, engagement, identifier. Every attribute is converted to a dense integer
rank at construction time, an ``int32`` in which smaller means earlier, with
direction (``desc``/``asc``) and missing-value placement folded into the rank.

Consequences:

1. The comparator degenerates to a plain ascending lexicographic comparison of
   ``(float, int, int, int, int)``: no direction branch, no missing-value branch,
   no rational arithmetic inside the sort.
2. No floating point survives in the tie-break, which is G8's requirement. The
   only ``double`` in the key is the similarity score.
3. A NaN cannot enter through an attribute, as a type-level guarantee rather
   than a runtime check. A NaN in a sort key destroys the strict weak ordering,
   which is undefined behaviour in ``std::sort`` and shows up as an
   out-of-bounds write.
4. The C++ mirror receives the rank matrix as data and never re-derives it, so
   the cross-language question becomes the integer equality
   ``py_ranks == cpp_ranks``. Same move ``spec_addenda.md#g13`` makes for
   ``idf``.

Why not ``fractions.Fraction`` in the sort key
----------------------------------------------
It has no C++ counterpart, so the two languages would run structurally different
algorithms over a result the test suite asserts is identical: the drift that
produced this project's ``pairwise_sum`` bug, where two legitimate formulations
agreed on every input until n = 129. It also runs ``math.gcd`` and allocates an
object per element, and ``Fraction(0, 0)`` raises, so an absent rating still
needs a separate ``has_value`` bit. Cross-multiplication is the right rule; it
belongs in the rank construction, evaluated once over the distinct values,
rather than inside a sort that runs per query.
"""

from __future__ import annotations

import functools
import hashlib
import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from tfidf_stability.utils.validation import (
    TfidfStabilityError,
    check_finite,
    check_unique_ids,
)

__all__ = [
    "DEFAULT_SPECS",
    "AttributeColumn",
    "AttributeDType",
    "AttributeSpec",
    "AttributeTable",
    "Direction",
    "MissingPolicy",
    "check_ratio_fits_int64",
    "ratio_less",
]

#: Products in the C++ cross-multiplication must fit in int64. MovieLens-scale
#: magnitudes (2*sum <= 1e6, count <= 1e5) give products near 1e11, four orders
#: of magnitude below this; the check exists for data that is not MovieLens.
_INT64_MAX: Final = (1 << 63) - 1


class Direction(str, Enum):
    """Whether larger or smaller values rank earlier."""

    DESC = "desc"
    ASC = "asc"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class AttributeDType(str, Enum):
    """How an attribute's values are represented."""

    #: Plain integers: popularity, engagement.
    INT64 = "int64"
    #: G8's exact mean, as the pair ``(2 * sum_rating, count)``. Compared by
    #: cross-multiplication, so no division is ever performed.
    RATIO_I64 = "ratio_i64"
    #: Permitted but discouraged. A derived mean belongs in RATIO_I64.
    FLOAT64 = "float64"
    #: Text, ordered by UTF-8 bytes. Matches ``vocabulary.py`` and is
    #: reproducible in C++ via ``memcmp``, unlike locale or Unicode collation.
    BYTES = "bytes"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class MissingPolicy(str, Enum):
    """Where documents lacking a value are placed."""

    LAST = "last"
    FIRST = "first"
    #: Absence is a data error. Used for the identifier, which must always exist
    #: because it is what makes the ordering total.
    FORBID = "forbid"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass(frozen=True, slots=True)
class AttributeSpec:
    """The pinned semantics of one attribute (G8)."""

    name: str
    direction: Direction = Direction.DESC
    dtype: AttributeDType = AttributeDType.INT64
    missing_policy: MissingPolicy = MissingPolicy.LAST


#: The tuple README section 2.3.1 names, minus the identifier, which the sort key
#: appends implicitly and never permutes.
DEFAULT_SPECS: Final[tuple[AttributeSpec, ...]] = (
    AttributeSpec("popularity", Direction.DESC, AttributeDType.INT64),
    AttributeSpec("rating", Direction.DESC, AttributeDType.RATIO_I64),
    AttributeSpec("engagement", Direction.DESC, AttributeDType.INT64),
)


# ---------------------------------------------------------------------------
# Exact rational comparison
# ---------------------------------------------------------------------------
def ratio_less(a_num: int, a_den: int, b_num: int, b_den: int) -> bool:
    """``a_num/a_den < b_num/b_den``, exactly, by cross-multiplication (G8).

    Both denominators must be strictly positive, which the column invariant
    guarantees: a zero denominator means "no rating", is carried by the
    ``has_value`` bit, and never reaches here.

    Python ints are arbitrary precision, so this cannot overflow. The C++ mirror
    uses ``__int128``, and :func:`check_ratio_fits_int64` guards it at
    construction, so the native inner loop needs no checked multiply.
    """
    return a_num * b_den < b_num * a_den


def check_ratio_fits_int64(pairs: Sequence[tuple[int, int]], what: str) -> None:
    """Reject ratio data whose cross-products would overflow the C++ mirror.

    The products that occur are ``num_i * den_j`` for distinct ``i != j``, since
    nothing is compared with itself. Bounding by ``max(num) * max(den)`` over the
    whole column is therefore too conservative: when the largest numerator and
    the largest denominator belong to the same document, which is the usual case,
    that product never arises and rejecting on it would be a false positive.

    The exact maximum over ``i != j`` comes from the two largest numerators and
    the two largest denominators: a maximising pair must draw from those, and at
    least one such combination avoids the index collision. O(n), and exact.
    """
    if len(pairs) < 2:
        return  # nothing is ever compared with itself

    def _top2(values: Sequence[int]) -> list[tuple[int, int]]:
        """The two largest ``(value, index)`` pairs, largest first."""
        return sorted(((abs(v), i) for i, v in enumerate(values)), reverse=True)[:2]

    top_nums = _top2([n for n, _ in pairs])
    top_dens = _top2([d for _, d in pairs])

    worst = 0
    worst_pair: tuple[int, int] = (0, 0)
    for num, i in top_nums:
        for den, j in top_dens:
            if i != j and num * den > worst:
                worst = num * den
                worst_pair = (num, den)

    if worst > _INT64_MAX:
        raise TfidfStabilityError(
            f"{what}: comparing a numerator of {worst_pair[0]} against a denominator "
            f"of {worst_pair[1]} gives {worst}, which overflows int64, so the native "
            f"backend could not reproduce this ordering."
        )


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class AttributeColumn:
    """One attribute's values for every document, plus its integer ranks.

    Attributes:
        spec: The pinned semantics.
        has_value: Per document. Absence is an explicit bit, never NaN.
        ranks: Dense ``int32``-compatible ranks; smaller sorts earlier. Direction
            and missing placement are already folded in, so a consumer needs
            neither.
        n_distinct: Number of distinct present values.
        values: The raw values, retained for reporting (section 7.4 prints the
            tie-break attributes of a near-tie pair).
    """

    spec: AttributeSpec
    has_value: tuple[bool, ...]
    ranks: tuple[int, ...]
    n_distinct: int
    values: tuple[Any, ...]

    def __len__(self) -> int:
        return len(self.ranks)

    def value_of(self, i: int) -> Any:
        """The raw value of document ``i``, or ``None`` if absent."""
        return self.values[i] if self.has_value[i] else None


def _order_distinct(values: Sequence[Any], spec: AttributeSpec) -> list[Any]:
    """Sort the distinct present values into rank order (earliest first)."""
    distinct = list(dict.fromkeys(values))  # order-preserving unique
    if spec.dtype is AttributeDType.RATIO_I64:
        distinct.sort(key=functools.cmp_to_key(_ratio_cmp))
    elif spec.dtype is AttributeDType.BYTES:
        distinct.sort(key=lambda s: str(s).encode("utf-8"))
    else:
        distinct.sort()
    if spec.direction is Direction.DESC:
        distinct.reverse()
    return distinct


def _ratio_cmp(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Ascending comparison of exact rationals, for sorting distinct values only.

    ``cmp_to_key`` is affordable here because it runs over the ``m`` distinct
    pairs once per corpus, rather than over ``N`` documents inside a sort that
    executes per query.
    """
    if ratio_less(a[0], a[1], b[0], b[1]):
        return -1
    if ratio_less(b[0], b[1], a[0], a[1]):
        return 1
    return 0


def _dense_positions(ordered: Sequence[Any], spec: AttributeSpec) -> dict[Any, int]:
    """Rank the sorted distinct values by equivalence class rather than identity.

    Only visible for :data:`AttributeDType.RATIO_I64`. ``_order_distinct``
    deduplicates with ``dict.fromkeys``, which is tuple equality, so ``(14, 2)``
    and ``(21, 3)`` both survive even though ``_ratio_cmp`` reports them equal
    (14/2 == 21/3). Numbering the survivors ``enumerate``-style would hand two
    documents with the same mean rating two different ranks.

    Two guarantees break together if it does: the rating component stops tying
    where G8 says it must, so the tie-break never falls through to engagement;
    and since ``dict.fromkeys`` preserves insertion order, which of the two equal
    representations sorts first depends on the order the records arrived in,
    making the ranking depend on corpus order, which section 2.3.1's total-order
    argument forbids.

    Ranks are therefore dense: consecutive entries that compare equal share a
    rank, and the next class takes the next integer.
    """
    if not ordered:
        return {}

    # Only ratios can have two distinct representations of one value. Every other
    # dtype was deduplicated by `==` under an ordering consistent with its sort
    # key, so its survivors are genuinely distinct.
    ratios = spec.dtype is AttributeDType.RATIO_I64

    positions: dict[Any, int] = {ordered[0]: 0}
    rank = 0
    for previous, current in itertools.pairwise(ordered):
        if not (ratios and _ratio_cmp(previous, current) == 0):
            rank += 1
        positions[current] = rank
    return positions


def _build_column(
    spec: AttributeSpec,
    values: Sequence[Any],
    has_value: Sequence[bool],
) -> AttributeColumn:
    """Rank-encode one column."""
    present = [v for v, ok in zip(values, has_value, strict=True) if ok]

    if spec.dtype is AttributeDType.FLOAT64:
        check_finite([float(v) for v in present], f"attribute {spec.name!r}")
    if spec.dtype is AttributeDType.RATIO_I64:
        check_ratio_fits_int64(present, f"attribute {spec.name!r}")

    ordered = _order_distinct(present, spec)
    position = _dense_positions(ordered, spec)
    # Counts classes, which can be fewer than the representations kept; see
    # `_dense_positions`.
    n_distinct = (max(position.values()) + 1) if position else 0

    # Missing sorts last by taking the rank one past every present value, or
    # first by shifting the present values up and taking 0.
    if spec.missing_policy is MissingPolicy.FIRST:
        rank_of = {v: i + 1 for v, i in position.items()}
        missing_rank = 0
    else:
        rank_of = position
        missing_rank = n_distinct

    ranks = tuple(
        rank_of[v] if ok else missing_rank for v, ok in zip(values, has_value, strict=True)
    )
    return AttributeColumn(
        spec=spec,
        has_value=tuple(has_value),
        ranks=ranks,
        n_distinct=n_distinct,
        values=tuple(values),
    )


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class AttributeTable:
    """Every tie-break attribute for a corpus, rank-encoded.

    ``id_ranks`` is a bijection onto ``0..N-1`` given by UTF-8 byte order of the
    document identifiers. It terminates every sort key and makes the ordering a
    strict total order rather than a weak one, which in turn makes the sorted
    permutation unique and sort stability irrelevant. Identifier uniqueness is
    validated rather than assumed.
    """

    doc_ids: tuple[str, ...]
    columns: tuple[AttributeColumn, ...]
    id_ranks: tuple[int, ...]

    @property
    def n_documents(self) -> int:
        return len(self.doc_ids)

    def names(self) -> tuple[str, ...]:
        return tuple(c.spec.name for c in self.columns)

    def column(self, name: str) -> AttributeColumn:
        for c in self.columns:
            if c.spec.name == name:
                return c
        raise KeyError(f"no attribute named {name!r}; have {self.names()}")

    def rank_matrix(self, priority: Sequence[str]) -> tuple[tuple[int, ...], ...]:
        """Rank rows for the named attributes, in the given priority order.

        This is what crosses the language boundary: the native backend receives
        these integers and never recomputes them.
        """
        return tuple(self.column(name).ranks for name in priority)

    def attributes_of(self, i: int) -> dict[str, Any]:
        """Raw attribute values of document ``i``, for reporting (section 7.4)."""
        out: dict[str, Any] = {"doc_id": self.doc_ids[i], "id_rank": self.id_ranks[i]}
        for c in self.columns:
            out[c.spec.name] = c.value_of(i)
        return out

    def digest(self) -> str:
        """SHA-256 over the specs and the integer ranks, for the run manifest.

        Over the ranks rather than the raw values: the ranks determine the
        ordering, so two tables with identical digests provably induce identical
        rankings.
        """
        h = hashlib.sha256()
        for c in self.columns:
            h.update(
                f"{c.spec.name}|{c.spec.direction}|{c.spec.dtype}|"
                f"{c.spec.missing_policy}|{c.n_distinct}\n".encode()
            )
            h.update(repr(c.ranks).encode())
            h.update(repr(c.has_value).encode())
        h.update(repr(self.id_ranks).encode())
        return h.hexdigest()

    @classmethod
    def from_records(
        cls,
        records: Sequence[Mapping[str, Any]],
        specs: Sequence[AttributeSpec] = DEFAULT_SPECS,
        *,
        id_field: str = "doc_id",
    ) -> AttributeTable:
        """Build from per-document mappings.

        Reads the field names ``tests/fixtures/mini_corpus.jsonl`` already
        carries. A ``RATIO_I64`` attribute named ``rating`` is assembled from
        ``rating_sum2`` and ``rating_count``: the fixture was written to G8's
        exact-pair representation, so it is honoured rather than replaced.

        Raises:
            DuplicateIdentifierError: If two documents share an identifier.
            TfidfStabilityError: On non-finite floats, inconsistent ratio pairs,
                a violated ``FORBID`` policy, or int64 overflow risk.
        """
        doc_ids = [str(r[id_field]) for r in records]
        check_unique_ids(doc_ids)

        columns: list[AttributeColumn] = []
        for spec in specs:
            values, has_value = _extract(records, spec)
            columns.append(_build_column(spec, values, has_value))

        # The identifier's own rank: a bijection by UTF-8 byte order.
        order = sorted(range(len(doc_ids)), key=lambda i: doc_ids[i].encode("utf-8"))
        id_ranks = [0] * len(doc_ids)
        for position, i in enumerate(order):
            id_ranks[i] = position

        return cls(tuple(doc_ids), tuple(columns), tuple(id_ranks))


def _extract(
    records: Sequence[Mapping[str, Any]], spec: AttributeSpec
) -> tuple[list[Any], list[bool]]:
    """Pull one attribute out of the records, with its presence bits."""
    values: list[Any] = []
    has_value: list[bool] = []
    value: Any

    for i, r in enumerate(records):
        if spec.dtype is AttributeDType.RATIO_I64:
            num = r.get(f"{spec.name}_sum2")
            den = r.get(f"{spec.name}_count")
            if num is None or den is None or int(den) == 0:
                if num is not None and den is not None and int(num) != 0 and int(den) == 0:
                    raise TfidfStabilityError(
                        f"document {i}: {spec.name}_count is 0 but {spec.name}_sum2 is "
                        f"{num}, which is inconsistent"
                    )
                present, value = False, (0, 1)
            else:
                if int(den) < 0:
                    raise TfidfStabilityError(
                        f"document {i}: {spec.name}_count is negative ({den})"
                    )
                present, value = True, (int(num), int(den))
        else:
            raw = r.get(spec.name)
            present = raw is not None
            if not present:
                value = 0 if spec.dtype is not AttributeDType.BYTES else ""
            elif spec.dtype is AttributeDType.INT64:
                value = int(raw)  # type: ignore[arg-type]
            elif spec.dtype is AttributeDType.FLOAT64:
                value = float(raw)  # type: ignore[arg-type]
            else:
                value = str(raw)

        if not present and spec.missing_policy is MissingPolicy.FORBID:
            raise TfidfStabilityError(
                f"document {i}: attribute {spec.name!r} is missing but its policy is FORBID"
            )
        values.append(value)
        has_value.append(present)

    return values, has_value
