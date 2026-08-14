"""Sparse vector and matrix primitives.

Index order and reduction policy together are what make the reference and native
backends bit-identical.

*Index order.* A :class:`SparseVector` stores its indices strictly ascending,
which fixes the order terms accumulate in a dot product; floating-point addition
is not associative, so a different order is a different number. The native
backend's term-at-a-time scoring loop also visits terms in ascending identifier
order, so it matches :func:`dot` bit for bit.

*Reduction policy.* Every sum takes an explicit
:class:`~tfidf_stability.utils.numerics.Reduction`. The default ``NAIVE`` is the
plain left-to-right fold section 2.3 specifies.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise

from tfidf_stability.utils.numerics import Reduction, reduce_sum, sqrt

__all__ = ["CsrMatrix", "SparseVector", "cosine_of", "dot", "l2_norm"]


@dataclass(frozen=True, slots=True)
class SparseVector:
    """A sparse vector with strictly ascending indices.

    Attributes:
        indices: Term identifiers, strictly ascending.
        values: Parallel values.
        dim: Ambient dimension (``|V|``), retained so operations can check
            compatibility instead of silently producing nonsense.
    """

    indices: tuple[int, ...]
    values: tuple[float, ...]
    dim: int

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.values):
            raise ValueError(
                f"indices and values differ in length: {len(self.indices)} vs {len(self.values)}"
            )

    @property
    def nnz(self) -> int:
        """Number of stored (structurally non-zero) entries."""
        return len(self.indices)

    def __len__(self) -> int:
        return self.dim

    def __iter__(self) -> Iterator[tuple[int, float]]:
        return zip(self.indices, self.values, strict=True)

    def is_canonical(self) -> bool:
        """Whether indices are strictly ascending and within range.

        O(nnz), so it is checked in tests rather than on every construction; the
        constructors below establish it.
        """
        return all(a < b for a, b in pairwise(self.indices)) and all(
            0 <= i < self.dim for i in self.indices
        )

    def to_dense(self) -> list[float]:
        """Expand to a dense list. For inspection and small-scale testing only."""
        out = [0.0] * self.dim
        for i, v in zip(self.indices, self.values, strict=True):
            out[i] = v
        return out

    @classmethod
    def from_mapping(cls, mapping: Mapping[int, float], dim: int) -> SparseVector:
        """Build from ``{term_id: value}``, sorting indices ascending.

        The sort gives canonical order whatever the mapping's iteration order:
        the point at which dictionary ordering stops being able to influence any
        downstream number.
        """
        items = sorted(mapping.items())
        return cls(
            indices=tuple(i for i, _ in items),
            values=tuple(v for _, v in items),
            dim=dim,
        )

    @classmethod
    def zero(cls, dim: int) -> SparseVector:
        """The zero vector, which section 2.3 gives a similarity of 0."""
        return cls(indices=(), values=(), dim=dim)


def dot(u: SparseVector, v: SparseVector, policy: Reduction = Reduction.NAIVE) -> float:
    """Inner product of two sparse vectors.

    A merge over the two ascending index lists, so products accumulate in
    ascending term-identifier order. The native backend's postings-list loop
    produces the identical sequence, hence the identical bits.
    """
    if u.dim != v.dim:
        raise ValueError(f"dimension mismatch: {u.dim} vs {v.dim}")

    products: list[float] = []
    i = j = 0
    ui, uv, vi, vv = u.indices, u.values, v.indices, v.values
    nu, nv = len(ui), len(vi)
    while i < nu and j < nv:
        a, b = ui[i], vi[j]
        if a == b:
            products.append(uv[i] * vv[j])
            i += 1
            j += 1
        elif a < b:
            i += 1
        else:
            j += 1
    return reduce_sum(products, policy)


def l2_norm(v: SparseVector, policy: Reduction = Reduction.NAIVE) -> float:
    """Euclidean norm.

    ``sqrt`` of the sum of squares, in that order and with no rescaling. A
    hypot-style scaled formulation resists overflow better but produces different
    digits, and section 6 forbids stabilising transformations.
    """
    return sqrt(reduce_sum([x * x for x in v.values], policy))


def cosine_of(
    u: SparseVector,
    v: SparseVector,
    policy: Reduction = Reduction.NAIVE,
    *,
    u_norm: float | None = None,
    v_norm: float | None = None,
) -> float:
    """Cosine similarity, with the zero-vector convention of section 2.3.

    Precomputed norms may be supplied: same value, same policy, so nothing
    changes numerically, and scoring a whole corpus becomes ``O(nnz)`` instead of
    ``O(nnz * queries)``.

    The expression is pinned as ``dot / (nu * nv)``. ``(dot / nu) / nv`` and
    ``dot * (1 / (nu * nv))`` are algebraically equal and numerically different;
    the native backend matches this form.
    """
    nu = l2_norm(u, policy) if u_norm is None else u_norm
    nv = l2_norm(v, policy) if v_norm is None else v_norm
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return dot(u, v, policy) / (nu * nv)


@dataclass(frozen=True, slots=True)
class CsrMatrix:
    """Compressed sparse row matrix: one row per document.

    Column indices within each row are strictly ascending, mirroring the native
    backend's layout so the two can be compared directly.
    """

    indptr: tuple[int, ...]
    indices: tuple[int, ...]
    values: tuple[float, ...]
    n_rows: int
    n_cols: int

    @property
    def nnz(self) -> int:
        return len(self.values)

    def row(self, i: int) -> SparseVector:
        """Row ``i`` as a :class:`SparseVector`."""
        lo, hi = self.indptr[i], self.indptr[i + 1]
        return SparseVector(indices=self.indices[lo:hi], values=self.values[lo:hi], dim=self.n_cols)

    def rows(self) -> Iterator[SparseVector]:
        for i in range(self.n_rows):
            yield self.row(i)

    def row_norms(self, policy: Reduction = Reduction.NAIVE) -> tuple[float, ...]:
        """L2 norm of every row, computed once and reused for all queries."""
        return tuple(l2_norm(self.row(i), policy) for i in range(self.n_rows))

    def is_canonical(self) -> bool:
        """Whether the structure is well formed and every row is ascending."""
        if len(self.indptr) != self.n_rows + 1:
            return False
        if self.indptr[0] != 0 or self.indptr[-1] != len(self.values):
            return False
        if len(self.indices) != len(self.values):
            return False
        for i in range(self.n_rows):
            lo, hi = self.indptr[i], self.indptr[i + 1]
            if lo > hi:
                return False
            seg = self.indices[lo:hi]
            if any(a >= b for a, b in pairwise(seg)):
                return False
            if any(not 0 <= c < self.n_cols for c in seg):
                return False
        return True

    @classmethod
    def from_rows(cls, rows: Sequence[SparseVector], n_cols: int) -> CsrMatrix:
        """Assemble from per-row sparse vectors."""
        indptr = [0]
        indices: list[int] = []
        values: list[float] = []
        for r in rows:
            if r.dim != n_cols:
                raise ValueError(f"row dimension {r.dim} does not match n_cols={n_cols}")
            indices.extend(r.indices)
            values.extend(r.values)
            indptr.append(len(values))
        return cls(
            indptr=tuple(indptr),
            indices=tuple(indices),
            values=tuple(values),
            n_rows=len(rows),
            n_cols=n_cols,
        )
