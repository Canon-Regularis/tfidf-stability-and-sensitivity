"""Smoothed inverse document frequency (README section 2.1).

    idf(t) = log((1 + N) / (1 + df(t))) + 1

Two properties are used throughout and asserted in tests: idf decays
monotonically in ``df``, and the additive constant keeps it strictly positive,
with ``idf(t) >= 1`` for every ``t`` with ``df(t) <= N``, which makes the
corpus-level Lipschitz bound of ``spec_addenda.md#g4`` computable.

The logarithm is computed exactly. IEEE-754 mandates correct rounding for
``+ - * / sqrt`` but not for ``log``, and platform libms disagree: on this
project's reference machine ``math.log`` differs from the correctly-rounded value
in 15.16% of idf entries (44.5% of the raw logarithms before the ``+1``). Left
alone, idf and therefore every weight, norm and score would differ by ~1 ulp
across operating systems.

idf is ``O(|V|)`` values computed once and never in a hot loop, so it is
evaluated in :class:`~decimal.Decimal` and rounded once. That also removes the
only transcendental from the native pipeline: the C++ core receives idf as data,
so it computes with correctly-rounded operations alone. See
``docs/spec_addenda.md#g13``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from tfidf_stability.utils.numerics import correctly_rounded_log_ratio, platform_log_ratio

__all__ = ["IdfVector", "LogImpl", "delta_idf", "idf_linf", "smoothed_idf", "smoothed_idf_one"]


class LogImpl(str, Enum):
    """Which logarithm to use when computing idf."""

    #: Exact ratio in Decimal, rounded once. Platform-independent. The default,
    #: and the only setting valid for published results.
    CORRECTLY_ROUNDED = "correctly_rounded"
    #: The platform libm. Faster, and differs from the above in ~15% of entries.
    #: Kept so the difference can be measured.
    PLATFORM = "platform"

    def __str__(self) -> str:
        return self.value


def smoothed_idf_one(df: int, n_documents: int, impl: LogImpl = LogImpl.CORRECTLY_ROUNDED) -> float:
    """``idf`` for a single token.

    Divide before taking the logarithm, as section 2.1 writes it: ``log(a/b)``
    and ``log(a) - log(b)`` give different binary64 results in 94.5% of the
    ratios arising at ``N = 9742``.

    Args:
        df: Document frequency of the token. Must satisfy ``0 <= df <= N``.
        n_documents: Corpus size ``N``.
        impl: Which logarithm implementation to use.

    Returns:
        ``log((1 + N) / (1 + df)) + 1``.
    """
    if df < 0:
        raise ValueError(f"df must be non-negative, got {df}")
    if df > n_documents:
        raise ValueError(f"df={df} exceeds the corpus size N={n_documents}")

    if impl is LogImpl.PLATFORM:
        return platform_log_ratio(1 + n_documents, 1 + df) + 1.0
    return correctly_rounded_log_ratio(1 + n_documents, 1 + df) + 1.0


@dataclass(frozen=True, slots=True)
class IdfVector:
    """IDF values indexed by term identifier, with their provenance."""

    values: tuple[float, ...]
    n_documents: int
    log_impl: LogImpl

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, term_id: int) -> float:
        return self.values[term_id]

    @property
    def linf(self) -> float:
        """``||idf||_inf``, the largest IDF value.

        Appears in the perturbation bound of section 4.2. Equals
        ``log((1 + N) / 2) + 1`` when every member has ``df >= 1``.
        """
        return max(self.values) if self.values else 0.0

    @property
    def minimum(self) -> float:
        """The smallest IDF value; ``>= 1`` whenever every ``df <= N``."""
        return min(self.values) if self.values else 0.0


def smoothed_idf(
    df: Sequence[int],
    n_documents: int,
    impl: LogImpl = LogImpl.CORRECTLY_ROUNDED,
) -> IdfVector:
    """Compute the IDF vector for a whole vocabulary.

    Args:
        df: Document frequency per term identifier.
        n_documents: Corpus size ``N``.
        impl: Which logarithm implementation to use.

    Returns:
        The :class:`IdfVector`, in term-identifier order.
    """
    return IdfVector(
        values=tuple(smoothed_idf_one(d, n_documents, impl) for d in df),
        n_documents=n_documents,
        log_impl=impl,
    )


def idf_linf(idf: IdfVector | Sequence[float]) -> float:
    """``||idf||_inf``, accepting either an :class:`IdfVector` or a raw sequence."""
    if isinstance(idf, IdfVector):
        return idf.linf
    return max(idf) if idf else 0.0


def delta_idf(
    df_before: int,
    df_after: int,
    n_before: int,
    n_after: int,
    impl: LogImpl = LogImpl.CORRECTLY_ROUNDED,
) -> float:
    """The IDF change induced by a corpus perturbation (section 4.1).

        delta_idf(t) = log((1 + N') / (1 + df'(t))) - log((1 + N) / (1 + df(t)))

    Section 4.1 uses this to separate a change in corpus size from a change in
    the document-frequency distribution; low-frequency tokens stay sensitive
    under smoothing. Computed as a difference of two exact logarithms rather than
    one ``log`` of a ratio of ratios, matching the expression as written. The
    ``+1`` in ``idf`` cancels in the difference, so it does not appear here.
    """
    log = platform_log_ratio if impl is LogImpl.PLATFORM else correctly_rounded_log_ratio
    return log(1 + n_after, 1 + df_after) - log(1 + n_before, 1 + df_before)
