"""Smoothed inverse document frequency (README section 2.1).

``idf(t) = log((1 + N) / (1 + df(t))) + 1`` decreases in ``df``, and ``idf >= 1`` for ``df <= N``
makes the ``spec_addenda.md#g4`` bound computable. Its log uses :class:`~decimal.Decimal`, rounded
once (``docs/spec_addenda.md#g13``). Rationale: docs/engineering_notes.md#idf-correctly-rounded-log
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
    """Return ``log((1 + N) / (1 + df)) + 1`` for one token, ``0 <= df <= N``.

    ``n_documents`` is ``N`` and ``impl`` selects the logarithm. The ratio is divided first,
    as section 2.1 writes it: ``log(a/b)`` and ``log(a) - log(b)`` give different binary64
    results in 94.5% of the ratios arising at ``N = 9742``.
    """
    if df < 0:
        raise ValueError(f"df must be non-negative, got {df}")
    if df > n_documents:
        raise ValueError(f"df={df} exceeds the corpus size N={n_documents}")

    # LogImpl is a str enum, so "platform" equals the member without being it.
    # An uncoerced comparison with `is` takes the correctly-rounded path and
    # reports the G13 gap as exactly zero, so coerce first. LogImpl() also
    # refuses an unrecognised value.
    if LogImpl(impl) is LogImpl.PLATFORM:
        return platform_log_ratio(1 + n_documents, 1 + df) + 1.0
    return correctly_rounded_log_ratio(1 + n_documents, 1 + df) + 1.0


@dataclass(frozen=True, slots=True)
class IdfVector:
    """IDF values indexed by term identifier, with their provenance."""

    values: tuple[float, ...]
    n_documents: int
    log_impl: LogImpl

    def __post_init__(self) -> None:
        """Normalise ``log_impl`` to the member, refusing an unknown value.

        ``LogImpl`` is a ``str`` enum, so ``"platform"`` equals the member without being it.
        ``persistence/save_load.py`` selects the container flag on ``is``, so an uncoerced
        ``"correctly_rounded"`` records ``PLATFORM`` and alters the round-tripped digest.
        """
        object.__setattr__(self, "log_impl", LogImpl(self.log_impl))

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

    ``df`` is the document frequency per term identifier, ``n_documents`` is ``N`` and
    ``impl`` selects the logarithm. Returns the :class:`IdfVector`, in term-identifier order.
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
    """Return the IDF change induced by a corpus perturbation (section 4.1).

    ``delta_idf(t) = log((1 + N') / (1 + df'(t))) - log((1 + N) / (1 + df(t)))``; the ``+1``
    cancels. Section 4.1 uses it to separate ``N`` from the df distribution; low-frequency tokens
    stay sensitive under smoothing. Two logs are subtracted, not one log of a ratio of ratios.
    """
    log = (
        platform_log_ratio
        if LogImpl(impl) is LogImpl.PLATFORM  # coerced, as in smoothed_idf_one
        else correctly_rounded_log_ratio
    )
    return log(1 + n_after, 1 + df_after) - log(1 + n_before, 1 + df_before)
