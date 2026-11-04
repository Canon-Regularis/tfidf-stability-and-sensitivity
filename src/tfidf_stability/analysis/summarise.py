"""Distribution summaries and the experiment-result envelope.

Percentiles are nearest-rank, so each is an observed value. :class:`ExperimentResult`
is what every runner writes; :meth:`~ExperimentResult.digest` ignores volatile fields.
Rationale: docs/engineering_notes.md#nearest-rank-percentiles
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from tfidf_stability.persistence.manifest import environment_block
from tfidf_stability.utils.hashing import hash_text
from tfidf_stability.utils.io import canonical_json, strip_volatile

__all__ = [
    "Distribution",
    "ExperimentResult",
    "percentile",
    "summarise_values",
]

#: Percentiles reported for every distribution.
DEFAULT_PERCENTILES: tuple[int, ...] = (0, 1, 5, 25, 50, 75, 95, 99, 100)


def _signed_order(value: float) -> tuple[float, float]:
    """Sort key placing ``-0.0`` below ``+0.0``.

    ``sorted`` is stable and ``-0.0 == 0.0``, so a value-only key lets arrival order decide
    the sign of ``minimum``, ``maximum`` and a zero percentile, breaking bit-exactness.
    Over finite values the key is IEEE-754 ``totalOrder``.
    """
    return (value, math.copysign(1.0, value))


def percentile(sorted_values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile of an already-sorted sample.

    ``sorted_values`` is non-decreasing and ``p`` is in ``[0, 100]``. The result
    is a sample element, never interpolated: ``p50`` of ``[1, 2, 3, 4]`` is 2.
    The caller sorts once, since re-sorting per percentile dominates the cost.
    """
    n = len(sorted_values)
    if n == 0:
        return math.nan
    if p <= 0.0:
        return sorted_values[0]
    if p >= 100.0:
        return sorted_values[-1]
    rank = math.ceil(p / 100.0 * n)
    return sorted_values[max(1, rank) - 1]


@dataclass(frozen=True, slots=True)
class Distribution:
    """A summarised sample of measurements."""

    name: str
    n: int
    n_nan: int
    #: Observations that were infinite. Counted apart from ``n_nan`` because
    #: ``canonical_json`` writes both as ``null``; the two counts are what
    #: separate them in the record.
    n_infinite: int
    n_zero: int
    minimum: float
    maximum: float
    mean: float
    percentiles: dict[str, float]

    @property
    def share_zero(self) -> float:
        """Share of observations equal to zero.

        For margins this is G3's headline statistic, the exact-tie rate. A
        percentile summary hides it: past 50% it only makes several percentiles
        read 0.
        """
        return self.n_zero / self.n if self.n else math.nan

    def as_dict(self) -> dict[str, Any]:
        """The record, under the names this class declares.

        ``minimum`` and ``maximum`` rather than ``min``/``max``, so a reader
        moving between this file and the JSON meets one set of names.
        """
        return {
            "name": self.name,
            "n": self.n,
            "n_nan": self.n_nan,
            "n_infinite": self.n_infinite,
            "n_zero": self.n_zero,
            "share_zero": self.share_zero,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "percentiles": self.percentiles,
            "percentile_method": "nearest-rank (no interpolation)",
        }


def summarise_values(
    name: str,
    values: Iterable[float],
    *,
    percentiles: Sequence[int] = DEFAULT_PERCENTILES,
) -> Distribution:
    """Summarise a sample, keeping non-finite values out of the statistics.

    NaN marks an undefined quantity (``m_min^top`` at ``k = 1`` (G16), a degenerate query's
    margin). ``boundary_margin`` gives a defined infinity for an infinite score, which would
    make ``mean`` and upper percentiles infinite. ``n_nan`` and ``n_infinite`` count each.
    """
    collected = list(values)
    finite = sorted((v for v in collected if math.isfinite(v)), key=_signed_order)
    n_nan = sum(1 for v in collected if math.isnan(v))
    n_infinite = len(collected) - len(finite) - n_nan

    if not finite:
        return Distribution(
            name=name,
            n=0,
            n_nan=n_nan,
            n_infinite=n_infinite,
            n_zero=0,
            minimum=math.nan,
            maximum=math.nan,
            mean=math.nan,
            percentiles={f"p{p}": math.nan for p in percentiles},
        )

    return Distribution(
        name=name,
        n=len(finite),
        n_nan=n_nan,
        n_infinite=n_infinite,
        n_zero=sum(1 for v in finite if v == 0.0),
        minimum=finite[0],
        maximum=finite[-1],
        mean=math.fsum(finite) / len(finite),
        percentiles={f"p{p}": percentile(finite, p) for p in percentiles},
    )


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """One experiment's output, with everything needed to reproduce it."""

    experiment: str
    #: What the experiment measured; JSON-serialisable. One concept carries one
    #: key across every report: an exactly-zero margin is always ``is_exact_tie``.
    #: A band *value* is not a property of a thing, so
    #: ``stratify.EXACT_TIE_BAND`` stays the string ``"exact_tie"``.
    payload: dict[str, Any]
    #: The dataset's provenance block, verbatim from :class:`LoadedDataset`.
    data_provenance: dict[str, Any] = field(default_factory=dict)
    #: Configuration actually in force, rather than the file it came from.
    parameters: dict[str, Any] = field(default_factory=dict)

    def digest(self) -> str:
        """Identity of the result, independent of when it was produced.

        Over payload and parameters with volatile fields stripped, so a rerun on
        the same data gives the same string and a reader can verify a published
        number by comparing one hex value.
        """
        return hash_text(
            canonical_json(
                strip_volatile({"payload": self.payload, "parameters": self.parameters}),
                indent=None,
            )
        )

    def as_dict(self) -> dict[str, Any]:
        """The full record, including the volatile environment block."""
        return {
            "experiment": self.experiment,
            "result_digest": self.digest(),
            "parameters": self.parameters,
            "data_provenance": self.data_provenance,
            "payload": self.payload,
            "environment": environment_block(),
        }
