"""Distribution summaries, and the experiment-result envelope.

Two responsibilities that belong together because they share one constraint:
**everything published must be reproducible from what is written down.**

Percentiles without a library
-----------------------------
The normative backend is standard-library only, so percentiles are computed
here rather than taken from NumPy. The method is nearest-rank on the sorted
sample -- ``p50`` of an even-length sample is the *lower* of the two central
values, not their mean.

That is a deliberate choice, not a simplification. Interpolating invents a value
that no query produced, and every quantity summarised here is a measured
floating-point observation whose exact bit pattern is meaningful elsewhere in the
study (margins, flip radii, score gaps). A reported ``p50`` that is not any
observed margin cannot be looked up, cannot be reproduced by inspection, and
would not be ``same_bits``-comparable against the raw data. NumPy's default
linear interpolation would do exactly that. The convention is recorded in the
output so a reader is never left guessing which one was used.

The result envelope
-------------------
:class:`ExperimentResult` is what every runner script writes. It carries the
payload, the provenance of the data it was computed from, and the environment --
and its :meth:`~ExperimentResult.digest` is taken over the payload **after
volatile fields are stripped**, so two runs of the same experiment on the same
data produce the same digest even though their timestamps differ.

That is the property that makes a published number checkable: a reader can rerun
the experiment and compare one hex string, rather than eyeballing a table.
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


def percentile(sorted_values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile of an already-sorted sample.

    Returns an element of the sample, never an interpolated value. See the module
    docstring for why that matters here.

    Args:
        sorted_values: Non-decreasing sample. Not sorted internally, because the
            callers already hold a sorted array and re-sorting per percentile
            would be the dominant cost.
        p: Percentile in ``[0, 100]``.

    Example:
        >>> percentile([1.0, 2.0, 3.0, 4.0], 50)
        2.0
        >>> percentile([1.0, 2.0, 3.0, 4.0], 100)
        4.0
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
    n_zero: int
    minimum: float
    maximum: float
    mean: float
    percentiles: dict[str, float]

    @property
    def share_zero(self) -> float:
        """Share of observations that are exactly zero.

        Broken out because for margins this is G3's headline statistic -- the
        exact-tie rate -- and it is invisible in any percentile summary once it
        exceeds 50%, where it would simply make several percentiles read 0.
        """
        return self.n_zero / self.n if self.n else math.nan

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n": self.n,
            "n_nan": self.n_nan,
            "n_zero": self.n_zero,
            "share_zero": self.share_zero,
            "min": self.minimum,
            "max": self.maximum,
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
    """Summarise a sample, keeping NaN out of the statistics but not the record.

    NaN marks an *undefined* quantity here -- ``m_min^top`` at ``k = 1`` (G16),
    or a margin on a degenerate query -- and is never a measurement. It is
    excluded from the statistics and counted, so a summary computed over mostly
    undefined values is visibly that rather than silently thin.
    """
    collected = list(values)
    finite = sorted(v for v in collected if not math.isnan(v))
    n_nan = len(collected) - len(finite)

    if not finite:
        return Distribution(
            name=name,
            n=0,
            n_nan=n_nan,
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
    #: What the experiment measured. Must be JSON-serialisable.
    payload: dict[str, Any]
    #: The dataset's provenance block, verbatim from :class:`LoadedDataset`.
    data_provenance: dict[str, Any] = field(default_factory=dict)
    #: Configuration actually in force, not the file it came from.
    parameters: dict[str, Any] = field(default_factory=dict)

    def digest(self) -> str:
        """Identity of the *result*, independent of when it was produced.

        Over the payload and parameters with volatile fields stripped -- so a
        rerun on the same data gives the same string, and a reader can verify a
        published number by comparing one hex value.
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
