"""Typed exceptions, warnings and validators.

Every condition ``docs/spec_addenda.md#g3`` gives defined behaviour for has a
named exception or warning here. Two modes run throughout:

``strict``
    Raise. The default for interactive and CLI use, where an unexpected corpus
    shape is more likely a mistake than an intention.

``lenient``
    Emit a diagnostic and return ``NaN``. The default inside experiment sweeps,
    where degenerate points (``tau`` larger than the score range, say) are
    legitimate members of the grid and must not abort the run.

The active mode goes into every run manifest: it changes which queries
contribute to a reported distribution.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import Enum
from typing import Final, NoReturn

__all__ = [
    "AbiVersionMismatchError",
    "ChainInflationWarning",
    "ConfigError",
    "DataIntegrityError",
    "DuplicateIdentifierError",
    "EmptyCorpusError",
    "EmptyVocabularyError",
    "KOutOfRangeError",
    "NativeBackendUnavailableError",
    "NumericEnvironmentError",
    "StrictMode",
    "TauExceedsScoreRangeWarning",
    "TfidfStabilityError",
    "check_finite",
    "check_non_negative",
    "check_unique_ids",
    "resolve_k",
]


class StrictMode(str, Enum):
    """Whether degenerate inputs raise or are flagged and returned as ``NaN``."""

    STRICT = "strict"
    LENIENT = "lenient"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class TfidfStabilityError(Exception):
    """Base class for every error this package raises on purpose."""


class EmptyVocabularyError(TfidfStabilityError):
    """The vocabulary is empty after filtering.

    A configuration error rather than a property of the data: it almost always
    means ``min_df`` is too high for the corpus size.
    """


class DuplicateIdentifierError(TfidfStabilityError):
    """Two documents share an identifier.

    Fatal. The ranking operator of section 2.3.1 is a strict total order only
    because the final tie-break key is unique; with duplicate ids the sorted
    order stops being determined and falls to the sorting algorithm.
    """


class KOutOfRangeError(TfidfStabilityError):
    """``k`` exceeds the number of rankable documents."""


class EmptyCorpusError(TfidfStabilityError):
    """Ranking was attempted over zero documents.

    ``docs/spec_addenda.md#g3`` calls this "an error on ranking" but names no
    exception class; this is that class (proposed as addendum G17).
    """


class ConfigError(TfidfStabilityError):
    """A configuration key is unrecognised, or its value is not admissible.

    Unrecognised keys are fatal. Every value in ``configs/default.yaml`` can move
    a published number and the whole file is hashed into the run manifest, so a
    key recorded but never read makes the manifest claim something the run did
    not do: a typo in ``n_max`` would silently take the default.
    """


class NumericEnvironmentError(TfidfStabilityError):
    """The floating-point environment cannot be trusted to produce valid results."""


class NativeBackendUnavailableError(TfidfStabilityError):
    """The compiled native backend was requested but could not be loaded."""


class AbiVersionMismatchError(TfidfStabilityError):
    """The compiled extension was built against a different Python-side contract."""


class DataIntegrityError(TfidfStabilityError):
    """An input dataset is absent, corrupt, or not the pinned version.

    Raised rather than warned: GroupLens updates ``ml-latest-small`` in place at
    a stable URL, so a dataset changing underneath a published result is
    plausible and invisible unless it aborts.
    """


# ---------------------------------------------------------------------------
# Warnings: diagnostics that tag a result rather than abort it
# ---------------------------------------------------------------------------
class TauExceedsScoreRangeWarning(UserWarning):
    """``tau`` is at least as large as the whole score range.

    Every tie ball then covers the whole corpus. A legitimate point at the top of
    a tau sweep, hence a warning, but the results there are degenerate and plots
    should mark them.
    """


class ChainInflationWarning(UserWarning):
    """Transitive chaining is inflating tie groups (see ``spec_addenda.md#g1``).

    Raised when the largest single-linkage chain far exceeds the largest clique:
    the reported groups are then held together by a sequence of small steps
    rather than by mutual indistinguishability.
    """


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------
_FINITE_HINT: Final = (
    "NaN or infinity in a sort key makes the comparator non-transitive, which is "
    "undefined behaviour for the underlying sort -- not merely a wrong answer."
)


def check_finite(values: Sequence[float], what: str) -> None:
    """Raise if any value is NaN or infinite."""
    for i, v in enumerate(values):
        if not math.isfinite(v):
            raise TfidfStabilityError(f"{what}[{i}] is {v!r}, which is not finite. {_FINITE_HINT}")


def check_non_negative(values: Sequence[float], what: str) -> None:
    """Raise if any value is negative.

    TF-IDF vectors live in the non-negative orthant (section 2.2), which is the
    only reason cosine similarity lies in [0, 1]. A negative coordinate voids
    that guarantee and means something upstream is wrong.
    """
    for i, v in enumerate(values):
        if v < 0.0:
            raise TfidfStabilityError(
                f"{what}[{i}] = {v!r} is negative; TF-IDF vectors must be non-negative "
                f"(README section 2.2), and cos in [0, 1] depends on it."
            )


def check_unique_ids(ids: Sequence[object]) -> None:
    """Raise if identifiers are not unique. See :class:`DuplicateIdentifierError`."""
    seen: dict[object, int] = {}
    for i, ident in enumerate(ids):
        if ident in seen:
            raise DuplicateIdentifierError(
                f"identifier {ident!r} appears at positions {seen[ident]} and {i}. "
                f"The ranking operator requires unique identifiers to be a total order."
            )
        seen[ident] = i


def resolve_k(k: int, n: int, mode: StrictMode = StrictMode.STRICT) -> int:
    """Validate ``k`` against a corpus of ``n`` rankable documents.

    Returns the effective ``k``. Strict mode raises on an over-large ``k``;
    lenient mode clamps to ``n``, and the caller is then expected to record
    ``k_effective``. See ``docs/spec_addenda.md#g3``.
    """
    if k <= 0:
        raise KOutOfRangeError(f"k must be positive, got {k}")
    if k <= n:
        return k
    if mode is StrictMode.STRICT:
        raise KOutOfRangeError(
            f"k={k} exceeds the {n} rankable documents. Use StrictMode.LENIENT to clamp."
        )
    return n


def _unreachable(msg: str) -> NoReturn:  # pragma: no cover - defensive
    raise AssertionError(f"unreachable: {msg}")
