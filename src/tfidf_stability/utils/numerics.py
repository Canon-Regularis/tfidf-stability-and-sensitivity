"""Reduction policies, exact logarithms and floating-point environment probes.

Functions match ``cpp/include/tfidf/core/`` bit for bit. Published sums use :data:`Reduction.NAIVE`
(section 2); policy spread sets tau (section 7.1); idf uses :func:`correctly_rounded_log_ratio`.
Specified in ``docs/spec_addenda.md#g13``. Rationale: docs/engineering_notes.md#bit-exactness
"""

from __future__ import annotations

import math
import struct
import sys
from collections.abc import Iterable, Sequence
from decimal import Decimal, getcontext, localcontext
from enum import Enum
from typing import Final

__all__ = [
    "DECIMAL_LOG_PRECISION",
    "Reduction",
    "assert_sane_float_environment",
    "bits_of",
    "correctly_rounded_log_ratio",
    "exact_sum",
    "float_environment",
    "naive_sum",
    "neumaier_sum",
    "pairwise_sum",
    "platform_log_ratio",
    "reduce_sum",
    "same_bits",
    "sqrt",
    "ulp",
    "ulps_between",
]

#: Working precision for :func:`correctly_rounded_log_ratio`. 60 decimal digits
#: against the ~17 needed to round binary64 correctly; agrees with a 120-digit
#: evaluation on every ratio this project produces.
DECIMAL_LOG_PRECISION: Final[int] = 60

#: Block size for :func:`pairwise_sum`, equal to numpy's. numpy unrolls its base case into
#: eight accumulators, so numpy's sum and this one differ at 208 of 262 tested lengths, first
#: at n = 8. The contract is agreement with the C++ core, bit for bit from n = 1 to 10,000.
_PAIRWISE_BLOCK: Final[int] = 128


class Reduction(str, Enum):
    """How a sum of floating-point numbers is accumulated.

    Never implicit: threaded through every API that sums anything, and recorded
    in every run manifest.
    """

    #: Plain left-to-right fold. The literal reading of the paper's formulas and
    #: the default for every published result.
    NAIVE = "naive"
    #: Kahan-Babuska-Neumaier compensated summation. More accurate than NAIVE,
    #: so it is an instrument rather than the policy the paper specifies.
    NEUMAIER = "neumaier"
    #: Pairwise summation over 128-element blocks: numpy's block size, a
    #: different tree (see :func:`pairwise_sum`).
    PAIRWISE = "pairwise"
    #: Correctly-rounded sum (``math.fsum``). Ground truth for error measurement.
    EXACT = "exact"

    def __str__(self) -> str:
        return self.value


# ---------------------------------------------------------------------------
# Reduction policies
# ---------------------------------------------------------------------------
def naive_sum(values: Iterable[float]) -> float:
    """Sum left to right with no compensation.

    The normative policy and the least accurate of the four: the paper specifies
    plain sums, and improving on them would publish numbers the stated
    mathematics does not produce.
    """
    total = 0.0
    for v in values:
        total += v
    return total


def neumaier_sum(values: Iterable[float]) -> float:
    """Kahan-Babuska-Neumaier compensated summation.

    Tracks the rounding error lost at each step in a separate accumulator and
    adds it back once at the end. Unlike plain Kahan, this variant stays correct
    when the running total is smaller in magnitude than the addend.
    """
    total = 0.0
    compensation = 0.0
    for v in values:
        t = total + v
        if abs(total) >= abs(v):
            compensation += (total - t) + v
        else:
            compensation += (v - t) + total
        total = t
    return total + compensation


def pairwise_sum(values: Iterable[float]) -> float:
    """Pairwise summation; error grows as O(log n) rather than O(n).

    Values fill blocks of :data:`_PAIRWISE_BLOCK`; full blocks merge through a binary-counter
    stack, a balanced tree built in one streaming pass. The result first differs from
    split-at-n // 2 recursion at n = 129. Rationale: docs/engineering_notes.md#streaming-pairwise
    """
    partials: list[float] = []
    weights: list[int] = []
    block = 0.0
    in_block = 0

    for v in values:
        block += v
        in_block += 1
        if in_block == _PAIRWISE_BLOCK:
            # Merge with each completed partial of equal weight, doubling the weight each time.
            carry, weight = block, 1
            while partials and weights[-1] == weight:
                weights.pop()
                carry = partials.pop() + carry
                weight *= 2
            partials.append(carry)
            weights.append(weight)
            block = 0.0
            in_block = 0

    # Fold the completed levels deepest-first, then the trailing partial block.
    total = 0.0
    for p in reversed(partials):
        total += p
    return total + block


def exact_sum(values: Iterable[float]) -> float:
    """Correctly-rounded sum: the exact real sum, rounded once.

    Ground truth for the absolute error of the other policies, which is how tau
    in section 7.1 becomes a measured quantity.
    """
    return math.fsum(values)


_REDUCERS = {
    Reduction.NAIVE: naive_sum,
    Reduction.NEUMAIER: neumaier_sum,
    Reduction.PAIRWISE: pairwise_sum,
    Reduction.EXACT: exact_sum,
}


def reduce_sum(values: Sequence[float], policy: Reduction = Reduction.NAIVE) -> float:
    """Sum ``values`` under the given reduction policy."""
    try:
        reducer = _REDUCERS[policy]
    except KeyError:  # pragma: no cover - defensive
        raise ValueError(f"unknown reduction policy: {policy!r}") from None
    return reducer(values)


def sqrt(x: float) -> float:
    """Square root.

    IEEE-754 mandates correct rounding for ``sqrt``, so unlike ``log`` it agrees
    on every conforming platform and the native core may call it freely.
    """
    return math.sqrt(x)


# ---------------------------------------------------------------------------
# Correctly-rounded logarithm  (docs/spec_addenda.md#g13)
# ---------------------------------------------------------------------------
def correctly_rounded_log_ratio(numerator: int, denominator: int) -> float:
    """Return ``ln(numerator / denominator)``, correctly rounded to binary64.

    ``numerator`` (``1 + N``) and ``denominator`` (``1 + df(t)``, section 2.1) must be positive,
    else ``ValueError``. The ratio is evaluated at :data:`DECIMAL_LOG_PRECISION` digits in
    :class:`~decimal.Decimal` and rounded once. Rationale: docs/engineering_notes.md#exact-log
    """
    if denominator <= 0:
        raise ValueError(f"denominator must be positive, got {denominator}")
    if numerator <= 0:
        raise ValueError(f"numerator must be positive, got {numerator}")
    with localcontext() as ctx:
        ctx.prec = DECIMAL_LOG_PRECISION
        return float((Decimal(numerator) / Decimal(denominator)).ln())


def platform_log_ratio(numerator: int, denominator: int) -> float:
    """``ln(numerator / denominator)`` using the platform libm.

    Here so the gap against :func:`correctly_rounded_log_ratio` can be measured
    and reported. Never used for published results.
    """
    return math.log(numerator / denominator)


# ---------------------------------------------------------------------------
# Bit-level helpers: the vocabulary of the differential tests
# ---------------------------------------------------------------------------
def bits_of(x: float) -> bytes:
    """The raw 8 little-endian bytes of a binary64 value.

    Differential tests compare these bytes: float ``==`` conflates ``-0.0`` with
    ``0.0`` and calls two NaNs unequal, so byte equality is what "bit-exact"
    means here.
    """
    return struct.pack("<d", x)


def same_bits(a: float, b: float) -> bool:
    """True when two floats have identical bit patterns."""
    return struct.pack("<d", a) == struct.pack("<d", b)


def ulp(x: float) -> float:
    """The spacing between ``x`` and the next representable float away from zero."""
    return math.ulp(x)


def ulps_between(a: float, b: float) -> float:
    """Signed distance from ``a`` to ``b`` measured in units in the last place.

    ``inf`` if either argument is non-finite. Test tolerances are stated in this
    scale-free unit rather than as an absolute epsilon.
    """
    if not (math.isfinite(a) and math.isfinite(b)):
        return math.inf
    if a == b:
        return 0.0
    scale = ulp(max(abs(a), abs(b)))
    if scale == 0.0:  # pragma: no cover - both subnormal-zero
        return 0.0
    # Scaled before the subtraction rather than after it: `b - a` is `-inf` for
    # arguments of opposite sign near the maximum, and the contract above reserves
    # an infinite result for a non-finite argument. `scale` is a power of two, so
    # dividing first is exact and the two orders agree elsewhere.
    return b / scale - a / scale


# ---------------------------------------------------------------------------
# Floating-point environment
# ---------------------------------------------------------------------------
def _rounding_is_to_nearest() -> bool:
    """Whether the process is rounding to nearest, ties to even, right now.

    Three exact-tie probes on variables, which CPython cannot constant-fold, detect the
    upward, downward and toward-zero modes. A C extension such as a BLAS can change the
    mode on load. Rationale: docs/engineering_notes.md#live-rounding-probe
    """
    one = 1.0
    ulp = 2.0**-52
    half = 2.0**-53

    # One expression rather than three early returns: every probe runs, and a chain of
    # `if ... return False` adds two arcs that only a machine in a directed rounding mode
    # can take, so no test run covers them and the 100% coverage gate fails.
    no_upward = (one + half) == one
    no_downward = (-one - half) == -one
    no_toward_zero = (one + (ulp + half)) == (one + 2.0 * ulp)
    return no_upward and no_downward and no_toward_zero


def float_environment() -> dict[str, object]:
    """Describe the live floating-point environment, for the run manifest.

    Python-side counterpart of ``tfidf::fp::selftest``. MXCSR is unreadable from
    here, so the probes are behavioural: a subnormal surviving arithmetic means
    flush-to-zero is off.
    """
    tiny = sys.float_info.min
    subnormal_survives = (tiny / 2.0) > 0.0

    a, b = 0.1, 0.2
    c, d = 1.0, 1e-17

    return {
        "mantissa_dig": sys.float_info.mant_dig,
        "epsilon": sys.float_info.epsilon,
        "max": sys.float_info.max,
        "min_normal": sys.float_info.min,
        # What CPython was COMPILED with, kept for provenance. It is a frozen
        # constant and cannot answer the question the guard asks; see
        # "rounds_live" below.
        "rounds": sys.float_info.rounds,  # 1 == round-to-nearest
        "rounds_live": _rounding_is_to_nearest(),
        "subnormals_supported": subnormal_survives,
        "constant_folding_ok": (a + b) != 0.3,
        "no_reassociation": ((c + d) - c) == 0.0,
        "decimal_default_prec": getcontext().prec,
    }


def assert_sane_float_environment() -> None:
    """Raise if the interpreter's floating-point environment is untrustworthy.

    Guards against a numerical library, usually a BLAS behind numpy, setting
    flush-to-zero process-wide on load: subnormal score differences would then
    flush to zero, destroying the near-tie margins under study.
    """
    env = float_environment()
    problems = []
    # `rounds_live`, not `rounds`: `sys.float_info.rounds` is FLT_ROUNDS captured when the
    # interpreter was built, so it keeps its value whatever the process later does to the
    # rounding mode. The C++ half of this guard reads `std::fegetround()` at run time.
    if not env["rounds_live"]:
        problems.append(
            f"the live rounding mode is not round-to-nearest-even "
            f"(the interpreter was built with FLT_ROUNDS={env['rounds']})"
        )
    if not env["subnormals_supported"]:
        problems.append("subnormals are flushed to zero (a BLAS may have set MXCSR.FTZ)")
    if sys.float_info.mant_dig != 53:
        problems.append(f"mantissa has {sys.float_info.mant_dig} bits, expected 53")
    if problems:
        from tfidf_stability.utils.validation import NumericEnvironmentError

        raise NumericEnvironmentError(
            "the floating-point environment is not trustworthy: " + "; ".join(problems)
        )
