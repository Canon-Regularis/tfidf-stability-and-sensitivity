"""Reduction policies, exact logarithms and floating-point environment probes.

The Python half of the bit-exactness contract: every function here has a
counterpart in ``cpp/include/tfidf/core/`` that must produce byte-identical
results, and the differential suite asserts that.

Summation order is part of the specification: ``(a + b) + c`` and ``a + (b + c)``
are different computations in binary64, and README section 2 writes sums without
bracketing, so the normative reading is a left-to-right fold
(:data:`Reduction.NAIVE`, used for every published result). The other policies
are instruments; the spread between them measures the floating-point noise floor
the near-tie tolerance tau of section 7.1 is derived from. See
``docs/spec_addenda.md#g13`` and ``docs/numerics.md``.

IEEE-754 mandates correct rounding for ``+ - * / sqrt`` but not for ``log``, so
platform libms disagree: 15.16% of idf entries on this project's reference
machine differ from the correctly-rounded value. idf is computed once over the
vocabulary and never in a hot loop, so :func:`correctly_rounded_log_ratio` buys
cross-platform bit-reproducibility for a few microseconds.
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
    "bits_of",
    "correctly_rounded_log_ratio",
    "exact_sum",
    "float_environment",
    "naive_sum",
    "neumaier_sum",
    "pairwise_sum",
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

#: Block size for :func:`pairwise_sum`, matching numpy's. Equal block sizes do
#: not make the two agree: numpy unrolls its base case into eight accumulators,
#: so its order departs from a straight fold well below the boundary, and the
#: results differ at 208 of 262 sizes tried, first at n = 8. The contract is
#: agreement with the C++ core, which holds bit for bit from n = 1 to 10,000.
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

    Streaming form: values accumulate into blocks of :data:`_PAIRWISE_BLOCK`, and
    completed blocks merge through a binary-counter stack combining equal-weight
    partials, giving a balanced tree without knowing the length in advance. The
    dot-product kernel consumes products one at a time and cannot see the
    intersection size up front, and the C++ core that must match this bit for bit
    has the same constraint, so ``split at n // 2 and recurse`` is unavailable.

    The two formulations build different trees for any length that is not a power
    of two times the block size, first disagreeing at n = 129. Both are
    legitimate pairwise schemes; this one is the pinned specification.
    """
    partials: list[float] = []
    weights: list[int] = []
    block = 0.0
    in_block = 0

    for v in values:
        block += v
        in_block += 1
        if in_block == _PAIRWISE_BLOCK:
            # Merge with any completed partial of equal weight, doubling as we go.
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

    ``math.log`` delegates to the platform libm, which IEEE-754 does not require
    to be correctly rounded. Over ``N`` in {100, 610, 9742, 20000, 50000} and all
    valid ``df``, the platform result differs from the correctly rounded one in
    15.16% of cases, leaving idf and every weight, norm and score below it
    platform-dependent at the 1-ulp level. Evaluating the ratio in
    :class:`~decimal.Decimal` at :data:`DECIMAL_LOG_PRECISION` digits and
    rounding once costs a few microseconds per vocabulary entry.

    The division happens before the logarithm, matching section 2.1:
    ``log(a/b)`` and ``log(a) - log(b)`` differ in 94.53% of the ratios arising
    at ``N = 9742``.

    Args:
        numerator: Exact integer numerator (``1 + N`` in section 2.1).
        denominator: Exact integer denominator (``1 + df(t)``). Must be positive.

    Returns:
        The correctly-rounded natural logarithm of the exact rational ratio.
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
    return (b - a) / scale


# ---------------------------------------------------------------------------
# Floating-point environment
# ---------------------------------------------------------------------------
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
        "rounds": sys.float_info.rounds,  # 1 == round-to-nearest
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
    if env["rounds"] != 1:
        problems.append(f"rounding mode is {env['rounds']}, expected 1 (to-nearest)")
    if not env["subnormals_supported"]:
        problems.append("subnormals are flushed to zero (a BLAS may have set MXCSR.FTZ)")
    if sys.float_info.mant_dig != 53:
        problems.append(f"mantissa has {sys.float_info.mant_dig} bits, expected 53")
    if problems:
        from tfidf_stability.utils.validation import NumericEnvironmentError

        raise NumericEnvironmentError(
            "the floating-point environment is not trustworthy: " + "; ".join(problems)
        )
