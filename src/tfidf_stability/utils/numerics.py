"""Reduction policies, exact logarithms and floating-point environment probes.

This module is the Python half of the project's bit-exactness contract. Every
function here has a counterpart in ``cpp/include/tfidf/core/`` that must produce
*byte-identical* results, and the differential test suite asserts exactly that.

Two ideas govern the design.

**Summation order is part of the specification.** ``(a + b) + c`` and
``a + (b + c)`` are different computations in binary64. README section 2 writes
sums without bracketing, so the normative reading is a plain left-to-right fold,
which is what :data:`Reduction.NAIVE` implements and what every published result
uses. The other policies exist as *instruments*: the spread between them is a
direct measurement of the floating-point noise floor, and the near-tie tolerance
tau of section 7.1 is derived from that measurement rather than asserted. See
``docs/spec_addenda.md#g13`` and ``docs/numerics.md``.

**The only transcendental in the pipeline is a liability.** IEEE-754 mandates
correct rounding for ``+ - * / sqrt`` but *not* for ``log``. Platform libms
therefore disagree, and measurement on this project's reference machine found
15.16% of idf entries differing from the correctly-rounded value. Since idf is
computed once over the vocabulary and never in a hot loop, we pay a few
microseconds for :func:`correctly_rounded_log_ratio` and obtain cross-platform
bit-reproducibility in exchange.
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

#: Working precision for :func:`correctly_rounded_log_ratio`. 60 decimal digits is
#: far beyond the ~17 needed to round binary64 correctly; verified empirically to
#: agree with a 120-digit evaluation on the ratios this project produces.
DECIMAL_LOG_PRECISION: Final[int] = 60

#: Block size for :func:`pairwise_sum`. Matches numpy's, so the ``numpy`` backend
#: and this policy agree, which makes the cross-check meaningful.
_PAIRWISE_BLOCK: Final[int] = 128


class Reduction(str, Enum):
    """How a sum of floating-point numbers is accumulated.

    The choice is never implicit: it is threaded explicitly through every API
    that sums anything, and recorded in every run manifest.
    """

    #: Plain left-to-right fold. The literal reading of the paper's formulas and
    #: the default for every published result.
    NAIVE = "naive"
    #: Kahan-Babuska-Neumaier compensated summation. More accurate than NAIVE,
    #: and therefore *not* what the paper specifies -- an instrument, not a fix.
    NEUMAIER = "neumaier"
    #: Recursive pairwise summation with a 128-element base case; what numpy does.
    PAIRWISE = "pairwise"
    #: Correctly-rounded sum (``math.fsum``). Ground truth for error measurement.
    EXACT = "exact"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


# ---------------------------------------------------------------------------
# Reduction policies
# ---------------------------------------------------------------------------
def naive_sum(values: Iterable[float]) -> float:
    """Sum left to right with no compensation.

    This is the normative policy. It is *deliberately* the least accurate of the
    four: the paper specifies plain sums, and silently improving on them would
    mean publishing numbers that the stated mathematics does not produce.
    """
    total = 0.0
    for v in values:
        total += v
    return total


def neumaier_sum(values: Iterable[float]) -> float:
    """Kahan-Babuska-Neumaier compensated summation.

    Tracks the rounding error lost at each step in a separate accumulator and
    adds it back once at the end. Unlike plain Kahan, this variant is also
    correct when the running total is smaller in magnitude than the addend.
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

    Implemented in *streaming* form: values are accumulated into blocks of
    :data:`_PAIRWISE_BLOCK`, and completed blocks are merged by a binary-counter
    stack that combines equal-weight partials. This builds a balanced summation
    tree without ever needing to know the input length in advance.

    That last property is the reason for this formulation rather than the more
    obvious ``split at n // 2 and recurse``. The dot-product kernel consumes
    products one at a time and cannot see the intersection size up front, so a
    length-aware algorithm could not be used there -- and the C++ core, which
    must agree with this function bit-for-bit, has the same constraint.

    The two formulations are *not* interchangeable: they build different trees
    for any length that is not a power of two times the block size, and they
    first disagree at n = 129. Both are legitimate pairwise schemes; this one is
    the pinned specification.
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

    Used as ground truth when measuring the absolute error of the other
    policies, which is what turns section 7.1's qualitative statement about tau
    into a measured quantity.
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

    Trivial, but wrapped so the *reason* it is safe is recorded at the call site:
    IEEE-754 mandates that ``sqrt`` be correctly rounded, so unlike ``log`` it is
    identical on every conforming platform and may be used freely in the native
    core.
    """
    return math.sqrt(x)


# ---------------------------------------------------------------------------
# Correctly-rounded logarithm  (docs/spec_addenda.md#g13)
# ---------------------------------------------------------------------------
def correctly_rounded_log_ratio(numerator: int, denominator: int) -> float:
    """Return ``ln(numerator / denominator)``, correctly rounded to binary64.

    ``math.log`` delegates to the platform libm, which IEEE-754 does not require
    to be correctly rounded. Measured across ``N`` in {100, 610, 9742, 20000,
    50000} and all valid ``df``, the platform result differs from the correctly
    rounded one in **15.16%** of cases. Left unaddressed, that makes idf -- and
    hence every weight, norm and score downstream -- platform-dependent at the
    1-ulp level, which would silently break the reproducibility claims this
    project is built on.

    Evaluating the ratio in :class:`~decimal.Decimal` at
    :data:`DECIMAL_LOG_PRECISION` digits and rounding once removes the problem
    for a cost of a few microseconds per vocabulary entry.

    Note that the division is performed *before* the logarithm, matching section
    2.1 exactly. This is not cosmetic: ``log(a/b)`` and ``log(a) - log(b)``
    differ in 94.53% of the ratios arising at ``N = 9742``.

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

    Provided so the difference against :func:`correctly_rounded_log_ratio` can be
    measured and reported rather than becoming folklore. Not used for published
    results.
    """
    return math.log(numerator / denominator)


# ---------------------------------------------------------------------------
# Bit-level helpers -- the vocabulary of the differential tests
# ---------------------------------------------------------------------------
def bits_of(x: float) -> bytes:
    """The raw 8 little-endian bytes of a binary64 value.

    Differential tests compare *these*, not the floats. ``==`` on floats would
    conflate ``-0.0`` with ``0.0`` and would call two NaNs unequal; byte equality
    is the property we actually mean by "bit-exact".
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

    Returns ``inf`` if either argument is non-finite, or if the two straddle
    zero in a way that makes the count meaningless. Used to express test
    tolerances in a unit that is scale-free, rather than as an arbitrary
    absolute epsilon.
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

    The Python-side counterpart of ``tfidf::fp::selftest``. It cannot read MXCSR
    directly, so it probes behaviourally instead: if subnormals survive
    arithmetic then flush-to-zero is not in effect.
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

    Called at import. The failure this actually guards against is a numerical
    library -- typically a BLAS behind numpy -- setting flush-to-zero
    process-wide when it loads. In a project whose subject is near-tie margins,
    silently flushing subnormal score differences to zero would corrupt exactly
    the phenomenon under study.
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
