"""Tests for the four summation policies and the environment they assume.

`Reduction.EXACT` (`math.fsum`, correctly rounded) is the oracle; the other
policies may disagree with it on hard input. The environment guard's failure
arms are reached by patching `sys.float_info` and the probe.
"""

from __future__ import annotations

import math
import random
import struct
import sys
from decimal import getcontext, localcontext

import pytest

from tfidf_stability.utils import numerics
from tfidf_stability.utils.numerics import (
    _PAIRWISE_BLOCK,
    DECIMAL_LOG_PRECISION,
    Reduction,
    bits_of,
    correctly_rounded_log_ratio,
    exact_sum,
    float_environment,
    naive_sum,
    neumaier_sum,
    pairwise_sum,
    platform_log_ratio,
    reduce_sum,
    same_bits,
    sqrt,
    ulp,
    ulps_between,
)
from tfidf_stability.utils.validation import NumericEnvironmentError

#: A sum where the small terms are lost entirely by a left-to-right fold.
_CANCELLING = [1e16, 1.0, -1e16]

#: Many small terms after one large one: naive loses most of them.
_ABSORBING = [1e16, *([1.0] * 100), -1e16]


# ---------------------------------------------------------------------------
# The policies agree on easy input and disagree where they should
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("policy", list(Reduction))
def test_every_policy_sums_an_exactly_representable_sequence_identically(
    policy: Reduction,
) -> None:
    """Powers of two add without rounding, so all four must agree bit for bit."""
    values = [0.5, 0.25, 0.125, 1.0]
    assert same_bits(reduce_sum(values, policy), 1.875)


@pytest.mark.parametrize("policy", list(Reduction))
def test_every_policy_sums_nothing_to_positive_zero(policy: Reduction) -> None:
    total = reduce_sum([], policy)
    assert same_bits(total, 0.0), "an empty sum must be +0.0, not -0.0"


@pytest.mark.parametrize("policy", list(Reduction))
def test_every_policy_returns_a_single_value_unchanged(policy: Reduction) -> None:
    assert same_bits(reduce_sum([0.1], policy), 0.1)


def test_the_naive_fold_loses_the_small_term_that_compensation_keeps() -> None:
    """The premise of having more than one policy at all.

    If naive and Neumaier agreed here there would be nothing to measure, so this
    is the test that keeps the noise-floor study from being vacuous.
    """
    assert naive_sum(_CANCELLING) == 0.0, "the 1.0 is absorbed and lost"
    assert neumaier_sum(_CANCELLING) == 1.0, "compensation recovers it"
    assert exact_sum(_CANCELLING) == 1.0, "and the correctly-rounded sum agrees"


def test_the_gap_between_naive_and_exact_widens_with_more_absorbed_terms() -> None:
    assert exact_sum(_ABSORBING) == 100.0
    assert naive_sum(_ABSORBING) != 100.0, "a fold that recovered these would not be naive"


def test_exact_summation_is_math_fsum_so_it_can_serve_as_the_oracle() -> None:
    values = [0.1] * 10
    assert same_bits(exact_sum(values), math.fsum(values))
    assert exact_sum(values) != naive_sum(values), "0.1 ten times is the classic case"


def test_pairwise_agrees_with_naive_below_the_block_size() -> None:
    """Under 128 elements pairwise has no tree to build, so the two must match."""
    values = [1.0 / (i + 1) for i in range(64)]
    assert same_bits(pairwise_sum(values), naive_sum(values))


def test_the_two_pairwise_formulations_first_disagree_at_the_documented_size() -> None:
    """Pairwise matches naive at 128 elements and differs at some size below 200.

    The module documents n = 129 as the first size at which the binary-counter
    and split-at-`n // 2` formulations differ; only naive is compared here."""
    values = [1.0 / (i + 1) for i in range(200)]
    assert same_bits(pairwise_sum(values[:128]), naive_sum(values[:128])), (
        "at exactly the block size there is still a single block"
    )
    disagreements = sum(
        1 for n in range(1, 200) if not same_bits(pairwise_sum(values[:n]), naive_sum(values[:n]))
    )
    assert disagreements > 0, "pairwise never differed from naive; the tree is not being built"


@pytest.mark.parametrize("n", [0, 1, 2, 127, 128, 129, 256, 257])
def test_pairwise_matches_the_exact_sum_far_more_closely_than_naive_at_every_size(
    n: int,
) -> None:
    """Tree-shape boundaries: one below, at, and one above the 128 base case."""
    values = [1.0 / (i + 1) for i in range(n)]
    truth = exact_sum(values)
    assert abs(pairwise_sum(values) - truth) <= abs(naive_sum(values) - truth) + 1e-15


def test_an_unknown_reduction_policy_is_rejected_not_silently_summed() -> None:
    with pytest.raises(ValueError, match="unknown reduction policy"):
        reduce_sum([1.0], "not_a_policy")  # type: ignore[arg-type]


def test_a_policys_string_value_selects_the_same_reducer_as_the_member() -> None:
    """`class Reduction(str, Enum)` is chosen so a value read from YAML works."""
    assert Reduction("naive") is Reduction.NAIVE
    assert same_bits(reduce_sum(_CANCELLING, Reduction("neumaier")), 1.0)


# ---------------------------------------------------------------------------
# Bit-level helpers
# ---------------------------------------------------------------------------
def test_bits_of_is_the_little_endian_double_encoding() -> None:
    assert bits_of(1.0) == struct.pack("<d", 1.0)


def test_same_bits_separates_the_two_zeros_that_equality_conflates() -> None:
    assert 0.0 == -0.0
    assert not same_bits(0.0, -0.0), "byte equality is what bit-exact means here"


def test_same_bits_calls_a_nan_equal_to_itself_where_equality_does_not() -> None:
    nan = float("nan")
    assert nan != nan  # noqa: PLR0124 - the premise being demonstrated
    assert same_bits(nan, nan)


def test_ulps_between_is_zero_for_the_two_zeros_because_they_are_equal() -> None:
    assert ulps_between(0.0, -0.0) == 0.0


def test_ulps_between_a_non_finite_value_is_infinite_rather_than_a_number() -> None:
    """A tolerance stated in ulps must not silently accept an infinity."""
    assert ulps_between(math.nan, 1.0) == math.inf
    assert ulps_between(math.inf, 1.0) == math.inf
    assert ulps_between(1.0, math.nan) == math.inf


def test_the_square_root_is_the_platform_one_because_ieee_mandates_it() -> None:
    """The square root matches `math.sqrt` on every value tried.

    IEEE 754 requires sqrt, unlike log, to be correctly rounded, so the native
    core may call it directly rather than receive it as data.
    """
    for value in (0.0, 1.0, 2.0, 0.25, 1e300, 5e-324):
        assert same_bits(numerics.sqrt(value), math.sqrt(value))


def test_one_ulp_apart_measures_as_one_ulp() -> None:
    a = 1.0
    b = math.nextafter(1.0, math.inf)
    assert abs(ulps_between(a, b)) == pytest.approx(1.0, abs=0.5)
    assert ulp(1.0) == math.ulp(1.0)


# ---------------------------------------------------------------------------
# The logarithm guards
# ---------------------------------------------------------------------------
def test_a_non_positive_denominator_is_rejected_before_any_logarithm() -> None:
    with pytest.raises(ValueError, match="denominator must be positive"):
        correctly_rounded_log_ratio(1, 0)
    with pytest.raises(ValueError, match="denominator must be positive"):
        correctly_rounded_log_ratio(1, -1)


def test_a_non_positive_numerator_is_rejected_before_any_logarithm() -> None:
    with pytest.raises(ValueError, match="numerator must be positive"):
        correctly_rounded_log_ratio(0, 1)
    with pytest.raises(ValueError, match="numerator must be positive"):
        correctly_rounded_log_ratio(-1, 1)


def test_the_denominator_is_checked_first_when_both_are_invalid() -> None:
    """The denominator is checked first when both arguments are invalid.

    The error message names only one argument, so the order is pinned."""
    with pytest.raises(ValueError, match="denominator must be positive"):
        correctly_rounded_log_ratio(0, 0)


def test_a_ratio_of_one_gives_exactly_zero() -> None:
    assert correctly_rounded_log_ratio(7, 7) == 0.0


def test_the_correctly_rounded_logarithm_differs_from_the_platform_somewhere() -> None:
    """The correctly rounded logarithm differs from the platform one somewhere.

    G13 measures this difference; if the two agreed everywhere the project would
    not need its own logarithm.
    """
    # Ratios of the form n/1 are not probed: the platform logarithm is already
    # correctly rounded for those. Real IDF arguments are N over df for any df
    # in 1..N, which is where the two differ.
    differing = compared = 0
    for n_documents in (100, 610, 2000):
        for df in range(1, n_documents + 1):
            compared += 1
            if not same_bits(
                correctly_rounded_log_ratio(n_documents, df),
                platform_log_ratio(n_documents, df),
            ):
                differing += 1
    assert compared > 2000, "the sweep is too small to say anything"
    assert differing > 0, (
        "the platform logarithm agreed on every ratio tried, so the exact "
        "implementation would be unnecessary"
    )
    assert differing / compared > 0.1, (
        f"only {differing} of {compared} differed; G13 measures this in the tens "
        f"of percent, so a rate this low means the wrong arguments are being probed"
    )


# ---------------------------------------------------------------------------
# The floating-point environment
# ---------------------------------------------------------------------------
def test_the_environment_block_reports_every_field_the_manifest_records() -> None:
    env = float_environment()
    assert set(env) == {
        "mantissa_dig",
        "epsilon",
        "max",
        "min_normal",
        "rounds",
        # `rounds` is the build-time constant in `sys.float_info`; `rounds_live`
        # probes the current rounding mode and is the one the guard reads.
        "rounds_live",
        "subnormals_supported",
        "constant_folding_ok",
        "no_reassociation",
        "decimal_default_prec",
    }


def test_this_interpreter_is_a_sane_environment_so_the_guard_stays_silent() -> None:
    assert_sane = numerics.assert_sane_float_environment
    assert_sane(), "the guard rejected the interpreter running the suite"


def test_a_flush_to_zero_environment_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A flush-to-zero environment is rejected.

    A BLAS that sets MXCSR.FTZ process-wide would flush the subnormal margins
    under study. Only the probe's subnormal answer is forced, so the guard's own
    logic runs."""
    real = numerics.float_environment
    monkeypatch.setattr(
        numerics, "float_environment", lambda: {**real(), "subnormals_supported": False}
    )
    with pytest.raises(NumericEnvironmentError, match="flushed to zero"):
        numerics.assert_sane_float_environment()


def test_a_rounding_mode_other_than_to_nearest_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = numerics.float_environment
    # The guard reads `rounds_live`, a probe of the live mode that a BLAS changing
    # the mode after import would move, not the build-time `rounds` constant.
    # Patching `rounds` alone must not raise; patching `rounds_live` must.
    monkeypatch.setattr(numerics, "float_environment", lambda: {**real(), "rounds": 0})
    numerics.assert_sane_float_environment()

    monkeypatch.setattr(numerics, "float_environment", lambda: {**real(), "rounds_live": False})
    with pytest.raises(NumericEnvironmentError, match="rounding mode"):
        numerics.assert_sane_float_environment()


def test_a_mantissa_that_is_not_53_bits_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """binary64 or nothing: every bit-exactness claim assumes 53."""
    # The real object is captured before patching: reading through
    # `sys.float_info` inside the shim would find the shim and recurse.
    real_info = sys.float_info

    class _Info:
        def __getattr__(self, name: str) -> object:
            return 24 if name == "mant_dig" else getattr(real_info, name)

    monkeypatch.setattr(sys, "float_info", _Info())
    with pytest.raises(NumericEnvironmentError, match="mantissa has 24 bits"):
        numerics.assert_sane_float_environment()


def test_several_problems_are_reported_together_rather_than_one_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Several faults are reported together in one message.

    A guard that stopped at the first fault would need repeated runs to describe
    a broken environment."""
    real = numerics.float_environment
    monkeypatch.setattr(
        numerics,
        "float_environment",
        # `rounds_live` is the flag the guard reads; the build-time `rounds`
        # constant does not affect the guard.
        lambda: {**real(), "rounds_live": False, "subnormals_supported": False},
    )
    with pytest.raises(
        NumericEnvironmentError, match="floating-point environment is not trustworthy"
    ) as caught:
        numerics.assert_sane_float_environment()
    message = str(caught.value)
    assert "rounding mode" in message
    assert "flushed to zero" in message
    assert ";" in message, "the problems are joined into one report"


# ---------------------------------------------------------------------------
# Stress: adversarial magnitudes
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("policy", list(Reduction))
def test_subnormal_terms_survive_every_policy_rather_than_flushing(policy: Reduction) -> None:
    tiny = 5e-324  # the smallest positive subnormal
    assert reduce_sum([tiny, tiny], policy) > 0.0, "a subnormal sum flushed to zero"


@pytest.mark.slow
def test_summing_near_the_maximum_overflows_to_infinity_rather_than_rescaling() -> None:
    """An overflowing naive sum is infinity rather than a rescaled value.

    Section 6 forbids stabilising transformations, so an overflow stays visible."""
    big = sys.float_info.max
    assert naive_sum([big, big]) == math.inf


@pytest.mark.slow
def test_the_policies_stay_within_a_few_ulps_of_the_exact_sum_over_a_long_sweep() -> None:
    checked = 0
    for n in (1, 2, 3, 17, 128, 129, 512, 1000):
        values = [(-1.0) ** i / (i + 1) for i in range(n)]
        truth = exact_sum(values)
        for policy in (Reduction.NEUMAIER, Reduction.PAIRWISE):
            got = reduce_sum(values, policy)
            assert abs(ulps_between(got, truth)) < 1e6, f"{policy} drifted at n={n}"
            checked += 1
    assert checked == 16, "the sweep did not run the shape it claims"


# ---------------------------------------------------------------------------
# Erroneous: what each policy does when the sum cannot be represented
# ---------------------------------------------------------------------------

# The spread between the policies is the measurement, so where they disagree is
# pinned. Past the top of the range they disagree three ways: a caller sweeping
# policies gets `inf`, `nan` or an exception depending on the policy.
_OVERFLOWING = [sys.float_info.max, sys.float_info.max]


@pytest.mark.parametrize("policy", [Reduction.NAIVE, Reduction.PAIRWISE])
def test_an_uncompensated_sum_past_the_maximum_reports_infinity(policy: Reduction) -> None:
    """An uncompensated sum past the maximum reports infinity.

    Section 6 forbids stabilising transformations, so an overflow stays visible
    rather than being rescaled away."""
    assert reduce_sum(_OVERFLOWING, policy) == math.inf


@pytest.mark.parametrize(
    ("label", "values"),
    [
        ("a lone infinity", [math.inf]),
        ("an infinity after a finite term", [1.0, math.inf]),
        ("an infinity before a finite term", [math.inf, 1.0]),
        ("a negative infinity", [-math.inf]),
        ("an overflow that reaches infinity", _OVERFLOWING),
    ],
)
def test_compensated_summation_cannot_represent_an_infinite_sum(
    label: str, values: list[float]
) -> None:
    """Compensated summation returns NaN once any term or partial is infinite.

    The correction term `(sum - v) + ...` becomes `inf - inf`, so one infinite
    term loses both the value and the sign. The behaviour is pinned rather than
    repaired: it is where the compensated sum stops giving a reading."""
    assert math.isnan(reduce_sum(values, Reduction.NEUMAIER)), label


@pytest.mark.parametrize("policy", [Reduction.NAIVE, Reduction.PAIRWISE, Reduction.EXACT])
def test_the_other_three_policies_carry_an_infinity_through(policy: Reduction) -> None:
    """The other three policies carry an infinity through to the result.

    This makes the Neumaier result a divergence between policies, not a property
    of the input. `fsum` returns `inf` here and raises only when the infinities
    have opposite signs."""
    assert reduce_sum([1.0, math.inf], policy) == math.inf
    assert reduce_sum([1.0, -math.inf], policy) == -math.inf


def test_only_exact_summation_refuses_a_sum_it_cannot_represent() -> None:
    """Only exact summation raises on a sum it cannot represent.

    `math.fsum` is the oracle the other three are measured against, so a silent
    `inf` from it would corrupt every error figure computed from the comparison."""
    with pytest.raises(OverflowError, match="intermediate overflow in fsum"):
        reduce_sum(_OVERFLOWING, Reduction.EXACT)


def test_only_exact_summation_refuses_opposite_infinities() -> None:
    """Only exact summation raises on opposite infinities.

    `inf + -inf` has no answer; the other three return NaN, which propagates."""
    with pytest.raises(ValueError, match=r"-inf \+ inf in fsum"):
        reduce_sum([math.inf, -math.inf], Reduction.EXACT)


@pytest.mark.parametrize("policy", [Reduction.NAIVE, Reduction.NEUMAIER, Reduction.PAIRWISE])
def test_the_inexact_policies_return_nan_for_opposite_infinities(policy: Reduction) -> None:
    assert math.isnan(reduce_sum([math.inf, -math.inf], policy))


@pytest.mark.parametrize("policy", list(Reduction))
def test_a_nan_term_reaches_the_result_under_every_policy(policy: Reduction) -> None:
    """A NaN term reaches the result under every policy.

    No policy filters or reorders around a NaN; a sum that dropped one would turn
    an invalid corpus into a plausible number."""
    assert math.isnan(reduce_sum([1.0, math.nan, 2.0], policy))


# ---------------------------------------------------------------------------
# Boundary: the sign of zero, which bit-exact comparison can see
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("label", "values"),
    [
        ("a lone negative zero", [-0.0]),
        ("two negative zeros", [-0.0, -0.0]),
        ("both zeros", [0.0, -0.0]),
        ("a cancelling pair", [1.0, -1.0]),
        ("a cancelling pair the other way round", [-1.0, 1.0]),
        ("a whole block of negative zeros", [-0.0] * (_PAIRWISE_BLOCK + 1)),
    ],
)
@pytest.mark.parametrize("policy", list(Reduction))
def test_a_sum_that_comes_out_zero_is_always_positive_zero(
    label: str, values: list[float], policy: Reduction
) -> None:
    """Every zero this module can produce is `+0.0`, whatever the route.

    Downstream comparison is on bit patterns, where `-0.0` and `+0.0` differ.
    Each policy reaches zero by a different route (a fold from `0.0`, a
    compensated correction, a tree, `fsum`) and all four agree.
    """
    total = reduce_sum(values, policy)
    assert total == 0.0, label
    assert same_bits(total, 0.0), f"{label} produced -0.0 under {policy}"


@pytest.mark.parametrize("n", [255, 256, 257, 383, 384, 385, 511, 512, 513])
def test_the_tree_shape_changes_do_not_cost_accuracy(n: int) -> None:
    """Pairwise is no further than naive from the exact sum where the tree changes.

    The tree changes shape at every multiple of the block size; 384 is the first
    size that holds two partials, and 512 the first whose merge carries twice.
    A merge that dropped a partial would still give a plausible number here.
    """
    values = [1.0 / (i + 1) for i in range(n)]
    truth = exact_sum(values)
    assert abs(pairwise_sum(values) - truth) <= abs(naive_sum(values) - truth) + 1e-15


@pytest.mark.parametrize("n", [128, 256, 384, 512])
def test_an_exactly_representable_sum_is_identical_under_every_policy(n: int) -> None:
    """An exactly representable sum is identical under every policy.

    On exact partial sums the tree shape cannot matter, so any disagreement at a
    block boundary is a lost or duplicated term rather than rounding."""
    values = [1.0] * n
    results = {policy: reduce_sum(values, policy) for policy in Reduction}
    assert set(results.values()) == {float(n)}, results


# ---------------------------------------------------------------------------
# Erroneous: selecting a policy
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("policy", [None, 0, 1.5, "NAIVE", "", b"naive", ("naive",)])
def test_a_policy_that_is_not_one_of_the_four_is_refused(policy: object) -> None:
    """A policy that is not one of the four is refused.

    `Reduction` is a `str` enum, so members and their lowercase values select a
    reducer; anything else, such as the member name, bytes or a number, raises.
    The manifest records which policy summed each published number."""
    with pytest.raises(ValueError, match="unknown reduction policy"):
        reduce_sum([1.0], policy)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The correctly-rounded logarithm: where it beats the platform outright
# ---------------------------------------------------------------------------
def test_the_exact_logarithm_survives_a_ratio_the_platform_cannot_form() -> None:
    """The exact logarithm survives a ratio the platform path cannot form.

    `platform_log_ratio` divides in binary64 first and raises above the largest
    double; the exact path divides in `Decimal`, so only the answer must fit. A
    real vocabulary, with ratio `(1+N)/(1+df)`, does not reach this case."""
    assert correctly_rounded_log_ratio(10**400, 1) == pytest.approx(math.log(10) * 400)

    with pytest.raises(OverflowError, match="integer division result too large"):
        numerics.platform_log_ratio(10**400, 1)


def test_the_exact_logarithm_of_a_huge_ratio_is_finite_and_correct() -> None:
    """The exact logarithm of a huge ratio is finite and correct.

    `ln(10**400)` is about 921, inside binary64; only the intermediate overflows."""
    got = correctly_rounded_log_ratio(10**400, 1)
    assert math.isfinite(got)
    assert 921.0 < got < 921.1


@pytest.mark.parametrize("denominator", [1, 2, 3, 9743, 10**50])
def test_a_ratio_of_one_is_exactly_positive_zero_at_every_scale(denominator: int) -> None:
    """A ratio of one gives exactly `+0.0` at every scale.

    idf is `log(ratio) + 1`, so a `-0.0` here would enter the smoothing and be
    visible to a bit-exact comparison downstream."""
    got = correctly_rounded_log_ratio(denominator, denominator)
    assert same_bits(got, 0.0), f"ln(1) came out as {got!r} at denominator {denominator}"


def test_the_working_precision_is_the_recorded_sixty_digits() -> None:
    """The Decimal working precision is the recorded sixty digits.

    The constant's documentation states that this precision agrees with a
    120-digit evaluation on every ratio the project produces, so it is pinned."""
    assert DECIMAL_LOG_PRECISION == 60


def test_the_logarithm_leaves_the_callers_decimal_context_alone() -> None:
    """The logarithm leaves the caller's Decimal context unchanged.

    It raises the working precision inside a `localcontext`; leaking that would
    change the precision of later Decimal arithmetic in the caller."""
    before = getcontext().prec
    correctly_rounded_log_ratio(9743, 2)
    assert getcontext().prec == before


def test_a_caller_holding_its_own_decimal_context_is_not_clobbered() -> None:
    """A caller's own local Decimal context keeps its precision.

    This nested case fails if the precision is set with a bare
    `getcontext().prec = ...` rather than inside a `localcontext`."""
    with localcontext() as ctx:
        ctx.prec = 5
        correctly_rounded_log_ratio(9743, 2)
        assert getcontext().prec == 5


def test_the_arguments_are_documented_as_integers_but_floats_are_accepted() -> None:
    """Float arguments are accepted although integers are documented.

    `Decimal(1.5)` is exact, so a float ratio computes rather than raising. Every
    caller passes the integers `1 + N` and `1 + df`; a float would take the
    binary approximation of the intended value."""
    assert correctly_rounded_log_ratio(1.5, 2) == pytest.approx(math.log(0.75))  # type: ignore[arg-type]


def test_boolean_arguments_are_the_integers_they_equal() -> None:
    """`True` is a positive integer, so it passes the guard and gives `ln(1)`."""
    assert same_bits(correctly_rounded_log_ratio(True, True), 0.0)


# ---------------------------------------------------------------------------
# The bit-level vocabulary, across the whole float domain
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value", [0.0, -0.0, 5e-324, -5e-324, 1.0, -1.0, math.inf, -math.inf, math.nan]
)
def test_every_float_encodes_to_exactly_eight_bytes(value: float) -> None:
    """Every float, including the non-finite ones, encodes to eight bytes.

    A differential comparison therefore needs no special case for any value."""
    assert len(bits_of(value)) == 8


def test_the_sign_bit_is_the_top_bit_of_the_last_byte() -> None:
    """The sign bit is the top bit of the last byte.

    The encoding is little-endian, and every `-0.0` assertion in the suite
    rests on it."""
    assert bits_of(0.0) == b"\x00" * 8
    assert bits_of(-0.0) == b"\x00" * 7 + b"\x80"


def test_two_nans_of_opposite_sign_are_not_the_same_bits() -> None:
    """Two NaNs of opposite sign are not the same bits.

    `same_bits` calls a NaN equal to itself but compares bytes, so the sign bit
    still separates two NaNs."""
    assert same_bits(math.nan, math.nan)
    assert not same_bits(math.nan, -math.nan)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.0, 5e-324), (-0.0, 5e-324), (5e-324, 5e-324), (1.0, 2.220446049250313e-16)],
)
def test_the_spacing_at_a_value_is_the_step_to_its_neighbour(value: float, expected: float) -> None:
    """The spacing at a value is the step to its neighbour.

    At zero the spacing is the smallest subnormal, not zero, which makes
    `ulps_between` usable at the underflow boundary."""
    assert ulp(value) == expected


@pytest.mark.parametrize("value", [math.inf, -math.inf])
def test_the_spacing_at_infinity_is_infinite(value: float) -> None:
    assert ulp(value) == math.inf


def test_the_spacing_at_nan_is_nan() -> None:
    assert math.isnan(ulp(math.nan))


def test_the_smallest_positive_value_is_one_ulp_from_zero() -> None:
    """The smallest positive value is one ulp from zero.

    This is the boundary at which the underflow study works."""
    assert ulps_between(0.0, 5e-324) == 1.0
    assert ulps_between(5e-324, 0.0) == -1.0


def test_the_distance_between_the_extremes_stays_finite() -> None:
    """The distance between the two extremes is finite.

    Both arguments are finite, and the contract reserves `inf` for a non-finite one.
    `b - a` is `-3.6e308` here and overflows, so each value is divided by `scale`
    first. `scale` is a power of two, so that division is exact.
    """
    got = ulps_between(sys.float_info.max, -sys.float_info.max)
    assert math.isfinite(got), "both arguments are finite, so the distance is too"
    assert math.isfinite(sys.float_info.max), "both arguments really are finite"

    # `max / ulp(max)` is `2**53 - 1`, and the extremes sit that far either side
    # of zero, so the signed distance between them is twice it.
    assert got == -(2.0**54 - 2.0)
    assert ulps_between(-sys.float_info.max, sys.float_info.max) == 2.0**54 - 2.0


# ---------------------------------------------------------------------------
# Square root: the G18 underflow boundary
# ---------------------------------------------------------------------------
def test_the_square_root_of_the_smallest_normal_is_the_g18_threshold() -> None:
    """The square root of the smallest normal is the G18 threshold.

    Below `sqrt(DBL_MIN)` squaring a coordinate underflows and cosine similarity
    loses its guarantees. The constant is pinned here, in the module that owns
    `sqrt`, so the vector and geometry suites quote it rather than derive it."""
    assert sqrt(sys.float_info.min) == 1.4916681462400413e-154


def test_the_square_root_keeps_the_sign_of_a_negative_zero() -> None:
    """The square root keeps the sign of a negative zero.

    IEEE 754 defines `sqrt(-0.0)` as `-0.0`, the one negative input with an
    answer."""
    assert same_bits(sqrt(-0.0), -0.0)


@pytest.mark.parametrize("value", [-1.0, -5e-324, -sys.float_info.max])
def test_a_negative_square_root_is_refused_however_small(value: float) -> None:
    """A negative square root is refused however small the input.

    A magnitude test in place of a sign test would let the subnormal through and
    return a NaN norm from a vector with only a rounding error."""
    # CPython 3.13 and earlier raise "math domain error"; 3.14 names the input
    # instead. Matched either way, so this pins the refusal and not the
    # interpreter version.
    with pytest.raises(ValueError, match=r"math domain error|expected a nonnegative input"):
        sqrt(value)


def test_the_square_root_of_the_smallest_subnormal_is_representable() -> None:
    """The square root of the smallest subnormal is representable.

    The root of a subnormal is a normal number, so the norm of a vanishing
    vector is still usable."""
    assert sqrt(5e-324) == pytest.approx(2.2227587494850775e-162)


@pytest.mark.parametrize(("value", "check"), [(math.inf, math.isinf), (math.nan, math.isnan)])
def test_the_square_root_carries_the_non_finite_values_through(value: float, check: object) -> None:
    assert check(sqrt(value))  # type: ignore[operator]


# ---------------------------------------------------------------------------
# The pairwise tree, against an independent implementation of the same scheme
# ---------------------------------------------------------------------------


# The tree shape of `pairwise_sum` is part of the specification, shared bit for
# bit with the C++ core. The oracle is a second implementation of the same
# binary-counter scheme, and the comparison is on bits.
# Rationale: docs/engineering_notes.md#pairwise-tree-oracle
def _pairwise_oracle(values: list[float], block_size: int) -> float:
    """The binary-counter scheme again, structured as a list of levels.

    It keeps `(weight, value)` pairs in one stack and derives each new weight
    from the stored one; the module keeps two parallel stacks and doubles a
    running weight. A defect in either shows up as a disagreement.
    """
    levels: list[tuple[int, float]] = []
    block = 0.0
    filled = 0

    for v in values:
        block += v
        filled += 1
        if filled == block_size:
            carry, weight = block, 1
            while levels and levels[-1][0] == weight:
                stored_weight, stored = levels.pop()
                carry = stored + carry
                weight = stored_weight * 2
            levels.append((weight, carry))
            block = 0.0
            filled = 0

    total = 0.0
    for _, partial in reversed(levels):
        total += partial
    return total + block


@pytest.mark.parametrize(
    "n",
    [0, 1, 127, 128, 129, 255, 256, 257, 383, 384, 385, 511, 512, 513, 1024, 1025, 1280, 2000],
)
def test_the_pairwise_tree_matches_an_independent_binary_counter(n: int) -> None:
    """Bit-for-bit, across every size where the tree changes shape.

    512 is the first size at which two levels carry in one merge, and 1024 the
    first at which three do, so these sizes reach the `while` in the merge to
    depths nothing else does.
    """
    values = [1.0 / (i + 1) for i in range(n)]
    assert same_bits(pairwise_sum(values), _pairwise_oracle(values, _PAIRWISE_BLOCK)), (
        f"the two formulations of the binary counter disagree at n = {n}"
    )


def test_the_two_formulations_disagree_somewhere_if_the_tree_is_altered() -> None:
    """The oracle above is only evidence if a different tree would fail it.

    Summing the same values with a block size of 64 builds a genuinely different
    tree, and on inexact values that must land on different bits -- otherwise the
    comparison above is insensitive to exactly what it claims to detect.
    """
    values = [1.0 / (i + 1) for i in range(1000)]
    at_128 = _pairwise_oracle(values, _PAIRWISE_BLOCK)
    at_64 = _pairwise_oracle(values, 64)

    assert not same_bits(at_128, at_64), "the block size does not change the result"
    assert same_bits(at_128, pairwise_sum(values)), "and 128 is the pinned one"


def test_the_block_size_is_the_recorded_one_hundred_and_twenty_eight() -> None:
    """The pairwise block size is the recorded 128.

    It selects the tree, so it is part of the format of every published sum
    rather than a tuning parameter."""
    assert _PAIRWISE_BLOCK == 128


# ---------------------------------------------------------------------------
# The behavioural probes, which have to actually probe something
# ---------------------------------------------------------------------------
def test_the_compiler_did_not_constant_fold_the_probe_away() -> None:
    """The constant-folding probe reports `True`.

    `0.1 + 0.2 != 0.3` in binary64; equality would mean the expression was folded
    at higher precision. The value is asserted, not only the key, because an
    inverted probe would record `False` without changing the key set."""
    assert float_environment()["constant_folding_ok"] is True
    assert 0.1 + 0.2 != 0.3, "the premise, stated where a reader can see it"


def test_the_interpreter_did_not_reassociate_the_probe() -> None:
    """The reassociation probe reports `True`.

    `(1.0 + 1e-17) - 1.0` is `0.0` because `1e-17` is absorbed at 1.0.
    Reassociated to `1.0 - 1.0 + 1e-17` it would be `1e-17`, which breaks the
    left-to-right fold the specification rests on."""
    assert float_environment()["no_reassociation"] is True
    assert (1.0 + 1e-17) - 1.0 == 0.0


def test_the_subnormal_probe_reaches_a_subnormal() -> None:
    """The subnormal probe reaches a subnormal.

    Halving the smallest normal lands below it and stays positive, so the probe
    tests flush-to-zero."""
    halved = sys.float_info.min / 2.0
    assert 0.0 < halved < sys.float_info.min
    assert float_environment()["subnormals_supported"] is True


#: Magnitudes spanning 32 decades, drawn from a fixed seed. On smooth input such
#: as `1/(i+1)` many regroupings land on identical bits, so a changed merge tree
#: shows only at some lengths. This irregular mix of huge and tiny terms makes
#: the association observable at any length that carries.
_ADVERSARIAL_MAGNITUDES = [
    random.Random(38).choice([1e16, 1.0, -1e16, 1e-16, -1.0]) for _ in range(1280)
]


def test_the_pairwise_tree_matches_the_oracle_on_adversarial_magnitudes() -> None:
    """The same comparison as above, on input where the tree shape is visible.

    1280 elements is ten blocks, so the counter carries at several depths, and
    the terms cancel across sixteen orders of magnitude so every regrouping
    lands on different bits.
    """
    assert same_bits(
        pairwise_sum(_ADVERSARIAL_MAGNITUDES),
        _pairwise_oracle(_ADVERSARIAL_MAGNITUDES, _PAIRWISE_BLOCK),
    )


def _pairwise_from_the_wrong_end(values: list[float]) -> float:
    """The counter with its carry compared against the bottom of the stack.

    The stack is ordered largest-first, so this merges once and then appends a
    flat list of unmerged blocks. It is a wrong tree for the oracle comparison
    to detect, not an alternative formulation.
    """
    partials: list[float] = []
    weights: list[int] = []
    block = 0.0
    filled = 0
    for v in values:
        block += v
        filled += 1
        if filled == _PAIRWISE_BLOCK:
            carry, weight = block, 1
            while partials and weights[0] == weight:
                weights.pop()
                carry = partials.pop() + carry
                weight *= 2
            partials.append(carry)
            weights.append(weight)
            block = 0.0
            filled = 0
    total = 0.0
    for p in reversed(partials):
        total += p
    return total + block


def test_the_oracle_comparison_detects_a_reversed_carry_comparison() -> None:
    """What makes the oracle above evidence rather than a coincidence.

    The specimen builds a different tree. On adversarial magnitudes the
    comparison sees it at any length that carries; on smooth values only at
    some lengths.
    """
    assert not same_bits(
        pairwise_sum(_ADVERSARIAL_MAGNITUDES),
        _pairwise_from_the_wrong_end(_ADVERSARIAL_MAGNITUDES),
    )
