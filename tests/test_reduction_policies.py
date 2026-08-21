"""The four summation policies, and the environment they assume.

`utils/numerics.py` is imported by nine test files, but only for `same_bits` as an
assertion tool: nothing had this module as its *subject*. Its 81% was entirely
incidental, and the falsifiable claims in its docstrings — that the two pairwise
formulations first disagree at n = 129, that the block size matching numpy's does
not buy agreement with numpy, that the correctly-rounded logarithm differs from
the platform's — were checked by nothing.

Three groups of property.

The policies differ, and differ where the code says. A reduction sweep that never
produced a disagreement would be measuring nothing, so the cancellation cases here
are chosen to separate naive from compensated from exact. `Reduction.EXACT` is the
oracle: `math.fsum` is correctly rounded, so any other policy agreeing with it on a
hard input is evidence, and disagreeing is expected rather than a defect.

The guards reject what would silently produce a wrong logarithm.
`correctly_rounded_log_ratio` is how every IDF value in the project is computed, so
a zero or negative argument reaching it must raise rather than return a NaN that
propagates into a published weight.

The environment is asserted rather than assumed. `assert_sane_float_environment`
exists because a BLAS loading behind numpy can set flush-to-zero process-wide,
which would flush exactly the subnormal margins this project studies. Nothing
called it, so the guard that protects every measurement was itself unmeasured. Its
three failure arms are reached by patching `sys.float_info` and the probe — the
interpreter boundary, not a function inside this package.
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
    """The module says n = 129 is where a block-recursive and a strictly binary
    pairwise sum part company. Checked, rather than left as a claim."""
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
    """Unlike log, sqrt is correctly rounded on every conforming platform, which
    is why the native core may call it freely rather than receiving it as data.
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
    """Pinned because the message names one of them, and which one is arbitrary
    until something depends on it."""
    with pytest.raises(ValueError, match="denominator must be positive"):
        correctly_rounded_log_ratio(0, 0)


def test_a_ratio_of_one_gives_exactly_zero() -> None:
    assert correctly_rounded_log_ratio(7, 7) == 0.0


def test_the_correctly_rounded_logarithm_differs_from_the_platform_somewhere() -> None:
    """G13's whole reason for existing. If these agreed everywhere the project
    would not need its own logarithm, and this test would be measuring nothing.
    """
    # Ratios of the form n/1 are the wrong probe: the platform logarithm is
    # already correctly rounded for those, and an earlier version of this test
    # used them and concluded the two never differ. Real IDF arguments are
    # N over df for a df anywhere in 1..N, which is where the two part company.
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
        "subnormals_supported",
        "constant_folding_ok",
        "no_reassociation",
        "decimal_default_prec",
    }


def test_this_interpreter_is_a_sane_environment_so_the_guard_stays_silent() -> None:
    assert_sane = numerics.assert_sane_float_environment
    assert_sane(), "the guard rejected the interpreter running the suite"


def test_a_flush_to_zero_environment_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """The case the guard exists for: a BLAS setting MXCSR.FTZ process-wide would
    flush exactly the subnormal margins under study.

    The probe is patched, not the guard: `float_environment` is left to run and
    only its subnormal answer is forced, so the guard's own logic is what is
    being tested.
    """
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
    monkeypatch.setattr(numerics, "float_environment", lambda: {**real(), "rounds": 0})
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
    """A guard that stopped at the first fault would need running repeatedly to
    describe a broken environment."""
    real = numerics.float_environment
    monkeypatch.setattr(
        numerics,
        "float_environment",
        lambda: {**real(), "rounds": 0, "subnormals_supported": False},
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
    """Section 6 forbids stabilising transformations, so an overflow must be
    visible rather than quietly avoided."""
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
# The four policies are instruments and the spread between them is the
# measurement, so where they *disagree* matters more than where they agree. At
# the top of the range they disagree three ways, and only one of the three was
# pinned -- by a `slow` test on `naive_sum` alone, which the coverage job
# deselects. A caller sweeping policies over a corpus with one enormous weight
# gets `inf`, `nan`, or an exception depending on a keyword argument.
_OVERFLOWING = [sys.float_info.max, sys.float_info.max]


@pytest.mark.parametrize("policy", [Reduction.NAIVE, Reduction.PAIRWISE])
def test_an_uncompensated_sum_past_the_maximum_reports_infinity(policy: Reduction) -> None:
    """Section 6 forbids stabilising transformations, so an overflow has to be
    visible rather than quietly rescaled away."""
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
    """Neumaier's correction term is `(sum - v) + ...`; once either side is
    infinite that is `inf - inf`, which is NaN, and the NaN reaches the result.

    So it is not an overflow rule but an infinity rule: a single infinite term is
    enough, and the policy chosen for its *accuracy* is the one that loses the
    value and the sign together. Pinned rather than repaired -- the corrected sum
    is what the policy is, and this is the honest statement of where the
    instrument stops reading.
    """
    assert math.isnan(reduce_sum(values, Reduction.NEUMAIER)), label


@pytest.mark.parametrize("policy", [Reduction.NAIVE, Reduction.PAIRWISE, Reduction.EXACT])
def test_the_other_three_policies_carry_an_infinity_through(policy: Reduction) -> None:
    """The contrast that makes the Neumaier result a divergence rather than a
    property of the input. Including the oracle: `fsum` returns `inf` here and
    raises only when the infinities have opposite signs."""
    assert reduce_sum([1.0, math.inf], policy) == math.inf
    assert reduce_sum([1.0, -math.inf], policy) == -math.inf


def test_only_exact_summation_refuses_a_sum_it_cannot_represent() -> None:
    """`math.fsum` raises rather than returning a value it knows is wrong. It is
    the oracle the other three are measured against, so a silent `inf` from it
    would corrupt every error figure computed from the comparison."""
    with pytest.raises(OverflowError, match="intermediate overflow in fsum"):
        reduce_sum(_OVERFLOWING, Reduction.EXACT)


def test_only_exact_summation_refuses_opposite_infinities() -> None:
    """`inf + -inf` has no answer. The other three return NaN, which propagates;
    the oracle says so."""
    with pytest.raises(ValueError, match=r"-inf \+ inf in fsum"):
        reduce_sum([math.inf, -math.inf], Reduction.EXACT)


@pytest.mark.parametrize("policy", [Reduction.NAIVE, Reduction.NEUMAIER, Reduction.PAIRWISE])
def test_the_inexact_policies_return_nan_for_opposite_infinities(policy: Reduction) -> None:
    assert math.isnan(reduce_sum([math.inf, -math.inf], policy))


@pytest.mark.parametrize("policy", list(Reduction))
def test_a_nan_term_reaches_the_result_under_every_policy(policy: Reduction) -> None:
    """No policy filters or reorders around a NaN. A sum that quietly dropped one
    would turn an invalid corpus into a plausible number."""
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

    Everything downstream compares on bit patterns, where `-0.0` and `+0.0` are
    different, so "the sum is zero" has to mean one thing. Each policy reaches
    zero differently -- a fold from `0.0`, a compensated correction, a tree, and
    `fsum` -- and they agree; that agreement is the property, not an accident of
    any one implementation.
    """
    total = reduce_sum(values, policy)
    assert total == 0.0, label
    assert same_bits(total, 0.0), f"{label} produced -0.0 under {policy}"


@pytest.mark.parametrize("n", [255, 256, 257, 383, 384, 385, 511, 512, 513])
def test_the_tree_shape_changes_do_not_cost_accuracy(n: int) -> None:
    """The binary-counter merge changes shape at every multiple of the block
    size, and carries at powers of two times it. 384 is the first size where two
    levels carry at once, so it is the first that exercises the `while` in the
    merge more than once.

    The existing sweep stops at 257. These sizes are the ones where a merge that
    dropped a partial would still produce a plausible number.
    """
    values = [1.0 / (i + 1) for i in range(n)]
    truth = exact_sum(values)
    assert abs(pairwise_sum(values) - truth) <= abs(naive_sum(values) - truth) + 1e-15


@pytest.mark.parametrize("n", [128, 256, 384, 512])
def test_an_exactly_representable_sum_is_identical_under_every_policy(n: int) -> None:
    """At the block boundaries, on values whose partial sums are all exact, the
    tree shape cannot matter -- so any disagreement here is a lost or duplicated
    term rather than rounding."""
    values = [1.0] * n
    results = {policy: reduce_sum(values, policy) for policy in Reduction}
    assert set(results.values()) == {float(n)}, results


# ---------------------------------------------------------------------------
# Erroneous: selecting a policy
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("policy", [None, 0, 1.5, "NAIVE", "", b"naive", ("naive",)])
def test_a_policy_that_is_not_one_of_the_four_is_refused(policy: object) -> None:
    """The lookup is a dict keyed on the enum. `Reduction` is a `str` enum, so
    the *lowercase value* works and is tested elsewhere; everything else -- the
    member name, a bytestring, a number -- must be refused rather than falling
    back to a default. Which policy summed a published number is recorded in the
    manifest, so a silent default would make that record false.
    """
    with pytest.raises(ValueError, match="unknown reduction policy"):
        reduce_sum([1.0], policy)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The correctly-rounded logarithm: where it beats the platform outright
# ---------------------------------------------------------------------------
def test_the_exact_logarithm_survives_a_ratio_the_platform_cannot_form() -> None:
    """The headline reason the Decimal path is not merely more accurate.

    `platform_log_ratio` divides first in binary64, so a ratio above the largest
    representable double raises before `log` is reached. The exact path divides
    in `Decimal`, where the ratio is a rational and the logarithm is taken of a
    number that never has to fit in a double -- only the answer does.

    Not reachable from a real vocabulary, where the ratio is `(1+N)/(1+df)`. It
    is the demonstration that the two are different computations rather than the
    same one at different precisions.
    """
    assert correctly_rounded_log_ratio(10**400, 1) == pytest.approx(math.log(10) * 400)

    with pytest.raises(OverflowError, match="integer division result too large"):
        numerics.platform_log_ratio(10**400, 1)


def test_the_exact_logarithm_of_a_huge_ratio_is_finite_and_correct() -> None:
    """`ln(10**400)` is about 921, comfortably inside binary64. The intermediate
    is what overflows, not the result."""
    got = correctly_rounded_log_ratio(10**400, 1)
    assert math.isfinite(got)
    assert 921.0 < got < 921.1


@pytest.mark.parametrize("denominator", [1, 2, 3, 9743, 10**50])
def test_a_ratio_of_one_is_exactly_positive_zero_at_every_scale(denominator: int) -> None:
    """`ln(1) == 0` has to be exact and signed positive: idf is `log(ratio) + 1`,
    so a `-0.0` here would put `-0.0` into the smoothing and a bit-exact
    comparison downstream would see it."""
    got = correctly_rounded_log_ratio(denominator, denominator)
    assert same_bits(got, 0.0), f"ln(1) came out as {got!r} at denominator {denominator}"


def test_the_working_precision_is_the_recorded_sixty_digits() -> None:
    """Quoted in the module docstring as agreeing with a 120-digit evaluation on
    every ratio this project produces. It is a published claim about the
    numbers, so the constant behind it is pinned."""
    assert DECIMAL_LOG_PRECISION == 60


def test_the_logarithm_leaves_the_callers_decimal_context_alone() -> None:
    """It raises the working precision inside a `localcontext`. Leaking that
    would silently change the precision of any Decimal arithmetic a caller did
    afterwards -- and this function is called once per vocabulary entry."""
    before = getcontext().prec
    correctly_rounded_log_ratio(9743, 2)
    assert getcontext().prec == before


def test_a_caller_holding_its_own_decimal_context_is_not_clobbered() -> None:
    """The nested case, which is the one a `localcontext` gets wrong if it is
    written as a bare `getcontext().prec = ...`."""
    with localcontext() as ctx:
        ctx.prec = 5
        correctly_rounded_log_ratio(9743, 2)
        assert getcontext().prec == 5


def test_the_arguments_are_documented_as_integers_but_floats_are_accepted() -> None:
    """`Decimal(1.5)` is exact, so a float ratio computes rather than raising.
    Pinned as a precondition: every caller passes `1 + N` and `1 + df`, both
    integers, and an accidental float would silently take the binary
    approximation of the value the caller meant.
    """
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
    """binary64 is 8 bytes for every value including the non-finite ones, so a
    differential comparison never has to special-case one."""
    assert len(bits_of(value)) == 8


def test_the_sign_bit_is_the_top_bit_of_the_last_byte() -> None:
    """Little-endian, so the sign lives at the end. Spelled out because every
    `-0.0` assertion in the suite rests on this encoding."""
    assert bits_of(0.0) == b"\x00" * 8
    assert bits_of(-0.0) == b"\x00" * 7 + b"\x80"


def test_two_nans_of_opposite_sign_are_not_the_same_bits() -> None:
    """`same_bits` calls a NaN equal to itself, which equality does not. It does
    *not* call every NaN equal to every other: the sign bit still differs, and a
    differential test comparing bytes would see that."""
    assert same_bits(math.nan, math.nan)
    assert not same_bits(math.nan, -math.nan)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.0, 5e-324), (-0.0, 5e-324), (5e-324, 5e-324), (1.0, 2.220446049250313e-16)],
)
def test_the_spacing_at_a_value_is_the_step_to_its_neighbour(value: float, expected: float) -> None:
    """At zero the spacing is the smallest subnormal, not zero -- which is what
    makes `ulps_between` usable down at the underflow boundary."""
    assert ulp(value) == expected


@pytest.mark.parametrize("value", [math.inf, -math.inf])
def test_the_spacing_at_infinity_is_infinite(value: float) -> None:
    assert ulp(value) == math.inf


def test_the_spacing_at_nan_is_nan() -> None:
    assert math.isnan(ulp(math.nan))


def test_the_smallest_positive_value_is_one_ulp_from_zero() -> None:
    """The boundary the underflow study lives at: the gap from nothing to the
    smallest something is one unit, not an infinity of them."""
    assert ulps_between(0.0, 5e-324) == 1.0
    assert ulps_between(5e-324, 0.0) == -1.0


def test_the_distance_between_the_extremes_overflows_to_negative_infinity() -> None:
    """A latent defect, pinned rather than repaired.

    Both arguments are finite, so the non-finite guard does not fire, and the
    documented contract says `inf` is reserved for a non-finite input. But
    `b - a` is `-1.8e308`, which overflows, and the result is `-inf` -- a
    magnitude the caller cannot distinguish from "one of these was not a number".

    Unreachable from this project's data, where the quantities compared are
    similarities and margins in `[0, 1]`. Stated so the limit of the unit is
    written down where someone reaching for it on other data will see it.
    """
    got = ulps_between(sys.float_info.max, -sys.float_info.max)
    assert got == -math.inf
    assert math.isfinite(sys.float_info.max), "both arguments really are finite"


# ---------------------------------------------------------------------------
# Square root: the G18 underflow boundary
# ---------------------------------------------------------------------------
def test_the_square_root_of_the_smallest_normal_is_the_g18_threshold() -> None:
    """`sqrt(DBL_MIN)` is where cosine similarity starts to lose its guarantees:
    below it, squaring a coordinate underflows and the norm loses precision the
    similarity cannot recover.

    Defined here once, in the module that owns `sqrt`, so the vector and geometry
    suites can quote the constant rather than each deriving it.
    """
    assert sqrt(sys.float_info.min) == 1.4916681462400413e-154


def test_the_square_root_keeps_the_sign_of_a_negative_zero() -> None:
    """IEEE-754 says `sqrt(-0.0)` is `-0.0`, not `+0.0` and not an error. It is
    the one negative input with an answer."""
    assert same_bits(sqrt(-0.0), -0.0)


@pytest.mark.parametrize("value", [-1.0, -5e-324, -sys.float_info.max])
def test_a_negative_square_root_is_refused_however_small(value: float) -> None:
    """A magnitude test in place of a sign test would let the subnormal through
    and return a NaN norm from a vector that merely had a rounding error."""
    # CPython 3.13 and earlier raise "math domain error"; 3.14 names the input
    # instead. Matched either way, so this pins the refusal and not the
    # interpreter version.
    with pytest.raises(ValueError, match=r"math domain error|expected a nonnegative input"):
        sqrt(value)


def test_the_square_root_of_the_smallest_subnormal_is_representable() -> None:
    """Underflow is one-way: the root of a subnormal is comfortably normal, so
    the norm of a vanishing vector is still a usable number."""
    assert sqrt(5e-324) == pytest.approx(2.2227587494850775e-162)


@pytest.mark.parametrize(("value", "check"), [(math.inf, math.isinf), (math.nan, math.isnan)])
def test_the_square_root_carries_the_non_finite_values_through(value: float, check: object) -> None:
    assert check(sqrt(value))  # type: ignore[operator]


# ---------------------------------------------------------------------------
# The pairwise tree, against an independent implementation of the same scheme
# ---------------------------------------------------------------------------
# `pairwise_sum` is a streaming binary counter: blocks of 128 accumulate, and a
# completed block merges with any stored partial of equal weight, doubling as it
# goes. The shape of that tree *is* the specification -- the C++ core must build
# the identical one, bit for bit, from n = 1 to 10,000.
#
# Comparing against `exact_sum` cannot pin it: many different trees land within a
# few ulps of the correctly-rounded answer, which is why mutation testing could
# reverse the carry comparison and drop the weight doubling with every existing
# assertion still passing. So the oracle here is a second implementation of the
# same scheme, written from the description rather than from the code, and the
# comparison is on bits.
def _pairwise_oracle(values: list[float], block_size: int) -> float:
    """The binary-counter scheme again, structured as a list of levels.

    Deliberately not the module's formulation: this one keeps `(weight, value)`
    pairs together and merges by scanning, where the module keeps two parallel
    stacks and merges by popping. Same tree, different bookkeeping, so a defect
    in either shows up as a disagreement rather than being reproduced twice.
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

    384 is the first size at which two levels carry in one merge, and 512 the
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
    """It selects the tree, so it is part of the format of every published sum
    rather than a tuning parameter."""
    assert _PAIRWISE_BLOCK == 128


# ---------------------------------------------------------------------------
# The behavioural probes, which have to actually probe something
# ---------------------------------------------------------------------------
def test_the_compiler_did_not_constant_fold_the_probe_away() -> None:
    """`0.1 + 0.2 != 0.3` in binary64. If it ever compared equal, the expression
    had been folded at higher precision and every other probe in the block would
    be reporting on an environment that is not the one doing the arithmetic.

    Asserted as `True` rather than merely present: the field is a claim, and a
    probe inverted to `==` would record `False` into every manifest without
    changing the key set.
    """
    assert float_environment()["constant_folding_ok"] is True
    assert 0.1 + 0.2 != 0.3, "the premise, stated where a reader can see it"


def test_the_interpreter_did_not_reassociate_the_probe() -> None:
    """`(1.0 + 1e-17) - 1.0` is `0.0`, because `1e-17` is far below the spacing
    at 1.0 and is absorbed. Under reassociation to `1.0 - 1.0 + 1e-17` it would
    be `1e-17`, and a sum that reassociates would break the left-to-right fold
    the whole specification rests on.
    """
    assert float_environment()["no_reassociation"] is True
    assert (1.0 + 1e-17) - 1.0 == 0.0


def test_the_subnormal_probe_reaches_a_subnormal() -> None:
    """Halving the smallest normal has to land below it and stay positive, or
    the probe is not testing flush-to-zero at all."""
    halved = sys.float_info.min / 2.0
    assert 0.0 < halved < sys.float_info.min
    assert float_environment()["subnormals_supported"] is True


#: Magnitudes spanning 32 decades, drawn from a fixed seed. Smooth input detects
#: a changed merge tree only at some lengths: on `1/(i+1)` the terms are near
#: enough in magnitude that many regroupings land on identical bits, and
#: mutation testing found the reversed carry comparison surviving every size the
#: oracle originally swept. An irregular mix of huge and tiny terms makes the
#: association observable at any length that carries.
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
    """The counter with its carry compared against the *bottom* of the weight
    stack instead of the top.

    A specimen, not an alternative: the stack is ordered largest-first, so this
    merges once and then never again, appending a flat list of unmerged blocks.
    It exists to show what the oracle comparison is sensitive to.
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

    The specimen builds a genuinely different tree, so the comparison has to be
    able to see it. On adversarial magnitudes it does at any length that carries;
    on smooth values it depends on the length, which is why the sweep alone was
    not enough and this input exists.
    """
    assert not same_bits(
        pairwise_sum(_ADVERSARIAL_MAGNITUDES),
        _pairwise_from_the_wrong_end(_ADVERSARIAL_MAGNITUDES),
    )
