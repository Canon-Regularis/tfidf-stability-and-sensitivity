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
import struct
import sys

import pytest

from tfidf_stability.utils import numerics
from tfidf_stability.utils.numerics import (
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
    with pytest.raises(NumericEnvironmentError) as caught:
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
