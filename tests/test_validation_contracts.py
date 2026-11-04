"""Tests for the exception tree and the validators in ``utils/validation.py``.

Covers the error hierarchy, the split between ``check_finite`` and
``check_non_negative``, the tau and prefix guards, and the guard order of ``resolve_k``.
Rationale: docs/engineering_notes.md#validation-contracts
"""

from __future__ import annotations

import json
import math
import re
import sys

import pytest
import yaml

from tfidf_stability import _native
from tfidf_stability.utils import validation
from tfidf_stability.utils.validation import (
    AbiVersionMismatchError,
    DuplicateIdentifierError,
    KOutOfRangeError,
    NativeBackendUnavailableError,
    StrictMode,
    TfidfStabilityError,
    check_finite,
    check_non_negative,
    check_prefix_k,
    check_tau,
    check_unique_ids,
    resolve_k,
)

#: Finite values at the edges of what binary64 holds. Every one of these must
#: pass `check_finite`, so a guard that reached for a magnitude test instead of
#: `math.isfinite` fails here rather than on a real corpus.
_FINITE_EXTREMES = [0.0, -0.0, 5e-324, -5e-324, sys.float_info.min, sys.float_info.max]


# ---------------------------------------------------------------------------
# The exception hierarchy
# ---------------------------------------------------------------------------
def test_every_exported_error_descends_from_the_package_base() -> None:
    """Every exported error class is a subclass of `TfidfStabilityError`.

    `except TfidfStabilityError` is the documented way to catch a refusal. A class
    outside that tree reaches a caller as an unhandled error, like a bug.
    """
    errors = [n for n in validation.__all__ if n.endswith("Error")]
    assert len(errors) >= 9, f"only found {errors}; the walk is not seeing the module"

    for name in errors:
        cls = getattr(validation, name)
        assert issubclass(cls, TfidfStabilityError), f"{name} is outside the error tree"
        assert issubclass(cls, Exception)


def test_every_exported_warning_is_a_warning_and_not_an_error() -> None:
    """Each exported warning is a `UserWarning` and not a package error.

    The two diagnostics tag a degenerate result rather than abort it. A sweep that
    catches `TfidfStabilityError` around a tau grid must not swallow them.
    """
    warnings_exported = [n for n in validation.__all__ if n.endswith("Warning")]
    assert sorted(warnings_exported) == ["ChainInflationWarning", "TauExceedsScoreRangeWarning"]

    for name in warnings_exported:
        cls = getattr(validation, name)
        assert issubclass(cls, UserWarning)
        assert not issubclass(cls, TfidfStabilityError), f"{name} must not be catchable as an error"


def test_a_stale_extension_is_a_kind_of_unavailable_backend() -> None:
    """`AbiVersionMismatchError` subclasses `NativeBackendUnavailableError`.

    A subclass rather than a sibling, so `except NativeBackendUnavailableError`
    serves callers that only need to know whether to fall back.
    """
    assert issubclass(AbiVersionMismatchError, NativeBackendUnavailableError)
    assert issubclass(NativeBackendUnavailableError, TfidfStabilityError)

    with pytest.raises(NativeBackendUnavailableError, match="stale"):
        raise AbiVersionMismatchError("stale")


def test_the_two_backend_failures_stay_distinguishable() -> None:
    """The generic unavailable error is not a subclass of the ABI error.

    The two need different fixes, a rebuild against the current contract or a
    first build, so a caller must be able to tell them apart.
    """
    assert not issubclass(NativeBackendUnavailableError, AbiVersionMismatchError)


def test_an_absent_backend_is_reported_as_never_having_been_built(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The generic error, and the reason string the loader recorded at import."""
    monkeypatch.setattr(_native, "_MODULE", None)
    monkeypatch.setattr(_native, "_ABI_MISMATCH", False)
    monkeypatch.setattr(_native, "_REASON", "the extension was never compiled")

    with pytest.raises(NativeBackendUnavailableError, match="never compiled") as caught:
        _native.require_native()
    assert type(caught.value) is NativeBackendUnavailableError, "not the ABI subclass"


def test_a_stale_extension_is_reported_as_an_abi_mismatch(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A stale extension raises `AbiVersionMismatchError`.

    A stale `.pyd` is reported differently from a machine with no compiler.
    """
    monkeypatch.setattr(_native, "_MODULE", None)
    monkeypatch.setattr(_native, "_ABI_MISMATCH", True)
    monkeypatch.setattr(_native, "_REASON", "reports ABI '0.1.0' but this build expects '0.4.0'")

    with pytest.raises(AbiVersionMismatchError, match=r"ABI '0\.1\.0'"):
        _native.require_native()


def test_an_unavailable_backend_with_no_recorded_reason_still_says_something(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """With no recorded reason, the error still carries a default message.

    `_REASON` is set by the import-time loader, but a caller that monkeypatches
    the module out need not set it. An empty message would hide the cause.
    """
    monkeypatch.setattr(_native, "_MODULE", None)
    monkeypatch.setattr(_native, "_ABI_MISMATCH", False)
    monkeypatch.setattr(_native, "_REASON", None)

    with pytest.raises(NativeBackendUnavailableError, match="the native backend is unavailable"):
        _native.require_native()


# ---------------------------------------------------------------------------
# StrictMode: it reaches every run manifest
# ---------------------------------------------------------------------------
def test_the_mode_serialises_as_its_bare_value() -> None:
    """`str` of a mode is its bare value.

    The mode is written into every manifest, and `Enum.__str__` would render
    `StrictMode.STRICT`, a string that does not read back.
    """
    assert str(StrictMode.STRICT) == "strict"
    assert str(StrictMode.LENIENT) == "lenient"


def test_the_mode_survives_a_round_trip_through_json_and_yaml() -> None:
    """Each mode round-trips through JSON and YAML as itself.

    Manifests are JSON and configs are YAML; a recorded run is reproducible from
    its record only if the mode reads back from both.
    """
    for mode in StrictMode:
        assert StrictMode(json.loads(json.dumps(mode))) is mode
        assert StrictMode(yaml.safe_load(yaml.safe_dump(str(mode)))) is mode


@pytest.mark.parametrize("name", ["STRICT", "Strict", "", "lenient ", "loose"])
def test_a_mode_name_that_is_not_one_of_the_two_is_refused(name: str) -> None:
    """A mode name other than the two exact values is refused.

    Config values arrive as text. Case and whitespace are not forgiven: a silent
    default would change which queries contribute to a reported distribution
    without changing the manifest.
    """
    with pytest.raises(ValueError, match="is not a valid StrictMode"):
        StrictMode(name)


# ---------------------------------------------------------------------------
# resolve_k -- normal
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("k", "n"), [(1, 1), (1, 5), (4, 5), (5, 5)])
@pytest.mark.parametrize("mode", list(StrictMode))
def test_a_k_the_corpus_can_serve_is_returned_unchanged(k: int, n: int, mode: StrictMode) -> None:
    """A `k` in `[1, n]` is returned unchanged in either mode.

    `k == n` is the largest admissible request: `r_k` exists and only `r_{k+1}`
    does not, which concerns margins rather than producing the ranking.
    """
    assert resolve_k(k, n, mode) == k


# ---------------------------------------------------------------------------
# resolve_k -- erroneous
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k", [0, -1, -(2**63)])
@pytest.mark.parametrize("mode", list(StrictMode))
def test_a_non_positive_k_is_refused_in_either_mode(k: int, mode: StrictMode) -> None:
    """A non-positive `k` is refused in either mode.

    Lenient mode clamps an over-large `k` because a degenerate sweep point is
    legitimate. A non-positive `k` has nothing to clamp to, so both modes refuse.
    """
    with pytest.raises(KOutOfRangeError, match=f"k must be positive, got {k}"):
        resolve_k(k, 5, mode)


@pytest.mark.parametrize("k", [6, 2**63, 10**30])
def test_an_over_large_k_is_refused_in_strict_mode(k: int) -> None:
    """Strict mode refuses `k > n` with a message naming both numbers.

    A caller who hits this in a sweep needs lenient mode rather than a smaller k,
    so the message also names that way out.
    """
    with pytest.raises(KOutOfRangeError, match=f"k={k} exceeds the 5 rankable documents"):
        resolve_k(k, 5, StrictMode.STRICT)


def test_the_strict_refusal_names_the_mode_that_would_have_worked() -> None:
    """The strict refusal names `StrictMode.LENIENT` as the remedy.

    G3 makes k > N a protocol-level occurrence, not a mistake, so the error routes
    the reader to the mode that treats it as one.
    """
    with pytest.raises(KOutOfRangeError, match=r"Use StrictMode\.LENIENT to clamp\."):
        resolve_k(9, 5, StrictMode.STRICT)


@pytest.mark.parametrize("k", [6, 2**63, 10**30])
def test_an_over_large_k_clamps_to_the_corpus_in_lenient_mode(k: int) -> None:
    """Lenient mode clamps any `k > n` to `n`.

    The caller records `k_effective`, which makes a rate over a clamped k readable
    beside one over an unclamped k.
    """
    assert resolve_k(k, 5, StrictMode.LENIENT) == 5


# ---------------------------------------------------------------------------
# resolve_k -- boundary and degenerate
# ---------------------------------------------------------------------------
def test_an_empty_corpus_clamps_to_zero_rather_than_refusing_in_lenient_mode() -> None:
    """Lenient mode clamps to 0 on an empty corpus rather than refusing.

    A margin over an empty score array is undefined, not an error; only *ranking*
    an empty corpus raises (G17). The caller reports an undefined margin and the
    sweep continues.
    """
    assert resolve_k(1, 0, StrictMode.LENIENT) == 0


def test_the_positivity_check_runs_before_the_range_check() -> None:
    """The positivity check runs before the range check.

    Both fire for `k = 0` on an empty corpus. "k must be positive" is actionable,
    whereas "k=0 exceeds the 0 rankable documents" reads as a contradiction.
    """
    with pytest.raises(KOutOfRangeError, match="k must be positive, got 0"):
        resolve_k(0, 0, StrictMode.STRICT)


def test_a_negative_corpus_size_is_a_precondition_rather_than_a_checked_input() -> None:
    """A negative `n` is returned as is, because `n` is a precondition.

    `n` comes from `len(...)` at every call site and is not validated. A guard
    would be unreachable; the returned value exposes a violated precondition.
    """
    assert resolve_k(1, -5, StrictMode.LENIENT) == -5


def test_a_boolean_k_is_the_integer_it_equals() -> None:
    """A boolean `k` is treated as the integer it equals.

    `bool` is an `int`, so `True` is a positive k of 1 and passes. A config with
    `k: true` therefore ranks one document rather than being refused.
    """
    assert resolve_k(True, 5, StrictMode.STRICT) == 1


def test_only_an_explicit_lenient_mode_clamps_an_over_large_k() -> None:
    """Only an explicit lenient mode clamps an over-large `k`.

    `StrictMode` is a `str` enum, so `"strict"` compares equal to the strict member
    without being it. The clamp is guarded on `LENIENT`, so the string spelling of
    strict, and any value unequal to `LENIENT`, raise. G3 forbids a silent clamp.
    """
    assert StrictMode.STRICT == "strict", "the premise: equality holds"
    assert StrictMode.STRICT is not "strict"  # noqa: F632 - the premise, stated

    refused = 0
    for mode in (StrictMode.STRICT, "strict", "STRICT", "nonsense", None, 0):
        with pytest.raises(KOutOfRangeError, match="k=9 exceeds the 5 rankable documents"):
            resolve_k(9, 5, mode)  # type: ignore[arg-type]
        refused += 1
    assert refused == 6

    # The one mode that clamps, by either spelling of it.
    assert resolve_k(9, 5, StrictMode.LENIENT) == 5
    assert resolve_k(9, 5, "lenient") == 5  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# check_finite
# ---------------------------------------------------------------------------
def test_an_empty_sequence_has_nothing_to_reject() -> None:
    """`check_finite` accepts an empty sequence.

    Every caller reaches this on an empty corpus, and a guard that raised on
    emptiness would abort them.
    """
    assert check_finite([], "scores") is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_each_non_finite_value_is_named_with_its_position(bad: float) -> None:
    """The index is what makes the error actionable on a corpus of thousands."""
    with pytest.raises(TfidfStabilityError, match=rf"scores\[2\] is {bad!r}, which is not finite"):
        check_finite([0.0, 1.0, bad], "scores")


def test_the_first_offending_position_is_the_one_reported() -> None:
    """The earliest non-finite position is the one reported.

    The loop returns on the first failure, so the index is the earliest offender
    rather than the last or an arbitrary one.
    """
    with pytest.raises(TfidfStabilityError, match=r"scores\[1\]"):
        check_finite([1.0, float("nan"), float("inf")], "scores")

    with pytest.raises(TfidfStabilityError, match=r"scores\[0\]"):
        check_finite([float("inf"), float("nan")], "scores")


@pytest.mark.parametrize("value", _FINITE_EXTREMES)
def test_the_extremes_of_binary64_are_finite(value: float) -> None:
    """The finite extremes of binary64 pass `check_finite`.

    Both zeros and both smallest subnormals are included. The check is about
    representability, not magnitude; a range test would reject the values a
    low-norm corpus produces.
    """
    assert check_finite([value], "weights") is None


def test_the_refusal_explains_why_a_non_finite_sort_key_is_worse_than_wrong() -> None:
    """The refusal states that a NaN sort key is undefined behaviour.

    A NaN makes the comparator non-transitive, which is undefined behaviour in the
    native sort rather than a wrong ordering, so the check is not advisory.
    """
    with pytest.raises(TfidfStabilityError, match="undefined behaviour for the underlying sort"):
        check_finite([float("nan")], "scores")


def test_a_non_numeric_value_is_not_laundered_into_a_package_error() -> None:
    """A non-numeric value raises `TypeError`, not a package error.

    `math.isfinite` raises and the validator does not catch it. A string in a
    score array is an upstream programming error, not an unrankable corpus.
    """
    with pytest.raises(TypeError, match="must be real number"):
        check_finite(["x"], "scores")  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# check_non_negative, and where it differs from check_finite
# ---------------------------------------------------------------------------
def test_negative_zero_is_not_a_negative_weight() -> None:
    """Negative zero passes `check_non_negative`.

    `-0.0 < 0.0` is False. A zero weight is what an absent term has, and refusing
    it would refuse an ordinary sparse vector built by subtraction.
    """
    assert check_non_negative([-0.0], "weights") is None


def test_a_nan_weight_passes_the_sign_check_and_is_check_finites_business() -> None:
    """A NaN weight passes the sign check and is refused by `check_finite`.

    Every comparison with NaN is False, so `v < 0.0` cannot catch one. This
    validator owns the orthant and `check_finite` owns representability; a
    caller wanting both calls both.
    """
    assert check_non_negative([float("nan")], "weights") is None

    with pytest.raises(TfidfStabilityError, match="not finite"):
        check_finite([float("nan")], "weights")


def test_the_smallest_representable_negative_is_still_negative() -> None:
    """The smallest negative subnormal is refused like any negative value.

    The check is a sign test, not a magnitude test.
    """
    with pytest.raises(TfidfStabilityError, match=r"weights\[0\] = -5e-324 is negative"):
        check_non_negative([-5e-324], "weights")


@pytest.mark.parametrize("value", [0.0, 5e-324, 1.0, sys.float_info.max, float("inf")])
def test_a_non_negative_value_passes_however_large(value: float) -> None:
    """Any non-negative value passes, including `+inf`.

    Unbounded above is not this check's concern; that belongs to `check_finite`.
    """
    assert check_non_negative([value], "weights") is None


def test_a_negative_weight_is_named_with_its_position_and_its_consequence() -> None:
    """A negative weight is reported with its position and its consequence.

    The message cites section 2.2 because a negative weight voids
    `cos in [0, 1]`, which is more than merely unusual.
    """
    with pytest.raises(TfidfStabilityError, match=r"weights\[1\] = -1\.0 is negative"):
        check_non_negative([0.5, -1.0, 2.0], "weights")

    with pytest.raises(TfidfStabilityError, match=r"cos in \[0, 1\] depends on it"):
        check_non_negative([-1.0], "weights")


# ---------------------------------------------------------------------------
# check_tau and check_prefix_k
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tau", [0.0, -0.0, 5e-324, 0.2, math.inf])
def test_a_non_negative_tau_is_accepted(tau: float) -> None:
    """Both zeros, the smallest subnormal, an ordinary tau and infinity pass."""
    assert check_tau(tau) is None


@pytest.mark.parametrize("tau", [-5e-324, -0.2, -math.inf, math.nan])
def test_a_negative_or_nan_tau_is_refused_with_its_value(tau: float) -> None:
    """A negative or NaN tau is refused, and the message names the value."""
    with pytest.raises(ValueError, match=rf"^tau must be non-negative, got {re.escape(str(tau))}$"):
        check_tau(tau)


@pytest.mark.parametrize("k", [0, 1, 50])
def test_a_non_negative_k_bounds_a_prefix(k: int) -> None:
    """``k = 0``, the empty prefix, passes, as does any positive ``k``."""
    assert check_prefix_k(k, "top_k") is None


@pytest.mark.parametrize("caller", ["top_k", "compare_top_k"])
def test_a_negative_k_is_refused_in_its_callers_words(caller: str) -> None:
    """The message names the caller, so each public function keeps its own wording."""
    with pytest.raises(
        ValueError, match=rf"^{caller}\(-1\) is not a prefix; k must be non-negative$"
    ):
        check_prefix_k(-1, caller)


# ---------------------------------------------------------------------------
# check_unique_ids
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ids", [[], ["a"], ["a", "b", "c"]])
def test_identifiers_that_are_already_distinct_pass(ids: list[str]) -> None:
    assert check_unique_ids(ids) is None


def test_a_repeat_names_both_positions_it_appeared_at() -> None:
    """A repeat reports both positions at which the identifier appears.

    On a corpus of thousands, the first occurrence is the one the reader needs.
    """
    with pytest.raises(DuplicateIdentifierError, match="'a' appears at positions 0 and 1"):
        check_unique_ids(["a", "a"])

    with pytest.raises(DuplicateIdentifierError, match="'a' appears at positions 0 and 2"):
        check_unique_ids(["a", "b", "a"])


def test_the_refusal_says_why_duplicate_identifiers_are_fatal() -> None:
    """The duplicate refusal states why unique identifiers are required.

    The final tie-break key is the identifier; with a duplicate the comparator is
    not a strict total order and the result depends on the sort.
    """
    with pytest.raises(DuplicateIdentifierError, match="requires unique identifiers"):
        check_unique_ids(["a", "a"])


def test_identifiers_that_merely_look_alike_are_distinct() -> None:
    """Identifiers that print alike but are unequal are distinct.

    `1` and `"1"` print the same and are two documents; collapsing them would
    drop one from the corpus.
    """
    assert check_unique_ids([1, "1"]) is None


@pytest.mark.parametrize(("pair", "shown"), [([1, True], "True"), ([0, False], "False")])
def test_a_boolean_collides_with_the_integer_it_equals(pair: list[object], shown: str) -> None:
    """A boolean identifier collides with the integer it equals.

    `bool` is an `int`, so `1 == True` and they share a hash. A JSON corpus can
    carry `"doc_id": 1` and `"doc_id": true`; passed unconverted, the two collide
    rather than one silently overwriting the other.
    """
    with pytest.raises(DuplicateIdentifierError, match=f"identifier {shown} appears"):
        check_unique_ids(pair)


def test_two_distinct_nan_identifiers_do_not_collide() -> None:
    """Two distinct NaN objects are two distinct identifiers.

    NaN is unequal to itself and, from Python 3.10, distinct NaN objects hash
    differently. The result is degenerate, but it is what the dict does.
    """
    first, second = float("nan"), float("nan")
    assert first is not second
    assert check_unique_ids([first, second]) is None


def test_the_same_nan_object_twice_does_collide() -> None:
    """The same NaN object used twice is a duplicate.

    The dict's identity check runs before equality, so one object used twice
    collides even though `x == x` is False. Identity, not equality, decides.
    """
    same = float("nan")
    with pytest.raises(DuplicateIdentifierError, match="appears at positions 0 and 1"):
        check_unique_ids([same, same])


def test_an_unhashable_identifier_is_not_laundered_into_a_package_error() -> None:
    """An unhashable identifier raises `TypeError`, not a package error.

    A list cannot be a dict key. As with the `check_finite` string case, this is
    an upstream programming error rather than a corpus shape worth naming.
    """
    with pytest.raises(TypeError, match="unhashable type"):
        check_unique_ids(["a", ["b"]])


def test_identifiers_are_checked_in_order_so_the_earliest_repeat_wins() -> None:
    """With two duplicates, the first repeat to complete is the one reported.

    This keeps the message deterministic under a reordering that does not change
    the set of duplicates.
    """
    with pytest.raises(DuplicateIdentifierError, match="'b' appears at positions 1 and 2"):
        check_unique_ids(["a", "b", "b", "a"])


def test_the_math_module_is_what_decides_finiteness() -> None:
    """`check_finite` agrees with `math.isfinite` on each value here.

    A guard written as `v != v or abs(v) == inf` would agree on these values but
    differ from `math.isfinite` on a Decimal or a Fraction.
    """
    for value in [*_FINITE_EXTREMES, float("nan"), float("inf")]:
        expected = math.isfinite(value)
        try:
            check_finite([value], "v")
            actual = True
        except TfidfStabilityError:
            actual = False
        assert actual == expected, f"{value!r} disagreed with math.isfinite"
