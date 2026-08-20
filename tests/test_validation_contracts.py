"""The typed exceptions and the four validators everything else is built on.

Twelve test files import from ``utils/validation.py`` and none owns it. That is
not a filing problem: it is why ``NativeBackendUnavailableError`` and
``AbiVersionMismatchError`` were defined, exported, and asserted by nothing at
all, and why the validators are each exercised at exactly one input by whichever
caller happens to reach them.

The three properties this file establishes.

**The hierarchy is the contract.** Callers catch ``TfidfStabilityError`` to mean
"this package refused on purpose", and catch ``NativeBackendUnavailableError`` to
mean "fall back to the reference backend". A new error class filed outside that
tree silently escapes both, so the tree is asserted by walking ``__all__`` rather
than by listing the classes anyone remembered.

**The validators disagree with each other deliberately.** ``-0.0`` and ``NaN``
pass ``check_non_negative`` and fail ``check_finite``; the division is not an
oversight but the reason both exist. Asserted contrastively, because a reader
who does not know which one rejects NaN will reach for the wrong guard.

**Guard order is observable.** ``resolve_k`` has three sentences and a caller
sees only the first one that fires. Which error a bad ``k`` produces is part of
the interface, so each sentence is reached at an input the others do not claim.

What this file does not do is re-test the callers. ``boundary_margin`` and
``rank_top_k`` own their own ``k`` handling; here ``resolve_k`` is the subject.
"""

from __future__ import annotations

import json
import math
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
    """`except TfidfStabilityError` is the documented way to catch a deliberate
    refusal. A class filed outside that tree escapes it, and would reach a caller
    as an unhandled error indistinguishable from a bug."""
    errors = [n for n in validation.__all__ if n.endswith("Error")]
    assert len(errors) >= 9, f"only found {errors}; the walk is not seeing the module"

    for name in errors:
        cls = getattr(validation, name)
        assert issubclass(cls, TfidfStabilityError), f"{name} is outside the error tree"
        assert issubclass(cls, Exception)


def test_every_exported_warning_is_a_warning_and_not_an_error() -> None:
    """The two diagnostics tag a degenerate result rather than aborting it, so
    they must not be catchable as a refusal -- a sweep that caught
    `TfidfStabilityError` around a tau grid would otherwise swallow them."""
    warnings_exported = [n for n in validation.__all__ if n.endswith("Warning")]
    assert sorted(warnings_exported) == ["ChainInflationWarning", "TauExceedsScoreRangeWarning"]

    for name in warnings_exported:
        cls = getattr(validation, name)
        assert issubclass(cls, UserWarning)
        assert not issubclass(cls, TfidfStabilityError), f"{name} must not be catchable as an error"


def test_a_stale_extension_is_a_kind_of_unavailable_backend() -> None:
    """A subclass rather than a sibling, so `except NativeBackendUnavailableError`
    keeps working for callers that only want to know whether to fall back."""
    assert issubclass(AbiVersionMismatchError, NativeBackendUnavailableError)
    assert issubclass(NativeBackendUnavailableError, TfidfStabilityError)

    with pytest.raises(NativeBackendUnavailableError):
        raise AbiVersionMismatchError("stale")


def test_the_two_backend_failures_stay_distinguishable() -> None:
    """They need different fixes -- rebuild against the current contract, or
    build at all -- so a caller that wants to say which must be able to."""
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
    """Before this path existed the ABI case raised the generic error, so a
    stale `.pyd` looked exactly like a machine with no compiler."""
    monkeypatch.setattr(_native, "_MODULE", None)
    monkeypatch.setattr(_native, "_ABI_MISMATCH", True)
    monkeypatch.setattr(_native, "_REASON", "reports ABI '0.1.0' but this build expects '0.4.0'")

    with pytest.raises(AbiVersionMismatchError, match=r"ABI '0\.1\.0'"):
        _native.require_native()


def test_an_unavailable_backend_with_no_recorded_reason_still_says_something(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """`_REASON` is populated by the import-time loader, but a caller that
    monkeypatched the module out has no reason to have set it. An empty message
    would leave the fallback undiagnosable."""
    monkeypatch.setattr(_native, "_MODULE", None)
    monkeypatch.setattr(_native, "_ABI_MISMATCH", False)
    monkeypatch.setattr(_native, "_REASON", None)

    with pytest.raises(NativeBackendUnavailableError, match="the native backend is unavailable"):
        _native.require_native()


# ---------------------------------------------------------------------------
# StrictMode: it reaches every run manifest
# ---------------------------------------------------------------------------
def test_the_mode_serialises_as_its_bare_value() -> None:
    """It is written into every manifest, and `Enum.__str__` would render
    `StrictMode.STRICT` -- a string that does not read back."""
    assert str(StrictMode.STRICT) == "strict"
    assert str(StrictMode.LENIENT) == "lenient"


def test_the_mode_survives_a_round_trip_through_json_and_yaml() -> None:
    """Manifests are JSON and configs are YAML; the mode has to come back as
    itself from both or a recorded run cannot be reproduced from its record."""
    for mode in StrictMode:
        assert StrictMode(json.loads(json.dumps(mode))) is mode
        assert StrictMode(yaml.safe_load(yaml.safe_dump(str(mode)))) is mode


@pytest.mark.parametrize("name", ["STRICT", "Strict", "", "lenient ", "loose"])
def test_a_mode_name_that_is_not_one_of_the_two_is_refused(name: str) -> None:
    """Config values arrive as text. Case and whitespace are not forgiven,
    because a mode silently defaulting would change which queries contribute to
    a reported distribution without changing the manifest."""
    with pytest.raises(ValueError, match="is not a valid StrictMode"):
        StrictMode(name)


# ---------------------------------------------------------------------------
# resolve_k -- normal
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("k", "n"), [(1, 1), (1, 5), (4, 5), (5, 5)])
@pytest.mark.parametrize("mode", list(StrictMode))
def test_a_k_the_corpus_can_serve_is_returned_unchanged(k: int, n: int, mode: StrictMode) -> None:
    """Including `k == n`, the largest admissible request: `r_k` exists there and
    only `r_{k+1}` does not, which is a question about margins rather than about
    whether the ranking can be produced."""
    assert resolve_k(k, n, mode) == k


# ---------------------------------------------------------------------------
# resolve_k -- erroneous
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k", [0, -1, -(2**63)])
@pytest.mark.parametrize("mode", list(StrictMode))
def test_a_non_positive_k_is_refused_in_either_mode(k: int, mode: StrictMode) -> None:
    """Lenient mode clamps an over-large k because a degenerate sweep point is
    legitimate. A non-positive k is not a sweep point; there is nothing to clamp
    it to, so both modes refuse."""
    with pytest.raises(KOutOfRangeError, match=f"k must be positive, got {k}"):
        resolve_k(k, 5, mode)


@pytest.mark.parametrize("k", [6, 2**63, 10**30])
def test_an_over_large_k_is_refused_in_strict_mode(k: int) -> None:
    """The message names both numbers and the way out, because the caller who
    hits this in a sweep wants lenient mode rather than a smaller k."""
    with pytest.raises(KOutOfRangeError, match=f"k={k} exceeds the 5 rankable documents"):
        resolve_k(k, 5, StrictMode.STRICT)


def test_the_strict_refusal_names_the_mode_that_would_have_worked() -> None:
    """G3 makes k > N a protocol-level occurrence, not a mistake, so the error
    has to route the reader to the mode that treats it as one."""
    with pytest.raises(KOutOfRangeError, match=r"Use StrictMode\.LENIENT to clamp\."):
        resolve_k(9, 5, StrictMode.STRICT)


@pytest.mark.parametrize("k", [6, 2**63, 10**30])
def test_an_over_large_k_clamps_to_the_corpus_in_lenient_mode(k: int) -> None:
    """However far past N the request goes, the effective k is N -- the caller
    then records `k_effective`, which is what makes a rate over a clamped k
    readable beside one over an unclamped k."""
    assert resolve_k(k, 5, StrictMode.LENIENT) == 5


# ---------------------------------------------------------------------------
# resolve_k -- boundary and degenerate
# ---------------------------------------------------------------------------
def test_an_empty_corpus_clamps_to_zero_rather_than_refusing_in_lenient_mode() -> None:
    """A margin over an empty score array is undefined, not an error -- only
    *ranking* an empty corpus raises (G17). So the effective k here is 0, and the
    caller reports an undefined margin rather than aborting the sweep."""
    assert resolve_k(1, 0, StrictMode.LENIENT) == 0


def test_the_positivity_check_runs_before_the_range_check() -> None:
    """Both sentences fire for `k = 0` on an empty corpus. Which error the caller
    sees is part of the interface: "k must be positive" is actionable, whereas
    "k=0 exceeds the 0 rankable documents" reads as a contradiction."""
    with pytest.raises(KOutOfRangeError, match="k must be positive, got 0"):
        resolve_k(0, 0, StrictMode.STRICT)


def test_a_negative_corpus_size_is_a_precondition_rather_than_a_checked_input() -> None:
    """`n` comes from `len(...)` at every call site, so it is never validated.
    Pinned rather than guarded: a guard here would be unreachable, and the value
    returned makes the precondition visible if one ever is violated."""
    assert resolve_k(1, -5, StrictMode.LENIENT) == -5


def test_a_boolean_k_is_the_integer_it_equals() -> None:
    """`bool` is an `int` in Python, so `True` is a positive k of 1 and passes.
    Pinned because a config that produced `k: true` would silently rank one
    document rather than being refused."""
    assert resolve_k(True, 5, StrictMode.STRICT) == 1


def test_a_mode_given_as_a_string_clamps_instead_of_refusing() -> None:
    """A latent trap, pinned rather than fixed.

    `StrictMode` is a `str` enum and the check is `mode is StrictMode.STRICT`, so
    the string `"strict"` compares equal to the member and *is not* it -- an
    over-large k then takes the lenient branch and clamps. No current caller can
    reach this: every one passes the member. It would become live the moment a
    config key were read straight through into `mode`.
    """
    assert StrictMode.STRICT == "strict", "the premise: equality holds"
    assert resolve_k(9, 5, "strict") == 5  # type: ignore[arg-type]

    with pytest.raises(KOutOfRangeError):
        resolve_k(9, 5, StrictMode.STRICT)


# ---------------------------------------------------------------------------
# check_finite
# ---------------------------------------------------------------------------
def test_an_empty_sequence_has_nothing_to_reject() -> None:
    """The loop body never runs. Worth stating because every caller reaches this
    on an empty corpus and a guard that raised on emptiness would abort them."""
    assert check_finite([], "scores") is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_each_non_finite_value_is_named_with_its_position(bad: float) -> None:
    """The index is what makes the error actionable on a corpus of thousands."""
    with pytest.raises(TfidfStabilityError, match=rf"scores\[2\] is {bad!r}, which is not finite"):
        check_finite([0.0, 1.0, bad], "scores")


def test_the_first_offending_position_is_the_one_reported() -> None:
    """The loop returns on the first failure, so the reported index is the
    earliest offender rather than the last or an arbitrary one."""
    with pytest.raises(TfidfStabilityError, match=r"scores\[1\]"):
        check_finite([1.0, float("nan"), float("inf")], "scores")

    with pytest.raises(TfidfStabilityError, match=r"scores\[0\]"):
        check_finite([float("inf"), float("nan")], "scores")


@pytest.mark.parametrize("value", _FINITE_EXTREMES)
def test_the_extremes_of_binary64_are_finite(value: float) -> None:
    """Including both zeros and both smallest subnormals: the check is about
    representability, not magnitude, and a guard written as a range test would
    reject the very values a low-norm corpus produces."""
    assert check_finite([value], "weights") is None


def test_the_refusal_explains_why_a_non_finite_sort_key_is_worse_than_wrong() -> None:
    """The hint is the whole reason the check is not merely advisory: a NaN makes
    the comparator non-transitive, which is undefined behaviour in the native
    sort rather than a wrong ordering."""
    with pytest.raises(TfidfStabilityError, match="undefined behaviour for the underlying sort"):
        check_finite([float("nan")], "scores")


def test_a_non_numeric_value_is_not_laundered_into_a_package_error() -> None:
    """`math.isfinite` raises `TypeError` and the validator does not catch it.
    That is right: a string in a score array is a programming error upstream, not
    a corpus this package should describe as unrankable."""
    with pytest.raises(TypeError, match="must be real number"):
        check_finite(["x"], "scores")  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# check_non_negative -- and where it deliberately differs from check_finite
# ---------------------------------------------------------------------------
def test_negative_zero_is_not_a_negative_weight() -> None:
    """`-0.0 < 0.0` is False, so it passes. Correct: it is zero, and a zero
    weight is what a term absent from a document has. Rejecting it would refuse
    an ordinary sparse vector that happened to be built by subtraction."""
    assert check_non_negative([-0.0], "weights") is None


def test_a_nan_weight_passes_the_sign_check_and_is_check_finites_business() -> None:
    """Every comparison with NaN is False, so `v < 0.0` cannot catch one. The two
    validators divide the work: this one owns the orthant, `check_finite` owns
    representability. A caller wanting both must call both.
    """
    assert check_non_negative([float("nan")], "weights") is None

    with pytest.raises(TfidfStabilityError, match="not finite"):
        check_finite([float("nan")], "weights")


def test_the_smallest_representable_negative_is_still_negative() -> None:
    """The check is a sign test, not a magnitude test, so a subnormal a hair
    below zero is refused exactly like -1.0."""
    with pytest.raises(TfidfStabilityError, match=r"weights\[0\] = -5e-324 is negative"):
        check_non_negative([-5e-324], "weights")


@pytest.mark.parametrize("value", [0.0, 5e-324, 1.0, sys.float_info.max, float("inf")])
def test_a_non_negative_value_passes_however_large(value: float) -> None:
    """Including `+inf`: unbounded above is not this check's concern, and saying
    so keeps the two validators' responsibilities separate."""
    assert check_non_negative([value], "weights") is None


def test_a_negative_weight_is_named_with_its_position_and_its_consequence() -> None:
    """The message cites section 2.2 because the reader's next question is
    whether a negative weight is merely unusual: it voids `cos in [0, 1]`."""
    with pytest.raises(TfidfStabilityError, match=r"weights\[1\] = -1\.0 is negative"):
        check_non_negative([0.5, -1.0, 2.0], "weights")

    with pytest.raises(TfidfStabilityError, match=r"cos in \[0, 1\] depends on it"):
        check_non_negative([-1.0], "weights")


# ---------------------------------------------------------------------------
# check_unique_ids
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ids", [[], ["a"], ["a", "b", "c"]])
def test_identifiers_that_are_already_distinct_pass(ids: list[str]) -> None:
    assert check_unique_ids(ids) is None


def test_a_repeat_names_both_positions_it_appeared_at() -> None:
    """Both, not just the second: on a corpus of thousands the first occurrence
    is the one the reader has to go and find."""
    with pytest.raises(DuplicateIdentifierError, match="'a' appears at positions 0 and 1"):
        check_unique_ids(["a", "a"])

    with pytest.raises(DuplicateIdentifierError, match="'a' appears at positions 0 and 2"):
        check_unique_ids(["a", "b", "a"])


def test_the_refusal_says_why_duplicate_identifiers_are_fatal() -> None:
    """The final tie-break key is the identifier; with a duplicate the comparator
    stops being a strict total order and the result falls to the sort."""
    with pytest.raises(DuplicateIdentifierError, match="requires unique identifiers"):
        check_unique_ids(["a", "a"])


def test_identifiers_that_merely_look_alike_are_distinct() -> None:
    """Equality, not rendering. `1` and `"1"` print the same and are two
    documents; collapsing them would silently drop one from the corpus."""
    assert check_unique_ids([1, "1"]) is None


@pytest.mark.parametrize(("pair", "shown"), [([1, True], "True"), ([0, False], "False")])
def test_a_boolean_collides_with_the_integer_it_equals(pair: list[object], shown: str) -> None:
    """`bool` is an `int`, so `1 == True` and they share a hash. Pinned because
    a JSON corpus with `"doc_id": 1` and `"doc_id": true` is a real shape, and
    the collision -- not a silent overwrite -- is the behaviour that catches it.
    """
    with pytest.raises(DuplicateIdentifierError, match=f"identifier {shown} appears"):
        check_unique_ids(pair)


def test_two_distinct_nan_identifiers_do_not_collide() -> None:
    """NaN is not equal to itself and, since 3.10, distinct NaN objects hash
    differently. So two of them are two identifiers. Degenerate, but it is what
    the dict does, and the ranking would then be decided by a key that compares
    equal to nothing -- caught downstream by `check_finite`, not here.
    """
    first, second = float("nan"), float("nan")
    assert first is not second
    assert check_unique_ids([first, second]) is None


def test_the_same_nan_object_twice_does_collide() -> None:
    """The dict's identity shortcut fires before equality, so one object used
    twice is a duplicate even though `x == x` is False. The contrast with the
    test above is the whole point: identity, not equality, decides."""
    same = float("nan")
    with pytest.raises(DuplicateIdentifierError, match="appears at positions 0 and 1"):
        check_unique_ids([same, same])


def test_an_unhashable_identifier_is_not_laundered_into_a_package_error() -> None:
    """A list cannot be a dict key. Like the `check_finite` string case, this is
    a programming error upstream rather than a corpus shape worth naming."""
    with pytest.raises(TypeError, match="unhashable type"):
        check_unique_ids(["a", ["b"]])


def test_identifiers_are_checked_in_order_so_the_earliest_repeat_wins() -> None:
    """Two independent duplicates in one corpus: the reported pair is the first
    to complete, which keeps the message deterministic under a reordering that
    does not change the set of duplicates."""
    with pytest.raises(DuplicateIdentifierError, match="'b' appears at positions 1 and 2"):
        check_unique_ids(["a", "b", "b", "a"])


def test_the_math_module_is_what_decides_finiteness() -> None:
    """A guard rewritten as `v != v or abs(v) == inf` would agree on every value
    in this file and disagree with `math.isfinite` on a Decimal or a Fraction.
    Stated so the dependency is deliberate rather than incidental."""
    for value in [*_FINITE_EXTREMES, float("nan"), float("inf")]:
        expected = math.isfinite(value)
        try:
            check_finite([value], "v")
            actual = True
        except TfidfStabilityError:
            actual = False
        assert actual == expected, f"{value!r} disagreed with math.isfinite"
