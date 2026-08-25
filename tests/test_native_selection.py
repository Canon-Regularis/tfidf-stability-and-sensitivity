"""Backend selection, the ABI guard, and the floating-point self-test.

``src/tfidf_stability/_native/__init__.py`` is this project's own Python -- 44
statements deciding which backend a process uses and whether the environment it
runs in can be trusted -- and it was excluded from coverage by ``omit =
["*/_native/*"]``. That pattern was aimed at the compiled extension and caught
the hand-written module beside it, so the repository reported 100% while this
file sat at 69%.

What the exclusion hid is the point. :func:`check_float_environment` exists
because several BLAS builds set flush-to-zero and denormals-are-zero
process-wide when loaded, which would flush the subnormal near-tie margins this
project is a study of. The test suite imports scikit-learn, so that hazard is
live here. The function was never called from ``src/``, ``tests/`` or
``scripts/``, is absent from ``__all__``, and had no test.

The two module-level arms that select the backend cannot be reached from a
process that already imported it, so they are exercised the way
``test_validation_contracts.py`` reaches the same module: by substituting the
resolved state rather than by re-importing.
"""

from __future__ import annotations

import warnings

import pytest

from tfidf_stability import _native
from tfidf_stability._native import (
    REQUIRED_ABI,
    build_info,
    native_available,
    unavailable_reason,
)
from tfidf_stability.utils.logging import EventKind, capture


# ---------------------------------------------------------------------------
# Which backend was selected, and whether the record says so
# ---------------------------------------------------------------------------
def test_the_selected_backend_is_recorded_as_native_when_one_loaded() -> None:
    """The arm that runs on a machine with a build. `log_backend_selection` is
    called by the CLI once logging is configured, because selection happens at
    import -- before any application could have installed a handler -- so a
    record emitted there would reach nobody."""
    if not native_available():
        pytest.skip(unavailable_reason() or "no native backend")

    with capture() as recorder:
        _native.log_backend_selection()

    selected = recorder.of_kind(EventKind.BACKEND_SELECTED)
    assert len(selected) == 1
    assert dict(selected[0].fields) == {"backend": "native", "abi": REQUIRED_ABI}


def test_the_reference_backend_is_recorded_with_the_reason_it_was_chosen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other arm, and the one a compiler-free install always takes.

    Reached by substituting the resolved module state, as
    `test_validation_contracts.py` does: the selection itself happens once at
    import and cannot be replayed in a process that has already done it.

    The reason travels with the record. "reference" alone would say which
    backend ran but not whether that was a choice or a failure, and those are
    different findings when a published number is being explained.
    """
    monkeypatch.setattr(_native, "_MODULE", None)
    monkeypatch.setattr(_native, "_REASON", "the extension was never compiled")

    with capture() as recorder:
        _native.log_backend_selection()

    selected = recorder.of_kind(EventKind.BACKEND_SELECTED)
    assert len(selected) == 1
    assert dict(selected[0].fields) == {
        "backend": "reference",
        "reason": "the extension was never compiled",
    }


def test_availability_and_its_reason_are_complementary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exactly one of them is informative at a time: a reason when unavailable,
    `None` when available. A reason left set alongside a working backend would
    put a stale explanation into every manifest.

    `_REASON` is set to a stale value *before* the backend is declared
    available, which is the point of the test and was the omission. Setting only
    `_MODULE` left `_REASON` at whatever this machine's import had produced, so
    the assertion asked about a different state depending on where it ran:
    `None` where the extension had been built, a full "could not be imported"
    string where it had not. It passed all nine native legs of the CI matrix and
    failed all nine reference legs, for eight consecutive pushes.

    Reproduced before the fix on a machine with the extension present, by
    setting `_REASON` by hand: `native_available()` returned True beside that
    explanation.
    """
    monkeypatch.setattr(_native, "_REASON", "a reason left over from an earlier attempt")
    monkeypatch.setattr(_native, "_MODULE", object())
    assert _native.native_available() is True
    assert _native.unavailable_reason() is None, (
        "a backend reporting itself available must report no reason, whatever the "
        "import left behind"
    )

    monkeypatch.setattr(_native, "_MODULE", None)
    monkeypatch.setattr(_native, "_REASON", "no build here")
    assert _native.native_available() is False
    assert _native.unavailable_reason() == "no build here"


# ---------------------------------------------------------------------------
# The floating-point self-test
# ---------------------------------------------------------------------------
def test_the_float_environment_is_trustworthy_in_this_process() -> None:
    """`check_float_environment` returns the self-test bitmask; zero means the
    environment is sound.

    This is the guard against the failure this whole project is about. Several
    BLAS builds set flush-to-zero and denormals-are-zero process-wide when they
    load, and under those flags a subnormal margin -- the near-tie case section
    7.4 exists to measure -- silently becomes zero. The suite imports
    scikit-learn, so a BLAS is loaded in this very process.

    Nothing in `src/`, `tests/` or `scripts/` calls this function, which is the
    finding: the guard exists and is never invoked.
    """
    if not native_available():
        pytest.skip(unavailable_reason() or "no native backend")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        flags = _native.check_float_environment()

    assert flags == 0, (
        f"the floating-point environment is not trustworthy: {flags:#x}. "
        f"Subnormals would flush, and the near-tie margins under study with them."
    )


def test_a_subnormal_survives_arithmetic_in_this_process() -> None:
    """The same claim without the native backend: an independent check that
    flush-to-zero is off, so the test above is asserting something true rather
    than a self-test that agrees with itself."""
    import sys

    tiny = sys.float_info.min
    assert tiny / 2.0 > 0.0, "flush-to-zero is on; every subnormal margin is a lie"
    assert tiny / 2.0 < tiny, "and the halved value really is subnormal"


def test_the_self_test_can_report_and_repair_a_bad_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The arm no sound machine reaches. A non-zero mask must warn rather than
    raise -- the run continues, but its provenance records that the numbers were
    produced under an untrustworthy environment.

    `restore=True` repairs first and re-reads, so a BLAS that flipped the flags
    at load does not condemn the whole run; the warning fires only if the repair
    did not take.
    """
    if not native_available():
        pytest.skip(unavailable_reason() or "no native backend")

    calls: list[str] = []

    class _Stub:
        def fp_selftest(self) -> int:
            calls.append("selftest")
            return 0 if "restore" in calls else 0b101

        def fp_restore_subnormals(self) -> None:
            calls.append("restore")

        def fp_describe(self, flags: int) -> str:
            return f"flags={flags:#b}"

    monkeypatch.setattr(_native, "require_native", _Stub)

    assert _native.check_float_environment(restore=True) == 0
    assert calls == ["selftest", "restore", "selftest"], "repaired, then re-read"


def test_an_unrepairable_environment_warns_rather_than_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the repair does not take, the mask is returned and a RuntimeWarning
    names what is wrong. Returning it rather than raising is deliberate: the
    caller decides whether to proceed, and the manifest records the state."""

    class _Stub:
        def fp_selftest(self) -> int:
            return 0b11

        def fp_restore_subnormals(self) -> None:
            return None

        def fp_describe(self, flags: int) -> str:
            return "subnormals are flushed"

    monkeypatch.setattr(_native, "require_native", _Stub)

    with pytest.warns(RuntimeWarning, match="not trustworthy"):
        flags = _native.check_float_environment(restore=True)

    assert flags == 0b11, "the mask is reported, not swallowed"


def test_the_self_test_can_be_asked_not_to_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`restore=False` observes without touching the process, which is what a
    diagnostic wants: repairing would hide the state it came to measure."""
    calls: list[str] = []

    class _Stub:
        def fp_selftest(self) -> int:
            calls.append("selftest")
            return 0b1

        def fp_restore_subnormals(self) -> None:  # pragma: no cover - must not run
            calls.append("restore")

        def fp_describe(self, flags: int) -> str:
            return "flushed"

    monkeypatch.setattr(_native, "require_native", _Stub)

    with pytest.warns(RuntimeWarning, match="not trustworthy"):
        _native.check_float_environment(restore=False)

    assert calls == ["selftest"], "observed once, and nothing was repaired"


# ---------------------------------------------------------------------------
# The build record
# ---------------------------------------------------------------------------
def test_the_build_record_names_what_produced_the_extension() -> None:
    """`build_info` goes into every run manifest, so a surprising number can be
    traced to the compiler and flags that produced it."""
    if not native_available():
        pytest.skip(unavailable_reason() or "no native backend")

    info = build_info()

    assert {"compiler_id", "compiler_ver", "build_type"} <= set(info)
    assert info["compiler_id"], "an unnamed compiler explains nothing"
