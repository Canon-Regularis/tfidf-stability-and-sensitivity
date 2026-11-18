"""Backend selection, the ABI guard, and the floating-point self-test.

The selection arms run once at import, so tests substitute the resolved module
state instead of re-importing. :func:`check_float_environment` is called here
in-process. Rationale: docs/engineering_notes.md#native-package-coverage
"""

from __future__ import annotations

import os
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
    """A loaded native backend is recorded with its ABI tag.

    `log_backend_selection` is called by the CLI once logging is configured,
    because selection happens at import, before any handler is installed.
    """
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
    """The reference backend is recorded with the reason it was chosen.

    The resolved module state is substituted, as `test_validation_contracts.py`
    does, because selection runs once at import. The reason distinguishes a
    choice from a failure when a published number is being explained.
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
    """Exactly one of availability and its reason is informative at a time.

    `_REASON` is set to a stale value before `_MODULE` is substituted, so the
    assertion does not depend on what this machine's import left behind. A
    reason left set beside a working backend would reach every manifest.
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


def test_an_extension_built_for_this_interpreter_must_actually_have_loaded() -> None:
    """An extension tagged for this interpreter must have been selected.

    Native suites skip when `native_available()` is False, so an extension
    that is present but unloadable would skip them all while the run stays
    green. Matching on `EXTENSION_SUFFIXES` ignores builds for other Pythons.
    """
    import importlib.machinery
    from pathlib import Path

    beside_the_package = Path(_native.__file__).parent
    built = [
        candidate
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
        for candidate in beside_the_package.glob(f"_tfidf_native{suffix}")
    ]
    if not built:
        pytest.skip("no extension built for this interpreter; there is nothing to assert")

    assert native_available(), (
        f"{built[0].name} is present and is tagged for this interpreter, so the "
        f"native backend must have been selected. It was not, and the reason "
        f"given is: {unavailable_reason()}"
    )


def test_a_backend_that_loaded_records_no_abi_mismatch() -> None:
    """`_ABI_MISMATCH` is False after a successful load.

    Only the mismatch arm sets it, so a True initialiser would make
    `require_native()` without a build raise `AbiVersionMismatchError`, not
    `NativeBackendUnavailableError`. The arm is unreachable in-process.
    """
    if not native_available():
        pytest.skip(unavailable_reason() or "no native backend")

    assert _native._ABI_MISMATCH is False, (
        "the extension loaded, so nothing may still be claiming an ABI mismatch"
    )


# ---------------------------------------------------------------------------
# The floating-point self-test
# ---------------------------------------------------------------------------
def test_the_float_environment_is_trustworthy_in_this_process() -> None:
    """`check_float_environment` returns a zero mask in a sound process.

    Some BLAS builds set flush-to-zero and denormals-are-zero process-wide
    when loaded, which would flush the subnormal near-tie margins under study.
    Rationale: docs/engineering_notes.md#float-environment-self-test
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
    """A subnormal survives arithmetic without the native backend.

    An independent check that flush-to-zero is off, so the self-test above
    asserts something true rather than agreeing with itself.
    """
    import sys

    tiny = sys.float_info.min
    assert tiny / 2.0 > 0.0, "flush-to-zero is on; every subnormal margin is a lie"
    assert tiny / 2.0 < tiny, "and the halved value really is subnormal"


def test_the_self_test_can_report_and_repair_a_bad_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-zero mask is repaired and re-read before any warning.

    `restore=True` repairs first, so a BLAS that flipped the flags at load
    does not condemn the run; the warning fires only if the repair fails.
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
    """A failed repair returns the mask and emits a RuntimeWarning.

    The caller decides whether to proceed, and the manifest records the state.
    """

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
    """`restore=False` observes the process without repairing it.

    A diagnostic needs the unrepaired state, so nothing is touched.
    """
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


def test_the_default_is_to_repair_rather_than_merely_observe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default of `restore` repairs and re-reads.

    The two tests above pass the argument explicitly, so the signature's own
    default is under test only here. Repairing keeps an unattended run safe
    when a BLAS flips the flags at import.
    """
    calls: list[str] = []

    class _Stub:
        def fp_selftest(self) -> int:
            calls.append("selftest")
            return 0 if "restore" in calls else 0b101

        def fp_restore_subnormals(self) -> None:
            calls.append("restore")

        def fp_describe(self, flags: int) -> str:  # pragma: no cover - repair succeeds
            return f"flags={flags:#b}"

    monkeypatch.setattr(_native, "require_native", _Stub)

    assert _native.check_float_environment() == 0, "called with no arguments at all"
    assert calls == ["selftest", "restore", "selftest"], (
        "the default must repair and re-read; observing without repairing would "
        "leave a flushing environment in place for the rest of the run"
    )


def test_the_warning_is_attributed_to_the_caller_not_to_this_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The warning is raised with `stacklevel=2`, naming the caller.

    At `stacklevel=1` every warning would name `_native/__init__.py`, hiding
    which code ran under a flushing environment. `pytest.warns` does not check
    the filename; `simplefilter` lifts the project-wide error filter.
    """
    import warnings

    class _Stub:
        def fp_selftest(self) -> int:
            return 0b11

        def fp_restore_subnormals(self) -> None:
            return None

        def fp_describe(self, flags: int) -> str:
            return "subnormals are flushed"

    monkeypatch.setattr(_native, "require_native", _Stub)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _native.check_float_environment(restore=False)

    assert len(caught) == 1, "the premise: exactly one warning to attribute"
    # `normcase` rather than `==`: Windows paths are case-insensitive, and the
    # drive letter arrives as `C:` or `c:` depending on how the working
    # directory was spelled at interpreter start. The spelling is a property of
    # the shell, not of the warning's attribution.
    assert os.path.normcase(caught[0].filename) == os.path.normcase(__file__), (
        f"the warning must point at the code that asked, not at the module that "
        f"raised it; it named {caught[0].filename}"
    )


# ---------------------------------------------------------------------------
# The build record
# ---------------------------------------------------------------------------
def test_the_build_record_names_what_produced_the_extension() -> None:
    """`build_info` names the compiler and flags that produced the extension.

    The record goes into every run manifest, so a surprising number can be
    traced to its build.
    """
    if not native_available():
        pytest.skip(unavailable_reason() or "no native backend")

    info = build_info()

    assert {"compiler_id", "compiler_ver", "build_type"} <= set(info)
    assert info["compiler_id"], "an unnamed compiler explains nothing"
