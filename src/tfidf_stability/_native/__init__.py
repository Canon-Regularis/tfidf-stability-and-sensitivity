"""Loader for the compiled native backend.

The extension is an optional accelerator and the pure-Python reference needs no
compiler, so an absent extension is a capability difference and never an
import-time error. Asking for the native backend explicitly raises
:class:`~tfidf_stability.utils.validation.NativeBackendUnavailableError` saying
what was tried.

:func:`check_float_environment` probes the process floating-point environment
through the extension and repairs it: a BLAS that sets flush-to-zero as numpy
imports would silently zero the subnormal score differences under study.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from tfidf_stability.utils.logging import EventKind, get_logger, log_event

__all__ = [
    "REQUIRED_ABI",
    "build_info",
    "log_backend_selection",
    "native_available",
    "require_native",
    "unavailable_reason",
]

#: Bumped whenever the C++/Python contract changes shape: a symbol added,
#: removed, or changed in signature. A mismatch means a stale build.
#:
#: Read from the extension's ``__abi__`` rather than its ``__version__``: the
#: two agreed until mirroring the ordering distances added bindings without a
#: release, so keying off the release version would have let a .pyd predating
#: those symbols load and the differential suite skip silently. A build too old
#: to carry ``__abi__`` reports ``None`` and is rejected the same way.
REQUIRED_ABI = "0.4.0"

if TYPE_CHECKING:  # pragma: no cover
    _tfidf_native: Any

_MODULE: Any | None = None
_REASON: str | None = None
_ABI_MISMATCH: bool = False

# Dynamic import: the extension's existence depends on build state, and a static
# `from ... import` would make type checking depend on it too (missing attribute
# on a clean checkout, redundant `type: ignore` once compiled).
try:
    import importlib

    _loaded = importlib.import_module("tfidf_stability._native._tfidf_native")
except ImportError as exc:  # pragma: no cover - depends on whether a build exists
    _REASON = (
        f"the compiled extension could not be imported ({exc}). "
        f"Build it with `cmake --preset mingw && cmake --build --preset mingw`, "
        f"or `pip install -e .`. The pure-Python reference backend needs no compiler."
    )
else:
    _abi = getattr(_loaded, "__abi__", None)
    # pragma: no cover on the mismatch arm, for the same reason as the ImportError
    # arm above: selection happens once, at import, against whichever extension is
    # on disk. A process that has imported this module cannot replay it against a
    # differently-versioned build, so the branch is unreachable from a test rather
    # than merely untested. `test_validation_contracts.py` covers what the
    # mismatch *causes* -- AbiVersionMismatchError -- by substituting the resolved
    # state, which is the part a caller can observe.
    if _abi != REQUIRED_ABI:  # pragma: no cover - needs a stale build on disk
        _ABI_MISMATCH = True
        _REASON = (
            f"the compiled extension reports ABI {_abi!r} but this Python "
            f"package expects {REQUIRED_ABI!r}. Rebuild the native backend."
        )
    else:
        _MODULE = _loaded


def log_backend_selection() -> None:
    """Record which backend this process selected, and why.

    Selection happens during this module's import, before any application could
    have called :func:`~tfidf_stability.utils.logging.configure`, so a record
    emitted there would reach nobody. The caller emits it once logging is up.
    """
    log = get_logger(__name__)
    if _MODULE is None:
        log_event(log, EventKind.BACKEND_SELECTED, backend="reference", reason=_REASON)
    else:
        log_event(log, EventKind.BACKEND_SELECTED, backend="native", abi=REQUIRED_ABI)


def native_available() -> bool:
    """Whether the compiled backend is importable and version-compatible."""
    return _MODULE is not None


def unavailable_reason() -> str | None:
    """Why the native backend is unavailable, or ``None`` if it is available.

    Derived from :func:`native_available` rather than returned raw. The two
    globals are set on mutually exclusive paths during this module's import --
    ``_REASON`` on the two failure branches, ``_MODULE`` only in the ``else`` --
    so returning ``_REASON`` alone gave the documented answer for every state
    this module can actually reach, and the sentence above was true by
    construction rather than by the code.

    It stopped being true the moment anything set ``_MODULE`` without clearing
    ``_REASON``, which is what a test substituting a stub backend does. On a
    machine where the extension had been built ``_REASON`` was already ``None``
    and nothing showed; on one where it had not, the stale explanation came back
    beside a backend reporting itself available. That is the state
    ``tests/test_native_selection.py`` calls "a stale explanation in every
    manifest", and it failed all nine reference legs of the CI matrix while all
    nine native legs passed -- the answer depended on the runner, not the code.

    One line, and no behaviour changes for any state reachable by import alone.
    """
    return None if native_available() else _REASON


def require_native() -> Any:
    """Return the compiled module, raising a helpful error if it is missing."""
    if _MODULE is None:
        from tfidf_stability.utils.validation import (
            AbiVersionMismatchError,
            NativeBackendUnavailableError,
        )

        # A stale .pyd and a missing one need different fixes: rebuild against the
        # current contract, or build at all. AbiVersionMismatchError subclasses
        # NativeBackendUnavailableError, so callers that do not care are unaffected.
        error = AbiVersionMismatchError if _ABI_MISMATCH else NativeBackendUnavailableError
        raise error(_REASON or "the native backend is unavailable")
    return _MODULE


def build_info() -> dict[str, object]:
    """Compiler, flags and revision of the loaded extension, for the manifest."""
    info: dict[str, object] = dict(require_native().build_info())
    return info


def check_float_environment(*, restore: bool = True) -> int:
    """Verify the process's floating-point environment, optionally repairing it.

    Returns the self-test bitmask; ``0`` means the environment is trustworthy.
    Several BLAS builds set flush-to-zero and denormals-are-zero process-wide on
    load, which would flush the subnormal near-tie margins under study, so this
    warns and by default repairs.
    """
    mod = require_native()
    flags = int(mod.fp_selftest())
    if flags and restore:
        mod.fp_restore_subnormals()
        flags = int(mod.fp_selftest())
    if flags:
        warnings.warn(
            f"the floating-point environment is not trustworthy: {mod.fp_describe(flags)}",
            RuntimeWarning,
            stacklevel=2,
        )
    return flags
