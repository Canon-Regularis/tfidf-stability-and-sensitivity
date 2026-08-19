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
    if _abi != REQUIRED_ABI:
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
    """Why the native backend is unavailable, or ``None`` if it is available."""
    return _REASON


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
