"""Loader for the compiled native backend.

The native extension is an **optional accelerator**. The normative implementation
is the pure-Python reference, which requires no compiler, so its absence is a
capability difference and never an error at import time. Callers that explicitly
ask for the native backend get a clear
:class:`~tfidf_stability.utils.validation.NativeBackendUnavailableError`
explaining what was tried.

The extension is also checked for numerical trustworthiness on first load: if the
process's floating-point environment has been altered -- most commonly by a BLAS
setting flush-to-zero when numpy imports -- that is surfaced immediately rather
than silently corrupting the subnormal score differences this project studies.
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

#: Bumped whenever the C++ <-> Python contract changes shape -- a symbol added,
#: removed, or changed in signature. A mismatch means a stale build is present
#: and must be recompiled.
#:
#: This is the extension's ``__abi__``, not its ``__version__``. The two were the
#: same string until the ordering distances were mirrored, which added bindings
#: without releasing anything: tying the check to the release version would have
#: let a .pyd predating those symbols load and the differential suite skip
#: silently, which is precisely the failure this constant exists to catch. A
#: build too old to carry ``__abi__`` at all reports ``None`` and is rejected on
#: the same path.
REQUIRED_ABI = "0.3.0"

if TYPE_CHECKING:  # pragma: no cover
    _tfidf_native: Any

_MODULE: Any | None = None
_REASON: str | None = None

# Imported dynamically rather than with a plain `from ... import`. The extension
# may or may not be present depending on whether the native backend has been
# built, and a static import would make type checking depend on build state:
# flagged as a missing attribute on a clean checkout, and as a redundant
# `type: ignore` once compiled.
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
        _REASON = (
            f"the compiled extension reports ABI {_abi!r} but this Python "
            f"package expects {REQUIRED_ABI!r}. Rebuild the native backend."
        )
    else:
        _MODULE = _loaded


def log_backend_selection() -> None:
    """Record which backend this process selected, and why.

    Deliberately not emitted at import. The selection is made while this module
    is being imported, which is necessarily before any application has had the
    chance to call :func:`~tfidf_stability.utils.logging.configure`, so an
    import-time record would be addressed to nobody. The caller emits it once
    logging is up.
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
        from tfidf_stability.utils.validation import NativeBackendUnavailableError

        raise NativeBackendUnavailableError(_REASON or "the native backend is unavailable")
    return _MODULE


def build_info() -> dict[str, object]:
    """Compiler, flags and revision of the loaded extension, for the manifest."""
    info: dict[str, object] = dict(require_native().build_info())
    return info


def check_float_environment(*, restore: bool = True) -> int:
    """Verify the process's floating-point environment, optionally repairing it.

    Returns the self-test bitmask; ``0`` means the environment is trustworthy.

    The failure this exists for is real rather than theoretical: several BLAS
    builds set flush-to-zero and denormals-are-zero process-wide when they load.
    In a project whose subject is near-tie margins, silently flushing subnormal
    score differences to zero would destroy exactly the phenomenon under study,
    so it is detected loudly and, by default, repaired.
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
