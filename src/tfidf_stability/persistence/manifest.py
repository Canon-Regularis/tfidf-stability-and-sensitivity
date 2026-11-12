"""The run manifest: what each published number depends on (README section 3).

Volatile fields are stripped (:func:`~tfidf_stability.utils.io.strip_volatile`)
so identical runs on different machines give the same :meth:`RunManifest.digest`.
Rationale: docs/engineering_notes.md#run-manifest-contents
"""

from __future__ import annotations

import platform
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar

from tfidf_stability.utils.hashing import hash_text
from tfidf_stability.utils.io import canonical_json, strip_volatile, write_json
from tfidf_stability.utils.numerics import float_environment
from tfidf_stability.utils.validation import UnreproducibleBuildError

__all__ = ["RunManifest", "environment_block", "is_reproducible_environment"]


def is_reproducible_environment(environment: Mapping[str, Any]) -> bool:
    """Whether an environment block describes a build that can publish numbers.

    An absent or null ``native`` is the pure-Python reference and is reproducible.
    Any other ``native`` must be a mapping whose ``reproducible`` is true; a
    non-mapping gives ``False``, not ``AttributeError``.
    """
    native = environment.get("native")
    if native is None:
        return True
    if not isinstance(native, Mapping):
        return False
    return bool(native.get("reproducible", False))


def environment_block() -> dict[str, Any]:
    """Interpreter, platform and native-build provenance.

    ``native`` is the build info when the compiled backend loaded, else ``None``.
    Its ``reproducible`` flag records whether the build was free of fast-math and
    architecture tuning, either of which can move the numbers.
    """
    block: dict[str, Any] = {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "float": float_environment(),
    }
    try:
        from tfidf_stability._native import build_info, native_available

        if native_available():
            block["native"] = build_info()
        else:
            block["native"] = None
    except Exception:  # pragma: no cover - the loader is defensive already
        block["native"] = None
    return block


@dataclass(slots=True)
class RunManifest:
    """Everything a published number depends on; ``notes`` is never hashed.

    ``model`` holds digests from :func:`~tfidf_stability.persistence.save_load.save_model`;
    ``preprocessing`` a fingerprint including the stopword digest; ``queries`` G19's spread.
    ``environment`` is from :func:`environment_block`, hashed without :data:`_MACHINE_KEYS`.
    """

    run_kind: str
    config: dict[str, Any] = field(default_factory=dict)
    dataset: dict[str, Any] = field(default_factory=dict)
    preprocessing: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    queries: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=environment_block)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """The full manifest, including the volatile parts.

        Read from the declared fields, so every field reaches the written manifest
        and :meth:`digest`. Not ``dataclasses.asdict``, which deep-copies values
        shared with the caller and would copy the whole manifest on every digest.
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}

    #: Environment keys identifying the machine, not the arithmetic. Stripped for
    #: the digest, kept in the written JSON, so ``.github/workflows/determinism.yml``
    #: gets one pipeline digest across runners while ``float`` and ``native`` move it.
    #: ``implementation`` is absent: CPython and PyPy are not claimed to agree.
    _MACHINE_KEYS: ClassVar[tuple[str, ...]] = ("platform", "machine", "python")

    def digest(self) -> str:
        """SHA-256 over the manifest with volatile fields stripped.

        Strips :data:`_MACHINE_KEYS` and :data:`~tfidf_stability.utils.io.VOLATILE_KEYS`
        and drops ``notes``; renders through ``canonical_json``, as :meth:`write` does.
        Rationale: docs/engineering_notes.md#manifest-digest
        """
        payload = strip_volatile(self.to_dict(), extra=self._MACHINE_KEYS)
        payload.pop("notes", None)
        return hash_text(canonical_json(payload, indent=None))

    @property
    def is_reproducible_build(self) -> bool:
        """Whether the native backend, if present, was built for reproducibility.

        ``True`` when no native backend is loaded. Delegates to
        ``is_reproducible_environment`` so ``cmd_verify`` answers one block the same way.
        """
        return is_reproducible_environment(self.environment)

    def write(self, path: Path | str) -> dict[str, Any]:
        """Write the manifest as canonical JSON, with its own digest embedded."""
        payload = self.to_dict()
        payload["manifest_digest"] = self.digest()
        write_json(path, payload)
        return payload

    def require_reproducible(self) -> None:
        """Refuse to proceed on a build that cannot produce publishable numbers."""
        if self.is_reproducible_build:
            return
        # Every refusal opens with the same phrase so a caller can match on it.
        # A non-mapping `native` gets its own clause, since `.get` on it raises
        # `AttributeError`. `None` reaches here only under a patched property,
        # so it takes the flag branch and reports no flags.
        native = self.environment.get("native")
        if native is not None and not isinstance(native, Mapping):
            raise UnreproducibleBuildError(
                f"this build is not reproducible: the manifest's native block "
                f"is {type(native).__name__}, not a mapping"
            )
        flags: Mapping[str, Any] = native if isinstance(native, Mapping) else {}
        raise UnreproducibleBuildError(
            "this build is not reproducible "
            f"(fast_math={flags.get('fast_math')}, arch_tune={flags.get('arch_tune')}); "
            "rebuild without TFIDF_FAST_MATH or TFIDF_ARCH_TUNE before producing results"
        )
