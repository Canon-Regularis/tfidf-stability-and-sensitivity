"""The run manifest: the reproducibility contract.

README section 3 promises that "all stages of the pipeline are deterministic
given a fixed corpus, configuration, and software environment (including library
versions)". The manifest makes that checkable: it records everything the numbers
depend on, so a rerun is verified rather than trusted.

Inclusion follows one test: could changing this move a published number? If yes
it goes in, however tedious. Hence the compiler's floating-point flags, the
reduction policy, the stopword list's digest and the ``tau`` in force, none of
which look like data.

Anything that varies between identical runs (timestamps, paths, hostnames) is
written for a human and stripped before the digest is taken
(:func:`~tfidf_stability.utils.io.strip_volatile`), so two identical runs on
different machines produce the same :meth:`RunManifest.digest`.
"""

from __future__ import annotations

import platform
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from tfidf_stability.utils.hashing import hash_text
from tfidf_stability.utils.io import canonical_json, strip_volatile, write_json
from tfidf_stability.utils.numerics import float_environment

__all__ = ["RunManifest", "environment_block", "is_reproducible_environment"]


def is_reproducible_environment(environment: Mapping[str, Any]) -> bool:
    """Whether an environment block describes a build that can publish numbers.

    An absent or null ``native`` is the pure-Python reference, reproducible on
    its own. Refusing it would refuse every result from a machine without a
    compiler. Any other ``native`` must be a mapping whose ``reproducible`` is
    true; a non-mapping gives ``False``, not ``AttributeError``.
    """
    native = environment.get("native")
    if native is None:
        return True
    if not isinstance(native, Mapping):
        return False
    return bool(native.get("reproducible", False))


def environment_block() -> dict[str, Any]:
    """Interpreter, platform and native-build provenance.

    ``native`` is always present: the build info when the compiled backend
    loaded, ``None`` otherwise. Its ``reproducible`` flag is how a reader checks
    that the build was free of fast-math and architecture tuning, either of
    which can move the numbers.
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
    """Everything a published number depends on.

    Attributes:
        run_kind: What produced this: ``"stability_profile"``,
            ``"tie_break_ablation"``, ``"noise_floor"`` and so on.
        config: The resolved configuration, after defaults and overrides.
        dataset: Dataset name, source and digests.
        preprocessing: The preprocessing map's fingerprint, including the
            stopword list's digest; editing that file moves the vocabulary and
            every number below it.
        model: Vocabulary and model digests from
            :func:`~tfidf_stability.persistence.save_load.save_model`.
        queries: The query-set provenance, including G19's candidate spread.
        parameters: Experiment parameters: ``tau``, the ``k`` set, the reduction
            policy, the operator priorities.
        results: Digests of the output artefacts.
        notes: Free text. Never hashed.
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
        """The full manifest, including the volatile parts."""
        return {
            "run_kind": self.run_kind,
            "config": self.config,
            "dataset": self.dataset,
            "preprocessing": self.preprocessing,
            "model": self.model,
            "queries": self.queries,
            "parameters": self.parameters,
            "results": self.results,
            "environment": self.environment,
            "notes": self.notes,
        }

    #: Environment keys identifying the machine, not the arithmetic. Stripped
    #: for the digest, kept in the written JSON, so
    #: ``.github/workflows/determinism.yml`` gets one pipeline digest across
    #: runners while ``float`` and ``native`` still move it. ``implementation``
    #: is deliberately absent: CPython and PyPy are not claimed to agree.
    _MACHINE_KEYS: ClassVar[tuple[str, ...]] = ("platform", "machine", "python")

    def digest(self) -> str:
        """SHA-256 over the manifest with volatile fields stripped.

        Two identical runs on different machines at different times give the
        same digest. :data:`_MACHINE_KEYS` is stripped alongside
        :data:`~tfidf_stability.utils.io.VOLATILE_KEYS`, and ``notes`` is
        excluded. ``environment["platform"]`` is ``platform.platform()``, which
        carries the OS build number. Without the strip, the digest would move on
        an operating system update.

        Stripped here rather than added to ``VOLATILE_KEYS`` because that set is
        shared with every other report this package hashes, and a machine
        identity is only meaningless to *this* digest.

        Rendered by ``canonical_json``, the same renderer :meth:`write` uses.
        ``hash_json`` would emit the non-standard ``NaN``/``Infinity`` tokens
        where the file holds ``null``, so a manifest with a non-finite margin
        would carry a digest over bytes the file does not contain. The two
        renderers agree wherever every value is finite.
        """
        payload = strip_volatile(self.to_dict(), extra=self._MACHINE_KEYS)
        payload.pop("notes", None)
        return hash_text(canonical_json(payload, indent=None))

    @property
    def is_reproducible_build(self) -> bool:
        """Whether the native backend, if present, was built for reproducibility.

        ``True`` when no native backend is loaded: the pure-Python reference is
        reproducible on its own.

        Delegates to ``is_reproducible_environment`` so ``cmd_verify`` answers
        one block the same way.
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
        # Two failures get two messages: a `native` that is not a mapping, and a
        # build whose flags say it is not reproducible. One message for both
        # would call `.get` on a non-mapping, raising `AttributeError`, and would
        # name `fast_math=None` as the cause when the flags are what disagree.
        native = self.environment.get("native")
        if not isinstance(native, Mapping):
            raise RuntimeError(
                f"this manifest's native block is {type(native).__name__}, not a "
                f"mapping, so the build cannot be shown to be reproducible"
            )
        raise RuntimeError(
            "this build is not reproducible "
            f"(fast_math={native.get('fast_math')}, arch_tune={native.get('arch_tune')}); "
            "rebuild without TFIDF_FAST_MATH or TFIDF_ARCH_TUNE before producing results"
        )
