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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from tfidf_stability.utils.hashing import hash_json
from tfidf_stability.utils.io import strip_volatile, write_json
from tfidf_stability.utils.numerics import float_environment

__all__ = ["RunManifest", "environment_block"]


def environment_block() -> dict[str, Any]:
    """Interpreter, platform and native-build provenance.

    The native block appears only when the compiled backend loaded. Its
    ``reproducible`` flag is how a reader checks that the build was free of
    fast-math and architecture tuning, either of which can move the numbers.
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

    #: Environment keys that identify the *machine* rather than the arithmetic.
    #: Stripped for the digest and kept in the written JSON, exactly as
    #: ``hostname`` and ``cwd`` are, and for the same reason.
    #:
    #: They belong here by this module's own inclusion rule -- "could changing
    #: this move a published number?" -- and the repository answers no in the
    #: strongest available form: ``.github/workflows/determinism.yml`` requires
    #: one identical pipeline digest across 18 legs, 3 operating systems by 3
    #: interpreters plus 3 operating systems by 3 build types. G13's
    #: correctly-rounded logarithm is what buys that, replacing the platform
    #: libm precisely so ``idf`` does not vary with it.
    #:
    #: ``implementation`` is deliberately absent: CPython and PyPy are not
    #: claimed to agree, and nothing in CI asserts they do. ``float`` and
    #: ``native`` stay covered too, so a fast-math or arch-tuned build still
    #: moves the digest -- those change the numbers, which is the whole test.
    _MACHINE_KEYS: ClassVar[tuple[str, ...]] = ("platform", "machine", "python")

    def digest(self) -> str:
        """SHA-256 over the manifest with volatile fields stripped.

        Two identical runs on different machines at different times give the same
        digest, which is what makes it a reproducibility check rather than a
        record. The environment block is covered, so a different compiler or
        reduction policy changes the digest; both change the numbers. ``notes``
        are excluded.

        :data:`_MACHINE_KEYS` is stripped alongside
        :data:`~tfidf_stability.utils.io.VOLATILE_KEYS`, which the sentence above
        requires and the code did not do. ``environment["platform"]`` is
        ``platform.platform()``, carrying the OS build number -- ``10.0.26200``
        on the machine this was found on -- so the digest changed on an operating
        system update, and no two machines ever agreed. Measured: two manifests
        for the same run, one with a Linux environment block substituted, gave
        ``1f223e30...`` and ``c88f892f...`` while their ``float`` and ``native``
        blocks were identical.

        Stripped here rather than added to ``VOLATILE_KEYS`` because that set is
        shared with every other report this package hashes, and a machine
        identity is only meaningless to *this* digest.

        The sibling identity in this package had it right already:
        :meth:`~tfidf_stability.analysis.summarise.ExperimentResult.digest`
        hashes payload and parameters and omits the environment block outright,
        which is why no published ``result_digest`` moves with this change.
        """
        payload = strip_volatile(self.to_dict(), extra=self._MACHINE_KEYS)
        payload.pop("notes", None)
        return hash_json(payload)

    @property
    def is_reproducible_build(self) -> bool:
        """Whether the native backend, if present, was built for reproducibility.

        ``True`` when no native backend is loaded: the pure-Python reference is
        reproducible on its own.
        """
        native = self.environment.get("native")
        if native is None:
            return True
        return bool(native.get("reproducible", False))

    def write(self, path: Path | str) -> dict[str, Any]:
        """Write the manifest as canonical JSON, with its own digest embedded."""
        payload = self.to_dict()
        payload["manifest_digest"] = self.digest()
        write_json(path, payload)
        return payload

    def require_reproducible(self) -> None:
        """Refuse to proceed on a build that cannot produce publishable numbers."""
        if not self.is_reproducible_build:
            native = self.environment.get("native") or {}
            raise RuntimeError(
                "this build is not reproducible "
                f"(fast_math={native.get('fast_math')}, arch_tune={native.get('arch_tune')}); "
                "rebuild without TFIDF_FAST_MATH or TFIDF_ARCH_TUNE before producing results"
            )
