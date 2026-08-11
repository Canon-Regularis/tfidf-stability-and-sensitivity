"""The run manifest: the reproducibility contract.

README section 3 promises that "all stages of the pipeline are deterministic
given a fixed corpus, configuration, and software environment (including library
versions)". A manifest is what turns that from a claim into something checkable:
it records *everything* on which the numbers depend, so a result can be traced
back to what produced it and a rerun can be verified rather than trusted.

What goes in is decided by one question -- **could changing this change a
published number?** If yes it belongs here, however tedious. That is why the
manifest carries the compiler's floating-point flags, the reduction policy, the
stopword list's digest and the exact ``tau``, none of which look like data.

What stays out is anything that varies between identical runs: timestamps,
paths, hostnames. Those are written for a human to read and stripped before the
digest is taken (:func:`~tfidf_stability.utils.io.strip_volatile`), so two
identical runs on different machines produce the same
:meth:`RunManifest.digest`.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tfidf_stability.utils.hashing import hash_json
from tfidf_stability.utils.io import strip_volatile, write_json
from tfidf_stability.utils.numerics import float_environment

__all__ = ["RunManifest", "environment_block"]


def environment_block() -> dict[str, Any]:
    """Interpreter, platform and native-build provenance.

    The native block is present only when the compiled backend loaded. Its
    ``reproducible`` flag is the one that matters: a build with fast-math or
    architecture tuning is explicitly *not* allowed to produce published
    numbers, and recording the flag is what lets a reader check that.
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
        run_kind: What produced this -- ``"stability_profile"``,
            ``"tie_break_ablation"``, ``"noise_floor"`` and so on.
        config: The resolved configuration, after defaults and overrides.
        dataset: Dataset name, source and digests.
        preprocessing: The preprocessing map's fingerprint, including the
            stopword list's digest -- editing that file changes the vocabulary
            and therefore every number.
        model: Vocabulary and model digests from
            :func:`~tfidf_stability.persistence.save_load.save_model`.
        queries: The query-set provenance, including G19's candidate spread.
        parameters: Experiment parameters -- ``tau``, the ``k`` set, the
            reduction policy, the operator priorities.
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

    def digest(self) -> str:
        """SHA-256 over the manifest with volatile fields stripped.

        Two identical runs on different machines at different times must produce
        the same digest -- otherwise it could not be used to *check*
        reproducibility, only to record a run.

        Note what this does and does not cover. It includes the environment
        block, so a different compiler or a different reduction policy gives a
        different digest, which is intended: those change the numbers. It
        excludes ``notes``, which do not.
        """
        payload = strip_volatile(self.to_dict())
        payload.pop("notes", None)
        return hash_json(payload)

    @property
    def is_reproducible_build(self) -> bool:
        """Whether the native backend, if present, was built for reproducibility.

        ``True`` when there is no native backend at all: the pure-Python
        reference is the normative implementation and is reproducible by
        construction.
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
