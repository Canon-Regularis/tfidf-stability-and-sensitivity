"""Load every dataset by name and record where each one came from.

:data:`DATASET_NAMES` names the loaders; ``configs/datasets.yaml`` mirrors it and nothing loads it.
:func:`load_dataset` returns a :class:`LoadedDataset` whose :attr:`~LoadedDataset.provenance` goes
into the manifest. See ``data/README.md``. Rationale: docs/engineering_notes.md#dataset-provenance
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

from tfidf_stability.datasets import movielens, synthetic
from tfidf_stability.utils.hashing import hash_text
from tfidf_stability.utils.io import canonical_json, read_jsonl
from tfidf_stability.utils.validation import DataIntegrityError

__all__ = [
    "DATASET_NAMES",
    "LoadedDataset",
    "load_dataset",
    "load_jsonl_corpus",
]

#: Smaller synthetic spec for tests and notebooks, where a 2000-document corpus
#: would dominate the runtime.
_TINY = synthetic.SyntheticSpec(
    n_docs=120, vocab_size=300, n_users=20, n_exact_duplicates=4, n_twin_pairs=6
)


@dataclass(frozen=True, slots=True)
class LoadedDataset:
    """A corpus plus everything needed to say where it came from."""

    name: str
    records: list[dict[str, Any]]
    interactions: list[tuple[str, str, float]] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def n_documents(self) -> int:
        return len(self.records)

    @property
    def doc_ids(self) -> list[str]:
        return [str(r["doc_id"]) for r in self.records]

    def digest(self) -> str:
        """Identity of the loaded corpus, for the manifest.

        Over the records, not the provenance or the interaction set: two loads
        at different ``min_weight`` share it; :attr:`provenance` records the threshold.
        """
        return hash_text(canonical_json(self.records, indent=None))


def load_jsonl_corpus(path: Path | str) -> LoadedDataset:
    """Load a corpus from a JSONL file, as written by ``build_corpus.py``."""
    target = Path(path)
    if not target.exists():
        raise DataIntegrityError(f"corpus file not found: {target}")

    records = list(read_jsonl(target))
    if not records:
        raise DataIntegrityError(f"{target} contains no records")

    missing = [i for i, r in enumerate(records) if "doc_id" not in r or "text" not in r]
    if missing:
        raise DataIntegrityError(
            f"{target}: {len(missing)} record(s) lack 'doc_id' or 'text' "
            f"(first at line {missing[0] + 1})"
        )

    return LoadedDataset(
        name=f"jsonl:{target.name}",
        records=records,
        provenance={
            "kind": "jsonl",
            "path": str(target),
            # Newline-normalised by `hash_text`, matching `cmd_build_corpus`, so
            # a corpus another tool wrote with CRLF still has one identity.
            # `atomic_write_text` emits LF everywhere, so a corpus this
            # repository wrote hashes to `sha256sum` of the file.
            "sha256": hash_text(target.read_text(encoding="utf-8")),
            "n_documents": len(records),
        },
    )


#: A registered loader: called with the name, ``archive`` and ``spec`` as given to
#: :func:`load_dataset`, and ignoring whichever of the two its source does not use.
_Loader = Callable[[str, Path | str | None, synthetic.SyntheticSpec | None], LoadedDataset]


@dataclass(frozen=True, slots=True)
class _Synthetic:
    """Generate a synthetic dataset from ``default``, or from ``spec`` when given.

    ``archive`` is ignored.
    """

    default: synthetic.SyntheticSpec

    def __call__(
        self, name: str, archive: Path | str | None, spec: synthetic.SyntheticSpec | None
    ) -> LoadedDataset:
        corpus = synthetic.generate(spec or self.default)
        return LoadedDataset(
            name=name,
            records=corpus.records(),
            interactions=list(corpus.interactions),
            provenance={
                "kind": "synthetic",
                # SyntheticSpec is slots=True, so there is no instance __dict__
                # to read.
                "spec": {
                    k: list(v) if isinstance(v, tuple) else v
                    for k, v in asdict(corpus.spec).items()
                },
                "spec_digest": corpus.spec.digest(),
                "n_documents": corpus.n_documents,
                "n_twin_pairs": len(corpus.twins),
                "n_exact_duplicate_pairs": len(corpus.exact_duplicate_pairs),
                # The corpus regenerates from its spec anywhere, so a result
                # built on it is reproducible from the repository alone.
                "redistributable": True,
            },
        )


def _load_movielens(
    name: str, archive: Path | str | None, spec: synthetic.SyntheticSpec | None
) -> LoadedDataset:
    """Parse ``ml-latest-small`` from the zip at ``archive``; ``spec`` is ignored."""
    if archive is None:
        raise DataIntegrityError(
            "movielens_small needs an archive path. MovieLens may not be "
            "redistributed, so it is not in the repository -- run "
            "'python scripts/fetch_data.py' first, then pass its path."
        )
    films = movielens.load(archive)
    return LoadedDataset(
        name=name,
        records=films.records(),
        interactions=list(films.interactions),
        provenance={
            "kind": "movielens",
            "variant": "ml-latest-small",
            "archive_sha256": films.archive_sha256,
            "n_documents": films.n_documents,
            "n_ratings": films.n_ratings,
            "n_users": films.n_users,
            "n_unrated_documents": films.n_unrated,
            # The threshold that decides which ratings become positive
            # interactions. Changing it moves every profile built from this
            # dataset, so the manifest records it.
            "interaction_min_weight": films.min_weight,
            # Recorded so the manifest shows at a glance that this result
            # cannot be reproduced from the repository alone.
            "redistributable": False,
        },
    )


#: Each registered name and its loader; :data:`DATASET_NAMES` is the keys, in this order.
_REGISTRY: Final[dict[str, _Loader]] = {
    "synthetic_small": _Synthetic(synthetic.SyntheticSpec()),
    "synthetic_tiny": _Synthetic(_TINY),
    "movielens_small": _load_movielens,
}

#: Registered names. ``jsonl:<path>`` is also accepted, for a corpus already on
#: disk, which is how a MovieLens-derived corpus is consumed once
#: ``scripts/build_corpus.py`` has written it out.
DATASET_NAMES: Final[tuple[str, ...]] = tuple(_REGISTRY)


def load_dataset(
    name: str,
    *,
    archive: Path | str | None = None,
    spec: synthetic.SyntheticSpec | None = None,
) -> LoadedDataset:
    """Load a registered dataset by name.

    ``name`` is one of :data:`DATASET_NAMES` or ``jsonl:<path>``; ``archive``, the zip path, is
    required for ``movielens_small`` and ignored otherwise; ``spec`` overrides the synthetic
    generator. Raises DataIntegrityError for an unknown name or a missing or invalid corpus.
    """
    if name.startswith("jsonl:"):
        return load_jsonl_corpus(name[len("jsonl:") :])

    loader = _REGISTRY.get(name)
    if loader is None:
        raise DataIntegrityError(
            f"unknown dataset {name!r}; expected one of {DATASET_NAMES} or 'jsonl:<path>'"
        )
    return loader(name, archive, spec)
