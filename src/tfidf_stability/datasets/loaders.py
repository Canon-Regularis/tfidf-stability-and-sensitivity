"""One way in to every dataset, and one place the provenance is recorded.

The registry in ``configs/datasets.yaml`` names datasets; this module turns a
name into records. Everything downstream -- the CLI, the experiment scripts, the
notebooks -- goes through :func:`load_dataset` and none of them know whether the
corpus was generated, downloaded or read off disk.

That indirection earns its keep in one specific way: :class:`LoadedDataset`
carries a :attr:`~LoadedDataset.provenance` block that goes verbatim into the run
manifest. A result computed on synthetic data and a result computed on MovieLens
are then distinguishable *after the fact*, from the manifest alone, including
which archive digest and which generator spec produced it. Without that, the two
are a filename apart, and filenames do not survive being copied.

The asymmetry between the two datasets is deliberate and is a licence
consequence, not an oversight: synthetic data is regenerated from its spec on
demand, MovieLens is loaded from a local archive that this repository will never
contain. See ``data/README.md``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

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

#: Registered names. ``jsonl:<path>`` is also accepted, for a corpus already on
#: disk -- which is how a MovieLens-derived corpus is consumed once
#: ``scripts/build_corpus.py`` has written it out.
DATASET_NAMES = ("synthetic_small", "synthetic_tiny", "movielens_small")

#: Smaller synthetic spec for tests and doctests, where a 2000-document corpus
#: would dominate the runtime of the suite that uses it.
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

        Over the records, not over the provenance -- two loads that produce
        identical documents must agree here even if one came from a regenerated
        spec and the other from a file, since it is the *documents* that
        determine every downstream number.
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
            # Of the file, so a corpus edited in place is detectable.
            "sha256": hash_text(target.read_text(encoding="utf-8")),
            "n_documents": len(records),
        },
    )


def load_dataset(
    name: str,
    *,
    archive: Path | str | None = None,
    spec: synthetic.SyntheticSpec | None = None,
) -> LoadedDataset:
    """Load a registered dataset by name.

    Args:
        name: One of :data:`DATASET_NAMES`, or ``jsonl:<path>``.
        archive: Path to ``ml-latest-small.zip``. Required for ``movielens_small``
            and ignored otherwise.
        spec: Override for the synthetic generator. Ignored for MovieLens.

    Raises:
        DataIntegrityError: Unknown name, or the dataset could not be loaded.

    Example:
        >>> data = load_dataset("synthetic_tiny")
        >>> data.n_documents
        120
        >>> data.provenance["kind"]
        'synthetic'
    """
    if name.startswith("jsonl:"):
        return load_jsonl_corpus(name[len("jsonl:") :])

    if name in ("synthetic_small", "synthetic_tiny"):
        resolved = spec or (_TINY if name == "synthetic_tiny" else synthetic.SyntheticSpec())
        corpus = synthetic.generate(resolved)
        return LoadedDataset(
            name=name,
            records=corpus.records(),
            interactions=list(corpus.interactions),
            provenance={
                "kind": "synthetic",
                # asdict, not __dict__: SyntheticSpec is slots=True, so it has
                # no instance dict at all.
                "spec": {
                    k: list(v) if isinstance(v, tuple) else v
                    for k, v in asdict(corpus.spec).items()
                },
                "spec_digest": corpus.spec.digest(),
                "n_documents": corpus.n_documents,
                "n_twin_pairs": len(corpus.twins),
                "n_exact_duplicate_pairs": len(corpus.exact_duplicate_pairs),
                # Redistributable, so this one *can* be regenerated anywhere --
                # the distinction that matters when reproducing a result.
                "redistributable": True,
            },
        )

    if name == "movielens_small":
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
                # False, and recorded as such: a reader of the manifest can see
                # immediately that this result cannot be reproduced from the
                # repository alone.
                "redistributable": False,
            },
        )

    raise DataIntegrityError(
        f"unknown dataset {name!r}; expected one of {DATASET_NAMES} or 'jsonl:<path>'"
    )
