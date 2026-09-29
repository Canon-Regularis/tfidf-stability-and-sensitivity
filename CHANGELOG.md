# Changelog

Notable changes to this project, in [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
form. Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Written by hand rather than generated. `release.yml` sets
`generate_release_notes: true`, which lists merged pull-request titles; this
repository's history is direct commits, so that list is close to empty and says
nothing about what moved. A published number changing is the thing a reader of
this project needs told, and no commit log states it.

**Every entry that moves a published number names the digest it moved to.** The
recorded values live in `configs/pipeline_digest.txt` and in each
`reports/*.json`, and `scripts/snapshot.py --check` is what compares them.

## [Unreleased]

## [0.2.0] - 2026-09-29

The first tagged release. Published to PyPI as `tfidf-stability` 0.2.0, with
wheels for CPython 3.11 to 3.13 on Linux, macOS and Windows, and archived at
[10.5281/zenodo.23047931](https://doi.org/10.5281/zenodo.23047931).
`scripts/check_versions.py --tag v0.2.0` is what holds the tag and the four
files that state the version together.

### Added

- `wheels.yml` and the reusable `build-wheels.yml`: the wheel build now runs
  nightly, and on any push or pull request touching `cpp/**`, `CMakeLists.txt`,
  `CMakePresets.json` or the build workflows, rather than only on a release
  tag. The push trigger matters because this repository's history is direct
  commits. The nightly run also builds cp314, which no release ships until a
  nightly has produced it.
- `codeql.yml`: static analysis for `python` and `c-cpp`, the latter built
  through the `ci` preset so the flags analysed are the flags shipped.
- `SECURITY.md`, and `REUSE.toml` giving the Apache-2.0 / CC-BY-4.0 split a
  machine-readable form. `reuse lint` runs as the ninth repository gate.
- `reports/summary.md`, generated from the committed reports by
  `scripts/render_summary.py`, with a `--check` mode that fails when the two
  disagree.
- A TestPyPI rehearsal that blocks `publish`, build-provenance attestations over
  the release assets, a CycloneDX SBOM generated from a clean runtime
  environment, and recorded SHA-256 digests for everything published.

### Changed

- `release.yml`: the `wheels` matrix now `needs: [sdist]`, so the tag-versus-
  declared-version check stops the build before three platforms run rather than
  only blocking publication.
- The wheel now carries `LICENSES/BSD-3-Clause.txt` and
  `THIRD_PARTY_NOTICES.md`. The vendored Snowball stemmer is BSD-3-Clause and
  nanobind is statically linked into the extension, so both the licence text
  and the copyright notices it requires go out with the package.
- `THIRD_PARTY_NOTICES.md` records nanobind as linked into the distributed
  package rather than as a build-time dependency, which is what
  `nanobind_add_module(... NB_STATIC ...)` makes it.
- `REUSE.toml` names the vendored files individually. The three generated
  Snowball modules and the two test vectors are BSD-3-Clause; the `__init__.py`
  and the `MANIFEST.sha256` beside them are first-party and stay Apache-2.0.
  Each upstream holder is recorded as a separate copyright statement, so the
  generated SPDX document attributes the years correctly.
- Report payloads use one key per concept: `is_exact_tie` everywhere,
  `E1_excluded_undefined` for what it counts, `trajectory_seed` distinct from
  the run seed, and document ids rather than candidate indices.
  `intermediates_*.json` gained the envelope its siblings carry, so it can be
  identity-checked for the first time.
- The seven figures are one visual system: a shared style, a palette in which a
  colour means one thing throughout, and a stamp grammar that is the same on
  each. A measured zero is now drawn as a zero rather than as an absent bar.

### Fixed

- `ruff check --fix` was rewriting the vendored Snowball stemmer through
  pre-commit and breaking the SHA-256 that `scripts/check_vendored.py` verifies.
  `extend-exclude` does not apply to filenames passed explicitly, which is what
  pre-commit does; both ruff hooks now pass `--force-exclude`.
- The `vendored-digests` hook did not select `data/assets/`, so a change to a
  hashed asset did not run the check that verifies it.
- The figures carried provenance stamps naming result digests that no longer
  existed, because `reports/*.json` was regenerated without re-rendering them.
- The mutation sandbox did not copy `CMakePresets.json`, `.github`,
  `requirements-dev.txt` or `reports/`, so a suite that reads any of them failed
  the campaign baseline before a mutant was applied.

[Unreleased]: https://github.com/Canon-Regularis/tfidf-stability-and-sensitivity/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Canon-Regularis/tfidf-stability-and-sensitivity/releases/tag/v0.2.0
