# Third-Party Notices

Material from other projects that is redistributed as part of this repository.
Each entry records the upstream version and the digests under which it was
vendored, so the provenance of every byte is checkable. Digests are verified by
`scripts/check_vendored.py`, which runs as a pre-commit hook
(`.pre-commit-config.yaml`) and as a CI step (`.github/workflows/ci.yml`).

This previously said the digests were verified "at build time
(`cpp/cmake/VendorCheck.cmake`)". There is no such file, and no CMake code in
this repository computes a SHA-256 of anything: a corrupted vendored header
built cleanly and hashed nothing. The verification described above is real and
is what the claim now names.

Nothing here is a runtime dependency of the pure-Python `reference` backend
beyond the vendored Snowball stemmer; that backend is otherwise standard-library
only, by design.

---

## Vendored into the distributed package

### Snowball English stemmer
- **Location:** `src/tfidf_stability/preprocessing/_snowball/`
- **Upstream:** https://snowballstem.org — `snowballstemmer` 3.1.1
- **Licence:** BSD-3-Clause
- **Files:** `among.py`, `basestemmer.py`, `english_stemmer.py`
- **Digests:** `src/tfidf_stability/preprocessing/_snowball/MANIFEST.sha256`

Vendored rather than depended upon so that the Python and C++ backends run
implementations generated from the *same* upstream source, and therefore agree by
construction rather than by hand-porting. Validated against the official
42 649-word test vectors on every CI run.

> Copyright (c) 2001, Dr Martin Porter; Copyright (c) 2004, Richard Boulton;
> Copyright (c) 2013, Yoshiki Shibukawa; Copyright (c) 2006, 2007, 2009, 2010,
> 2011, 2014, Olly Betts. All rights reserved. Redistribution and use in source
> and binary forms, with or without modification, are permitted provided that the
> conditions of the BSD 3-Clause Licence are met.

---

## Vendored for building and testing only (not shipped in wheels)

### doctest
- **Location:** `cpp/third_party/doctest/doctest.h`
- **Upstream:** https://github.com/doctest/doctest — v2.4.11
- **Licence:** MIT
- **Digest:** `cpp/third_party/MANIFEST.sha256`

Single-header C++ test framework. Vendored so the native build is hermetic and
works offline, which a reproducibility-focused artefact should.

### nanobench
- **Location:** `cpp/third_party/nanobench/nanobench.h`
- **Upstream:** https://github.com/martinus/nanobench — v4.3.11
- **Licence:** MIT
- **Digest:** `cpp/third_party/MANIFEST.sha256`

Single-header microbenchmark library, used to measure kernel costs without the
FFI overhead that `pytest-benchmark` necessarily includes.

### Snowball test vectors
- **Location:** `tests/fixtures/snowball/{voc.txt,output.txt}`
- **Upstream:** https://github.com/snowballstem/snowball-data
- **Licence:** BSD-3-Clause
- **Digest:** `tests/fixtures/snowball/MANIFEST.sha256`

---

## Build-time dependencies (not redistributed)

| Project | Licence | Role |
|---|---|---|
| nanobind | BSD-3-Clause | Python bindings for the native backend |
| scikit-build-core | Apache-2.0 | PEP 517 build backend |
| numpy | BSD-3-Clause | Array interchange with the native backend |
| PyYAML | MIT | Configuration loading |
| scikit-learn | BSD-3-Clause | **Test-only** external oracle for differential tests |
| snowballstemmer | BSD-3-Clause | **Test-only** oracle; the stemmer itself is vendored |
| pytest, Hypothesis | MIT / MPL-2.0 | Test framework and property-based testing |

---

## Data

**No dataset is redistributed in this repository.**

MovieLens (`ml-latest-small`, GroupLens Research) is fetched on demand by
`scripts/fetch_data.py` and verified against a pinned SHA-256. Its usage licence
**prohibits redistribution**, and it is therefore excluded by `.gitignore`. Users
of the MovieLens data must cite:

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History
> and Context. *ACM Transactions on Interactive Intelligent Systems* 5, 4:
> 19:1–19:19. https://doi.org/10.1145/2827872

See [`data/README.md`](data/README.md) for provenance and terms.
