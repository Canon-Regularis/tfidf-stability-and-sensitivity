# Licensing

Copyright © 2026
**Matthew Maksymilian Miezaniec**

This repository contains two kinds of material, and they are licensed separately.
Creative Commons explicitly advises against using CC BY for software — it grants
no patent rights, and its attribution and "adapted material" obligations do not
map cleanly onto compiled binaries or vendored dependencies. The split below
keeps the exposition under the licence originally chosen for it while putting the
code under a licence that is actually intended for code.

| Material | Licence | SPDX identifier |
|---|---|---|
| **Software** — `src/`, `cpp/`, `tests/`, `scripts/`, `examples/`, `notebooks/` (code cells), build and CI configuration | Apache License 2.0 | `Apache-2.0` |
| **Exposition** — `README.md`, `docs/`, `reports/`, figures, and generated result documents | Creative Commons Attribution 4.0 International | `CC-BY-4.0` |

Full texts are in [`LICENSES/`](LICENSES/). Individual files carry an
`SPDX-License-Identifier:` header where the format permits one.

---

## Software: Apache License 2.0

Chosen over MIT for its **express patent grant** (§3) and its explicit
contribution terms, both of which matter for a research artefact that others may
build on or cite.

You may use, modify and redistribute the code, including commercially, provided
you retain the copyright and licence notices, state significant changes, and
include a copy of the licence. See [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt).

## Exposition: CC BY 4.0

You may share and adapt the prose, figures and derived results, including
commercially, provided you give appropriate credit, link to the licence, and
indicate whether changes were made. Attribution must not suggest that the author
endorses you or your use. See [`LICENSES/CC-BY-4.0.txt`](LICENSES/CC-BY-4.0.txt).

---

## Third-party material

Vendored dependencies retain their own licences, recorded with their upstream
version and SHA-256 digest in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

**Datasets are not redistributed.** MovieLens in particular may not be
redistributed under its usage licence; it is fetched on demand and verified
against a pinned hash. See [`data/README.md`](data/README.md).

---

## Citation

If you use this work, please cite it. Machine-readable metadata is in
[`CITATION.cff`](CITATION.cff).

---

## Disclaimer

This work is provided **"as is"**, without warranty of any kind, express or
implied. Use of this material does **not** imply endorsement by the author.
