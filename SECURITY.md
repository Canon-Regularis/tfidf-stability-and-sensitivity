# Security policy

## What this project is

A research artefact: a pure-Python reference implementation of a TF-IDF
similarity pipeline, a C++20 core asserted to be bit-identical to it, and the
measurements in `reports/`. It is a library and a command-line tool. It opens no
socket, runs no server, and has no authentication or authorisation surface.

The realistic exposure is therefore parsing and native memory: the C++ extension
reached through `tfidf_stability._native`, the dataset and configuration parsers
that read files a caller supplies, and the `.tfsx` container reader in
`persistence/save_load.py`, which validates an untrusted file's header, offsets
and index arrays before use.

## Supported versions

The most recent release only. This is a single-maintainer artefact and no
earlier version receives fixes.

| Version | Supported |
|---------|-----------|
| 0.2.x   | yes       |
| < 0.2   | no        |

## Reporting a vulnerability

Report privately through GitHub's advisory flow:
[**Report a vulnerability**](https://github.com/Canon-Regularis/tfidf-stability-and-sensitivity/security/advisories/new).
That opens a channel visible only to the maintainer.

Do not open a public issue for a suspected vulnerability.

A useful report names the affected version, the platform and interpreter, and
the input that triggers the behaviour. A crashing file or a short script is
worth more than a description of one.

Expect an acknowledgement within 14 days. This is maintained alongside other
work and has no support commitment beyond best effort; `CONTRIBUTING.md` states
the same for correctness reports generally.

## What counts

In scope:

- Memory-unsafe behaviour in the C++ core or its bindings, including anything
  reachable from a crafted `.tfsx` file or a crafted corpus.
- A parser that can be made to consume unbounded memory or time by an input of
  bounded size.
- A published wheel whose contents do not match this repository at the tagged
  commit.

Out of scope:

- Numerical disagreement between the Python reference and the C++ core. That is
  a correctness defect and the test suite exists to find it; report it as an
  ordinary issue.
- Vulnerabilities in a dependency, unless this project's use of it is what makes
  the dependency exploitable. Dependency advisories arrive through Dependabot.
- Anything requiring the attacker to already be able to run code as the user.

## How releases are made

Wheels are published to PyPI from `.github/workflows/release.yml` using Trusted
Publishing over OIDC, so no long-lived credential exists to be stolen. Two jobs
hold write permissions and both run only on a `v*` tag: `testpypi`, which mints
an OIDC token to rehearse the upload against TestPyPI under the `testpypi`
environment, and `publish`, which attaches the release assets and uploads to
PyPI, and which requires approval through the `release` environment. Published
distributions carry PEP 740 attestations, and the files attached to the GitHub
release carry build provenance, so both can be verified against this repository.
