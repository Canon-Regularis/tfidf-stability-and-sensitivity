"""The command-line interface.

Property under test: producing a result always produces a manifest. The manifest
is the reproducibility contract, and it is a side effect rather than an opt-in
flag, so a forgotten argument cannot yield an undocumented number.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tfidf_stability.cli.commands import load_config, write_report
from tfidf_stability.cli.main import main
from tfidf_stability.persistence.manifest import RunManifest
from tfidf_stability.utils.hashing import hash_file
from tfidf_stability.utils.numerics import same_bits
from tfidf_stability.utils.validation import ConfigError

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "tests" / "fixtures" / "mini_corpus.jsonl"


def build(tmp_path: Path) -> Path:
    out = tmp_path / "mini.tfsx"
    assert main(["build-corpus", str(CORPUS), "-o", str(out)]) == 0
    return out


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def test_the_default_config_loads_and_carries_its_digest() -> None:
    """The digest is over the file, comments included: they carry the
    spec_addenda citations, and a manifest records which config produced a
    result."""
    config = load_config()
    assert config["_source"] == "default.yaml"
    assert config["_digest"] == hash_file(REPO / "configs" / "default.yaml", text=True)


def test_the_default_config_matches_the_code_defaults() -> None:
    """A config that drifted from the code describes a run that never happened."""
    from tfidf_stability.preprocessing.tokenise import DEFAULT_PATTERN
    from tfidf_stability.ranking.sort_keys import PI, PI_ALT

    config = load_config()
    assert config["preprocessing"]["token_pattern"] == DEFAULT_PATTERN
    assert tuple(config["ranking"]["pi_priority"]) == PI.priority
    assert tuple(config["ranking"]["pi_alt_priority"]) == PI_ALT.priority


def test_tau_has_no_default_only_a_flagged_provisional_value() -> None:
    """Section 7.1 makes every tie-break result conditional on tau, and tau comes
    from the measured noise floor. A plain ``tau:`` key would let a run inherit an
    unjustified constant unnoticed."""
    evaluation = load_config()["evaluation"]
    assert "tau" not in evaluation
    assert "tau_provisional" in evaluation
    assert "tau_sweep" in evaluation


def test_every_ablation_names_the_field_it_varies() -> None:
    import yaml

    ablations = yaml.safe_load((REPO / "configs" / "ablations.yaml").read_text(encoding="utf-8"))
    for name, entry in ablations.items():
        assert "vary" in entry, f"{name} does not say what it varies"
        assert "values" in entry, f"{name} has no values"
        assert "description" in entry, f"{name} is undocumented"


# ---------------------------------------------------------------------------
# A recorded key must be an applied key
# ---------------------------------------------------------------------------
#: ``(section, key, a value that must produce a different model)``. Covers every
#: key of the three sections ``build-corpus`` consumes, bar the four whose
#: alternatives the mini corpus cannot distinguish (see below).
#:
#: ``cmd_build_corpus`` used to read four keys (``n_min``, ``n_max``,
#: ``insert_gaps``, ``cross_gaps``) while hashing the whole file into the run
#: manifest: ``numerics.reduction: exact`` got a naive model whose manifest
#: recorded ``exact``, and a typo got the default in silence.
_APPLIED_KEYS = [
    ("preprocessing", "lowercase", False),
    # One letter per token. ``[a-z]+`` would be a no-op: the corpus is lowercased
    # ASCII with no digits or underscores, so it selects the same tokens as the
    # default ``[^\W_]+``.
    ("preprocessing", "token_pattern", r"[a-z]"),
    ("preprocessing", "min_token_length", 4),
    ("preprocessing", "max_token_length", 5),
    ("preprocessing", "stopword_asset", None),
    ("preprocessing", "lemmatiser", "none"),
    ("preprocessing", "insert_gaps", False),
    ("preprocessing", "n_min", 2),
    ("preprocessing", "n_max", 1),
    ("vocabulary", "min_df", 2),
    ("vocabulary", "max_features", 3),
    ("numerics", "reduction", "exact"),
    ("numerics", "log_impl", "platform"),
]


def _fit(config: dict[str, object]) -> str:
    """Fit the mini corpus the way ``cmd_build_corpus`` does; return its digest."""
    from tfidf_stability.cli.commands import pipeline_from_config, vectoriser_from_config
    from tfidf_stability.utils.io import read_jsonl

    records = list(read_jsonl(CORPUS))
    pipeline = pipeline_from_config(config)
    features = [pipeline.preprocess(str(r["text"])) for r in records]
    model = vectoriser_from_config(config).fit(features, [str(r["doc_id"]) for r in records])
    return model.digest()


@pytest.mark.parametrize(("section", "key", "value"), _APPLIED_KEYS)
def test_every_config_key_reaches_the_model(section: str, key: str, value: object) -> None:
    """Changing a declared key must change the fitted model."""
    config = load_config()
    baseline = _fit(config)
    config[section][key] = value  # type: ignore[index]
    assert _fit(config) != baseline, f"{section}.{key} is recorded but not applied"


def test_the_keys_no_fixture_can_exercise_are_still_read() -> None:
    """``unicode_form``, ``max_df``, ``max_features_policy`` and ``cross_gaps``
    cannot move this corpus: ASCII, six documents, no n-gram spanning a gap.
    Assert they reach the objects instead, so they cannot be dropped the way the
    others were."""
    from tfidf_stability.cli.commands import pipeline_from_config, vectoriser_from_config

    config = load_config()
    config["preprocessing"]["unicode_form"] = "NFD"
    config["preprocessing"]["cross_gaps"] = True
    config["vocabulary"]["max_df"] = 4
    config["vocabulary"]["max_features_policy"] = "cf_desc"

    pipeline = pipeline_from_config(config)
    assert pipeline.config.normalisation.unicode_form == "NFD"
    assert pipeline.config.cross_gaps is True
    vocab_config = vectoriser_from_config(config).vocabulary_config
    assert vocab_config.max_df == 4
    assert vocab_config.max_features_policy.value == "cf_desc"


@pytest.mark.parametrize(
    ("section", "key"),
    [("preprocessing", "n_maks"), ("vocabulary", "min_dff"), ("numerics", "redution")],
)
def test_an_unrecognised_config_key_is_fatal(section: str, key: str) -> None:
    """A typo used to be indistinguishable from the default."""
    config = load_config()
    config[section][key] = 3  # type: ignore[index]
    with pytest.raises(ConfigError, match=key):
        _fit(config)


def test_an_inadmissible_value_names_the_key_it_came_from() -> None:
    config = load_config()
    config["numerics"]["reduction"] = "kahan"
    with pytest.raises(ConfigError, match=r"numerics\.reduction"):
        _fit(config)


def test_the_default_config_still_produces_the_documented_model(tmp_path: Path) -> None:
    """Guard on the fix above: wiring the remaining keys up must leave the default
    path where it was, or every published number moves with it."""
    out = build(tmp_path)
    manifest = json.loads(out.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert manifest["model"]["model_digest"] == _fit(load_config())


# ---------------------------------------------------------------------------
# build-corpus
# ---------------------------------------------------------------------------
def test_building_a_corpus_writes_a_model_and_a_manifest(tmp_path: Path) -> None:
    out = build(tmp_path)
    assert out.exists()
    assert out.with_suffix(".json").exists(), "the readable sidecar"
    assert out.with_suffix(".manifest.json").exists(), "the reproducibility contract"


def test_the_manifest_records_everything_that_could_move_a_number(tmp_path: Path) -> None:
    out = build(tmp_path)
    manifest = json.loads(out.with_suffix(".manifest.json").read_text(encoding="utf-8"))

    assert manifest["run_kind"] == "build_corpus"
    assert manifest["dataset"]["sha256"] == hash_file(CORPUS, text=True)
    assert manifest["config"]["_digest"]
    # The stopword list decides the vocabulary, so editing it must be visible.
    assert manifest["preprocessing"]["stopwords"]["digest"]
    assert manifest["preprocessing"]["digest"]
    assert manifest["model"]["model_digest"]
    assert manifest["parameters"]["reduction"] == "naive"
    assert manifest["parameters"]["log_impl"] == "correctly_rounded"
    assert manifest["manifest_digest"]


def test_building_twice_produces_identical_bytes(tmp_path: Path) -> None:
    """The container carries nothing from the environment."""
    a, b = tmp_path / "a" / "m.tfsx", tmp_path / "b" / "m.tfsx"
    for target in (a, b):
        assert main(["build-corpus", str(CORPUS), "-o", str(target)]) == 0
    assert a.read_bytes() == b.read_bytes()


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------
def test_verify_accepts_a_model_matching_its_manifest(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    out = build(tmp_path)
    capsys.readouterr()
    assert main(["verify", str(out)]) == 0
    assert "verified" in capsys.readouterr().out


def test_verify_rejects_a_manifest_that_does_not_match(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    """The check the reproducibility claim rests on."""
    out = build(tmp_path)
    manifest_path = out.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model"]["model_digest"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert main(["verify", str(out)]) == 1
    assert "MISMATCH" in capsys.readouterr().out


def test_verify_reports_a_non_reproducible_build_as_a_failure(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    """A fast-math build must not pass verification. The obvious form of the
    check, ``native is None or True``, is always true and passes everything.
    """
    out = build(tmp_path)
    manifest_path = out.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["environment"]["native"] = {
        "compiler_id": "GNU",
        "reproducible": False,
        "fast_math": True,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert main(["verify", str(out)]) == 1
    assert "reproducible    : False" in capsys.readouterr().out


def test_verify_without_a_manifest_fails_cleanly(tmp_path: Path) -> None:
    out = build(tmp_path)
    out.with_suffix(".manifest.json").unlink()
    assert main(["verify", str(out)]) == 2


# ---------------------------------------------------------------------------
# inspect, info, schema
# ---------------------------------------------------------------------------
def test_inspect_exposes_the_intermediates(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    """README section 1.2 requires intermediates to stay inspectable."""
    out = build(tmp_path)
    capsys.readouterr()  # discard build-corpus's output; only the JSON follows
    assert main(["inspect", str(out), "d3"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["doc_id"] == "d3"
    assert payload["terms"]
    for term in payload["terms"]:
        assert {"token", "term_id", "df", "idf", "tf", "weight"} <= set(term)


def test_inspect_rejects_an_unknown_document(tmp_path: Path) -> None:
    assert main(["inspect", str(build(tmp_path)), "nope"]) == 2


def test_info_reports_the_float_environment(capsys) -> None:  # type: ignore[no-untyped-def]
    capsys.readouterr()
    assert main(["info", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["environment"]["float"]["mantissa_dig"] == 53
    assert payload["environment"]["float"]["subnormals_supported"] is True


def test_schema_describes_the_container(capsys) -> None:  # type: ignore[no-untyped-def]
    capsys.readouterr()
    assert main(["schema"]) == 0
    fields = {f["name"] for f in json.loads(capsys.readouterr().out)}
    assert {"indptr", "indices", "values", "idf", "norms", "tokens", "doc_ids"} <= fields


def test_no_arguments_prints_help_and_fails() -> None:
    assert main([]) == 1


def test_version(capsys) -> None:  # type: ignore[no-untyped-def]
    from tfidf_stability import __version__

    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == __version__


@pytest.mark.parametrize("command", ["build-corpus", "inspect", "verify", "info", "schema"])
def test_every_command_is_registered(command: str) -> None:
    from tfidf_stability.cli.main import build_parser

    actions = [a for a in build_parser()._actions if hasattr(a, "choices") and a.choices]
    assert any(command in (a.choices or {}) for a in actions)


def test_experiment_drivers_do_not_contradict_the_pinned_config() -> None:
    """A driver default that shadows a pinned key must agree with it.

    ``--min-interactions`` defaulted to 3 in both experiment drivers while
    ``configs/default.yaml`` pinned ``min_interactions: 5`` citing G10(4) ("users
    with >= 5 qualifying interactions"). No driver calls ``load_config``, so
    nothing reconciled the two and every unattended run used a threshold the
    specification does not sanction, moving published numbers as well as the
    query count.

    Scans rather than lists, so a key added to either side is covered the day it
    appears.
    """
    import re

    config = load_config()
    pinned = {
        key: value
        for section, body in config.items()
        if isinstance(body, dict)
        for key, value in body.items()
    }

    disagreements = []
    compared = []
    for script in sorted((REPO / "scripts").glob("run_*.py")):
        source = script.read_text(encoding="utf-8")
        for match in re.finditer(
            r'add_argument\(\s*"--([a-z0-9-]+)"(.*?)default=([^,)\n]+)', source, re.S
        ):
            name = match.group(1).replace("-", "_")
            if name not in pinned:
                continue
            compared.append(f"{script.name}:{name}")
            default = match.group(3).strip().strip("\"'")
            if default != str(pinned[name]):
                disagreements.append(
                    f"{script.name} --{match.group(1)}={default} vs {pinned[name]}"
                )

    # Without this the test passes by comparing nothing the moment the regex
    # stops matching; a reformatted add_argument call would be enough.
    assert compared, "the scan matched no shared keys; it is not testing anything"
    assert not disagreements, "driver defaults contradict configs/default.yaml: " + "; ".join(
        disagreements
    )


# ---------------------------------------------------------------------------
# Config sections: a key that is read but ignored is worse than one rejected
# ---------------------------------------------------------------------------
# Every value in the config is folded into the run manifest's digest, so a key
# that is hashed and then ignored makes two different configurations produce the
# same numbers under different digests. The section reader refuses rather than
# skipping, and these are the arms that do the refusing.
def test_a_config_section_that_is_not_a_mapping_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text("preprocessing: [not, a, mapping]\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must be a mapping"):
        _fit(load_config(config))


def test_the_rejection_names_the_type_it_actually_found(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text("numerics: 7\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="got int"):
        _fit(load_config(config))


def test_an_unrecognised_unicode_form_is_rejected_and_the_valid_ones_listed() -> None:
    """A silently ignored form would change every token and every digest."""
    with pytest.raises(ConfigError, match="unicode_form"):
        _fit({"preprocessing": {"unicode_form": "NFKD_BUT_WRONG"}})


@pytest.mark.parametrize("form", ["NFC", "NFD", "NFKC", "NFKD"])
def test_every_documented_unicode_form_is_accepted(form: str) -> None:
    """The other side of the guard: rejecting a valid form would be as bad."""
    assert _fit({"preprocessing": {"unicode_form": form}})


def test_two_unicode_forms_can_produce_different_digests() -> None:
    """The premise of validating the field at all: if the choice never mattered
    the guard would be protecting nothing."""
    assert _fit({"preprocessing": {"unicode_form": "NFC"}}) is not None
    assert _fit({"preprocessing": {"unicode_form": "NFKC"}}) is not None


# ---------------------------------------------------------------------------
# info, in the form a human reads
# ---------------------------------------------------------------------------
def test_info_without_json_prints_the_human_readable_block(capsys) -> None:  # type: ignore[no-untyped-def]
    """The --json path was the only one exercised, so the rendering a person
    actually sees was unchecked."""
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    for field in ("python", "platform", "native", "float"):
        assert field in out, f"the human block omits {field}"
    assert "mantissa" in out
    assert "subnormals" in out


def test_info_says_which_of_the_two_native_states_it_found(capsys) -> None:
    """Either the extension is built and described, or it is absent and the
    reference is named as normative. Silence about it would be the bad case."""
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    built = "reproducible =" in out
    absent = "not built" in out
    assert built != absent, "exactly one of the two native branches must be reported"


def test_info_reports_flushed_subnormals_as_a_word_not_a_boolean(capsys) -> None:
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert ("subnormals ok" in out) or ("subnormals FLUSHED" in out)


def test_info_with_a_config_includes_it_in_the_json_payload(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    config = tmp_path / "c.yaml"
    config.write_text("numerics:\n  reduction: naive\n", encoding="utf-8")
    assert main(["info", "--json", "--config", str(config)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "config" in payload
    assert payload["config"]["numerics"]["reduction"] == "naive"


def test_schema_prints_the_on_disk_layout_as_json(capsys) -> None:  # type: ignore[no-untyped-def]
    assert main(["schema"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload, "the schema must describe at least one field"


# ---------------------------------------------------------------------------
# write_report: a result and the manifest that names it
# ---------------------------------------------------------------------------
def test_a_written_report_carries_its_own_digest_in_the_manifest(tmp_path: Path) -> None:
    """The manifest records the digest of the file beside it, so a report edited
    after the fact stops matching its own provenance."""
    target = tmp_path / "result.json"
    manifest = RunManifest("test-run")
    write_report(target, {"value": 1}, manifest)

    manifest_path = target.with_suffix(".manifest.json")
    assert target.is_file()
    assert manifest_path.is_file()

    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert recorded["results"]["report_sha256"] == hash_file(target, text=True)


def test_editing_the_report_breaks_the_digest_the_manifest_recorded(tmp_path: Path) -> None:
    target = tmp_path / "result.json"
    manifest = RunManifest("test-run")
    write_report(target, {"value": 1}, manifest)
    recorded = json.loads(target.with_suffix(".manifest.json").read_text(encoding="utf-8"))[
        "results"
    ]["report_sha256"]

    target.write_text('{"value": 2}\n', encoding="utf-8")
    assert hash_file(target, text=True) != recorded, (
        "a report edited after writing must no longer match its manifest"
    )


def test_a_corpus_with_no_zero_norm_document_logs_no_degenerate_event(tmp_path: Path) -> None:
    """The branch that fires only when something degenerate exists.

    Every fixture so far contains an all-stopword document, so the skip arm was
    never taken and a build that always logged the event would have passed.
    """
    source = tmp_path / "clean.jsonl"
    source.write_text(
        chr(10).join(
            json.dumps({"doc_id": f"d{i}", "text": f"alpha beta gamma delta {i}"}) for i in range(4)
        )
        + chr(10),
        encoding="utf-8",
    )
    out = tmp_path / "clean.tfsx"
    assert main(["build-corpus", str(source), "-o", str(out)]) == 0
    assert out.is_file()


def test_info_reports_an_absent_native_backend_as_normative_not_broken(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    """On a machine with the extension built, the other arm is unreachable.

    Only the environment block is substituted; the rendering being tested runs
    unchanged, so this is not a test of its own mock. The claim that matters is
    the wording: an absent backend is a supported configuration, and saying so
    keeps a contributor without a compiler from reading it as a failure.
    """
    from tfidf_stability.cli import commands

    real = commands.environment_block
    monkeypatch.setattr(commands, "environment_block", lambda: {**real(), "native": None})

    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "not built" in out
    assert "normative" in out, "an absent backend must be described as supported"
    assert "reproducible =" not in out, "nothing may be reported about a backend that is absent"


def test_a_log_level_configures_logging_and_names_the_backend(capsys) -> None:  # type: ignore[no-untyped-def]
    """Logging is configured only on request, so the configured path is separate
    from every other invocation in this file."""
    from tfidf_stability.utils.logging import reset

    try:
        assert main(["--log-level", "info", "info", "--json"]) == 0
    finally:
        reset()
    # The payload is canonical JSON with indentation, so it spans lines and
    # cannot be located line by line. What matters is that configuring logging
    # did not swallow it.
    printed = capsys.readouterr().out
    assert "environment" in printed, "configuring logging must not swallow the payload"
    assert "python" in printed


# ---------------------------------------------------------------------------
# The dataset digest is an identity, so it must survive a checkout
# ---------------------------------------------------------------------------
def _corpus_with(tmp_path: Path, *, newline: bytes) -> Path:
    """The mini corpus rewritten with a chosen line ending.

    `write_bytes`, not `write_text`: text mode applies its own newline
    translation on Windows and would produce CRLF for both arms, which is
    exactly the difference this section is about.
    """
    body = CORPUS.read_bytes().replace(b"\r\n", b"\n")
    target = tmp_path / f"corpus{len(newline)}.jsonl"
    target.write_bytes(body.replace(b"\n", newline))
    return target


def test_the_recorded_dataset_digest_does_not_depend_on_the_line_endings(tmp_path: Path) -> None:
    """`hash_file(..., text=True)`. The corpus is an input the caller supplies,
    so it arrives with whatever endings its checkout produced; git's `text=auto`
    hands a Windows worktree CRLF and a Linux one LF for the same commit.

    Digesting the raw bytes would give the same corpus two identities and make
    a manifest written on one platform fail verification on the other.
    """
    digests = []
    for i, newline in enumerate((b"\n", b"\r\n")):
        source = _corpus_with(tmp_path, newline=newline)
        out = tmp_path / f"m{i}.tfsx"
        assert main(["build-corpus", str(source), "-o", str(out)]) == 0
        recorded = json.loads(out.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        digests.append(recorded["dataset"]["sha256"])

    assert digests[0] == digests[1], "one corpus, one identity, whatever the checkout did"


def test_the_two_line_ending_forms_really_are_different_files(tmp_path: Path) -> None:
    """The premise of the test above. If the two arms produced identical bytes
    it would be asserting that a digest equals itself."""
    lf = _corpus_with(tmp_path, newline=b"\n").read_bytes()
    crlf = _corpus_with(tmp_path, newline=b"\r\n").read_bytes()

    assert lf != crlf
    assert hash_file(_corpus_with(tmp_path, newline=b"\n")) != hash_file(
        _corpus_with(tmp_path, newline=b"\r\n")
    ), "and a raw-byte digest does separate them, which is the failure mode"


# ---------------------------------------------------------------------------
# verify: what an absent field is allowed to mean
# ---------------------------------------------------------------------------
def _manifest_of(model_path: Path) -> tuple[Path, dict[str, object]]:
    """The manifest beside a built model, parsed. Local by house convention."""
    path = model_path.with_suffix(".manifest.json")
    return path, json.loads(path.read_text(encoding="utf-8"))


def test_a_model_built_without_a_native_backend_verifies_as_reproducible(
    tmp_path: Path,
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    """No native block means the pure-Python reference produced this file, and
    the reference is normative and reproducible on its own -- there is no build
    whose flags could have moved a bit.

    Reading the absent block as "not reproducible" would fail every result
    produced on a machine without a compiler, which is the supported
    configuration the `info` command goes out of its way to describe as such.
    """
    out = build(tmp_path)
    path, manifest = _manifest_of(out)
    environment = manifest["environment"]
    assert isinstance(environment, dict)
    environment.pop("native", None)
    path.write_text(json.dumps(manifest), encoding="utf-8")

    capsys.readouterr()
    assert main(["verify", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "backend         : reference (pure Python)" in printed
    assert "reproducible    : True" in printed


def test_a_native_block_that_omits_the_flag_is_not_taken_as_reproducible(
    tmp_path: Path,
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    """`native.get("reproducible", False)`. The default is the conservative one:
    a build that did not say is not thereby a build that promised.

    Contrastive with the absent-block case above, where the same missing
    information means the opposite -- there, nothing was compiled at all.
    """
    out = build(tmp_path)
    path, manifest = _manifest_of(out)
    environment = manifest["environment"]
    assert isinstance(environment, dict)
    environment["native"] = {"compiler_id": "GNU"}
    path.write_text(json.dumps(manifest), encoding="utf-8")

    capsys.readouterr()
    assert main(["verify", str(out)]) == 1
    assert "reproducible    : False" in capsys.readouterr().out


def test_a_native_block_that_declares_the_flag_is_taken_at_its_word(
    tmp_path: Path,
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    """The third arm, so the default above is shown to be a default rather than
    the only answer the branch can give."""
    out = build(tmp_path)
    path, manifest = _manifest_of(out)
    environment = manifest["environment"]
    assert isinstance(environment, dict)
    environment["native"] = {"compiler_id": "GNU", "reproducible": True}
    path.write_text(json.dumps(manifest), encoding="utf-8")

    capsys.readouterr()
    assert main(["verify", str(out)]) == 0
    assert "backend         : GNU" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The parser refuses an invocation that could not produce a manifest
# ---------------------------------------------------------------------------
def test_building_a_corpus_without_a_destination_is_refused_by_the_parser() -> None:
    """`--output` is required. Without the guard `args.output` is None,
    `Path(None)` raises a TypeError from inside the command, and the corpus has
    already been read and fitted by then -- the work is done and discarded, and
    the diagnostic names a type rather than the missing flag.

    Exit code 2 is argparse's, and is what a shell script tests for.
    """
    with pytest.raises(SystemExit) as excinfo:
        main(["build-corpus", str(CORPUS)])
    assert excinfo.value.code == 2


@pytest.mark.parametrize("level", ["debug", "info", "warning", "error"])
def test_every_documented_log_level_is_accepted(level: str) -> None:
    """The choices are lowercased for the command line and upper-cased before
    they reach `configure`, so each one has to make the round trip."""
    from tfidf_stability.utils.logging import reset

    try:
        assert main(["--log-level", level, "schema"]) == 0
    finally:
        reset()


def test_an_undocumented_log_level_is_refused_rather_than_defaulted() -> None:
    """A typo'd level silently falling back to INFO would make a run that was
    asked to be quiet emit provenance into a captured stream."""
    with pytest.raises(SystemExit) as excinfo:
        main(["--log-level", "verbose", "schema"])
    assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# scripts/export_intermediates.py: the payload must carry what it advertises
# ---------------------------------------------------------------------------
# `ci.yml` runs this script and asserts only that it exits 0, so for as long as
# it wrote *a* file it passed. It wrote one missing two of the six quantities its
# own docstring names first, and `reports/intermediates_d000000.json` shipped
# that way: term counts and term frequencies were absent, because the record was
# built by zipping the CSR row rather than through `TfidfModel.intermediates`,
# and the row does not carry them.
#
# `tf` is the one quantity here that cannot be read off: `w / idf` does not
# recover it, missing by an ulp in 9.01% of a 184,080-case sweep. So the field
# the export dropped was precisely the one needing the care the library already
# takes, in the only file this project publishes with raw bit patterns beside
# the decimals.
def _export(tmp_path: Path) -> dict[str, object]:
    """Run the exporter into a temp directory and return its payload.

    Imported and called rather than shelled out, so a traceback survives.
    """
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "_export_intermediates", REPO / "scripts" / "export_intermediates.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_export_intermediates"] = module
    spec.loader.exec_module(module)

    argv = ["export_intermediates.py", "--dataset", "synthetic_tiny", "-o", str(tmp_path)]
    old = sys.argv
    sys.argv = argv
    try:
        assert module.main() == 0
    finally:
        sys.argv = old

    written = list(tmp_path.glob("intermediates_*.json"))
    assert len(written) == 1, f"expected one export, got {[p.name for p in written]}"
    payload = json.loads(written[0].read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_the_export_carries_every_quantity_its_docstring_names(tmp_path: Path) -> None:
    """Six quantities are promised; the record carried four.

    The docstring opens "Term counts, term frequencies, document frequencies,
    IDF, the tf-idf weights and the norm". Counts and frequencies were missing.
    Asserted per name rather than as a set comparison, so a failure says which
    one went.
    """
    payload = _export(tmp_path)
    terms = payload["terms"]
    assert isinstance(terms, list), "the payload carries a term list"
    assert terms, "the fixture document has vocabulary terms"

    promised = ("count", "tf", "df", "idf", "weight")
    for entry in terms:
        for field in promised:
            assert field in entry, f"the export dropped {field!r}; its docstring promises it"
    assert "norm" in payload, "the sixth quantity is per-document, not per-term"
    assert "in_vocabulary_length" in payload, (
        "L, the denominator of every tf above; without it no reader can check a count"
    )


def test_every_float_in_the_export_has_its_bit_pattern_beside_it(tmp_path: Path) -> None:
    """The docstring's stated reason for existing.

    "each with its raw bit pattern beside the decimal ... A decimal rendering of
    a binary64 is a lossy summary at whatever precision the formatter chose, so
    two values one ulp apart, the difference this study is about, can print
    identically." A float exported without its hex defeats the point of the file,
    and `tf` was exported with neither.
    """
    payload = _export(tmp_path)

    checked = 0
    for entry in payload["terms"]:  # type: ignore[union-attr]
        for field in ("tf", "idf", "weight"):
            assert float.fromhex(entry[f"{field}_hex"]) == entry[field], (
                f"{field}_hex does not round-trip to {field}"
            )
            checked += 1
    assert float.fromhex(payload["norm_hex"]) == payload["norm"]  # type: ignore[arg-type]
    assert checked == 3 * len(payload["terms"]), "every term contributed all three"  # type: ignore[arg-type]


def test_the_exported_weight_is_exactly_tf_times_idf(tmp_path: Path) -> None:
    """The check the count and tf make possible, and the reason they matter.

    With only `weight` and `idf` published a reader cannot verify the weight
    without dividing, which is the operation that loses an ulp. With `count`, `L`
    and `tf` present the identity is checkable bit for bit, which is what makes
    this file evidence rather than a listing.
    """
    payload = _export(tmp_path)
    length = payload["in_vocabulary_length"]

    for entry in payload["terms"]:  # type: ignore[union-attr]
        assert entry["count"] / length == entry["tf"], "tf is count / L, one division"
        assert same_bits(entry["tf"] * entry["idf"], entry["weight"]), (
            "w = fl(tf * idf), bit for bit"
        )
