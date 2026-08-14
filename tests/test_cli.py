"""The command-line interface.

Property under test: producing a result always produces a manifest. The manifest
is the reproducibility contract, and it is a side effect rather than an opt-in
flag, so a forgotten argument cannot yield an undocumented number.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tfidf_stability.cli.commands import load_config
from tfidf_stability.cli.main import main
from tfidf_stability.utils.hashing import hash_file
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
