"""Logging as a provenance channel. Three properties.

*Importing must not configure logging.* A library that installs a handler takes
the application's decision for it, and this package is imported by notebooks and
by other people's test suites.

*Captured output must be reproducible.* The events are hashed into a run record
beside the manifest, so a timestamp would make identical runs disagree; that is
the failure ``strip_volatile`` prevents.

*Warnings must survive.* ``filterwarnings = ["error"]`` makes
``TauExceedsScoreRangeWarning`` and ``ChainInflationWarning`` hard failures. If
logging filed either as a line on stderr, the suite would go quiet about the
degenerate configurations it exists to catch.
"""

from __future__ import annotations

import io
import logging
import os
import re
import shlex
import subprocess
import sys
import warnings
from collections.abc import Iterator
from pathlib import Path

import pytest

from tfidf_stability.utils.logging import (
    DETERMINISTIC_FORMAT,
    ROOT_NAME,
    Event,
    EventKind,
    capture,
    configure,
    get_logger,
    log_event,
    reset,
)

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "tests" / "fixtures" / "mini_corpus.jsonl"

#: Anything that looks like a wall-clock time, in any of the formats
#: ``logging.Formatter`` produces by default.
CLOCK = re.compile(r"\d{2}:\d{2}:\d{2}")


@pytest.fixture(autouse=True)
def _pristine_logging() -> Iterator[None]:
    """Leave the logging system as each test found it. Autouse: a leaked handler
    makes a later test pass or fail for reasons unrelated to what it asserts.
    """
    logger = logging.getLogger(ROOT_NAME)
    before = list(logger.handlers)
    level = logger.level
    yield
    reset()
    logger.handlers[:] = before
    logger.setLevel(level)


# ---------------------------------------------------------------------------
# No handler on import
# ---------------------------------------------------------------------------
def test_importing_the_package_installs_no_handler() -> None:
    """Subprocess: by collection time the package is long since imported and
    pytest's own logging plugin has a handler on the root logger, so only a fresh
    interpreter can answer the question."""
    probe = (
        "import logging;"
        "import tfidf_stability.cli.main;"
        "import tfidf_stability._native;"
        "print(len(logging.getLogger().handlers),"
        " len(logging.getLogger('tfidf_stability').handlers),"
        " logging.getLogger('tfidf_stability').level)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
        check=True,
    )
    assert result.stdout.split() == ["0", "0", "0"], result.stderr


def test_the_package_logger_has_no_handler_of_ours_until_configured() -> None:
    logger = logging.getLogger(ROOT_NAME)
    reset()
    assert logger.handlers == []


def test_emitting_without_configuration_does_not_raise() -> None:
    """With no handler anywhere ``logging.lastResort`` takes the record, the
    intended fallback: a NullHandler would silence diagnostics a reader of an
    unconfigured run wants to see."""
    reset()
    log_event(get_logger("probe"), EventKind.DEGENERATE, case="degenerate_query")


# ---------------------------------------------------------------------------
# The opt-in configuration
# ---------------------------------------------------------------------------
def test_configure_installs_exactly_one_handler_and_routes_events() -> None:
    buffer = io.StringIO()
    configure(stream=buffer)
    logger = logging.getLogger(ROOT_NAME)

    assert len(logger.handlers) == 1
    assert logger.level == logging.INFO

    log_event(get_logger("probe"), EventKind.DIGEST, artefact="model", sha256="ab")
    assert "digest artefact=model sha256=ab" in buffer.getvalue()


def test_configure_is_idempotent() -> None:
    """An entry point invoked twice in one process must not double every line."""
    configure(stream=io.StringIO())
    configure(stream=io.StringIO())
    assert len(logging.getLogger(ROOT_NAME).handlers) == 1


def test_configure_does_not_touch_the_root_logger() -> None:
    root = logging.getLogger()
    before = list(root.handlers)
    configure(stream=io.StringIO())
    assert root.handlers == before


def test_reset_leaves_a_handler_the_application_added() -> None:
    """Removing a handler we did not install is the same overreach as configuring
    the root logger."""
    logger = logging.getLogger(ROOT_NAME)
    foreign = logging.StreamHandler(io.StringIO())
    logger.addHandler(foreign)
    try:
        configure(stream=io.StringIO())
        reset()
        assert logger.handlers == [foreign]
    finally:
        logger.removeHandler(foreign)


def test_the_level_filters_below_threshold_events() -> None:
    buffer = io.StringIO()
    configure(level=logging.WARNING, stream=buffer)
    log = get_logger("probe")
    log_event(log, EventKind.DIGEST, artefact="model", sha256="ab")
    log_event(log, EventKind.DIAGNOSTIC, diagnostic="TauExceedsScoreRange")
    output = buffer.getvalue()

    # A level fixed per event kind is what makes filtering at WARNING yield the
    # diagnostics and nothing else; it holds only if no call site can choose.
    assert "digest" not in output
    assert "diagnostic diagnostic=TauExceedsScoreRange" in output


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def test_the_deterministic_format_names_no_volatile_record_attribute() -> None:
    for volatile in ("asctime", "created", "msecs", "relativeCreated", "process", "thread"):
        assert volatile not in DETERMINISTIC_FORMAT


def test_configured_output_carries_no_timestamp() -> None:
    buffer = io.StringIO()
    configure(stream=buffer)
    log_event(get_logger("probe"), EventKind.BACKEND_SELECTED, backend="reference")

    assert buffer.getvalue() == "INFO tfidf_stability.probe backend_selected backend=reference\n"
    assert CLOCK.search(buffer.getvalue()) is None


def test_timestamps_are_opt_in_and_do_appear_when_asked() -> None:
    """Negative control for the test above, whose assertion would also pass if the
    formatter had stopped working."""
    buffer = io.StringIO()
    configure(stream=buffer, timestamps=True)
    log_event(get_logger("probe"), EventKind.BACKEND_SELECTED, backend="reference")
    assert CLOCK.search(buffer.getvalue()) is not None


def test_field_order_does_not_change_the_record() -> None:
    """Two call sites recording the same decision must digest alike, under the
    canonicalisation ``hash_json`` applies to configs."""
    with capture() as first:
        log_event(get_logger("probe"), EventKind.DIGEST, artefact="model", sha256="ab")
    with capture() as second:
        log_event(get_logger("probe"), EventKind.DIGEST, sha256="ab", artefact="model")

    assert first.events == second.events
    assert first.digest() == second.digest()


def test_the_digest_is_over_decisions_not_over_when_they_were_taken() -> None:
    def record() -> tuple[str, list[dict[str, object]]]:
        with capture() as recorder:
            log = get_logger("probe")
            log_event(log, EventKind.BACKEND_SELECTED, backend="reference")
            log_event(log, EventKind.DEGENERATE, case="undefined_margin", k=1)
        return recorder.digest(), recorder.to_list()

    assert record() == record()


def test_the_digest_is_sensitive_to_order() -> None:
    """Selecting the native backend after a fallback is a different run from
    selecting it first, so the record must distinguish them."""
    log = get_logger("probe")
    with capture() as forward:
        log_event(log, EventKind.BACKEND_SELECTED, backend="reference")
        log_event(log, EventKind.DEGENERATE, case="degenerate_query")
    with capture() as backward:
        log_event(log, EventKind.DEGENERATE, case="degenerate_query")
        log_event(log, EventKind.BACKEND_SELECTED, backend="reference")

    assert forward.digest() != backward.digest()


def test_a_subnormal_field_is_rendered_without_loss() -> None:
    """A margin of 5e-324 and a margin of 0.0 separate a near-tie from an exact
    tie; a rounded rendering erases the difference."""
    event = Event.build(EventKind.DEGENERATE, {"margin": 5e-324})
    assert event.render() == "degenerate margin=5e-324"
    assert Event.build(EventKind.DEGENERATE, {"margin": 0.0}).render() == "degenerate margin=0.0"


def test_a_multiline_value_cannot_forge_extra_fields() -> None:
    """The native loader's fallback reason is a paragraph. One event is one line,
    or a reader cannot count events."""
    event = Event.build(EventKind.BACKEND_SELECTED, {"reason": "not built\nrun cmake x=1"})
    rendered = event.render()

    assert rendered == 'backend_selected reason="not built run cmake x=1"'
    # Quoted, so the embedded "x=1" stays inside one value and a reader
    # splitting the line still counts one key.
    assert shlex.split(rendered) == ["backend_selected", "reason=not built run cmake x=1"]


def test_free_text_records_stay_out_of_the_run_record() -> None:
    """Only the closed vocabulary is digestible; a third-party library logging a
    path or an address must not land in a hash."""
    with capture() as recorder:
        get_logger("probe").info("an unstructured message from somewhere")
        log_event(get_logger("probe"), EventKind.DIGEST, artefact="model", sha256="ab")

    assert [e.kind for e in recorder.events] == [EventKind.DIGEST]


def test_capture_restores_the_logger_afterwards() -> None:
    logger = logging.getLogger(ROOT_NAME)
    before, level = list(logger.handlers), logger.level
    with pytest.raises(RuntimeError), capture():
        raise RuntimeError("boom")
    assert logger.handlers == before
    assert logger.level == level


# ---------------------------------------------------------------------------
# Warnings still abort the suite
# ---------------------------------------------------------------------------
def test_configuring_logging_does_not_capture_warnings() -> None:
    """``logging.captureWarnings`` would reroute ``showwarning`` process-wide.
    Under ``filterwarnings = ["error"]`` these must abort the run instead of being
    filed."""
    configure(stream=io.StringIO())
    with pytest.raises(UserWarning):
        warnings.warn("load-bearing", UserWarning, stacklevel=1)


def test_a_real_diagnostic_still_raises_with_logging_configured() -> None:
    """End to end: the tie-group diagnostics are how the suite notices a degenerate
    tau, and logging must not soften them."""
    from tfidf_stability.ranking.tie_groups import TieGroupIndex
    from tfidf_stability.utils.validation import TauExceedsScoreRangeWarning

    configure(stream=io.StringIO())
    with pytest.raises(TauExceedsScoreRangeWarning):
        TieGroupIndex.build([1.0, 0.5, 0.0], tau=2.0)


# ---------------------------------------------------------------------------
# The wired-in call sites
# ---------------------------------------------------------------------------
def test_the_selected_backend_is_recorded() -> None:
    from tfidf_stability._native import log_backend_selection, native_available

    with capture() as recorder:
        log_backend_selection()

    (event,) = recorder.of_kind(EventKind.BACKEND_SELECTED)
    expected = "native" if native_available() else "reference"
    assert dict(event.fields)["backend"] == expected


def test_building_a_corpus_records_its_policy_degeneracies_and_digests(tmp_path: Path) -> None:
    """The three questions asked of a published number: what summed it, what was
    degenerate in the data, and what came out."""
    from tfidf_stability.cli.main import main

    with capture() as recorder:
        assert main(["build-corpus", str(CORPUS), "-o", str(tmp_path / "mini.tfsx")]) == 0

    kinds = {e.kind for e in recorder.events}
    assert EventKind.REDUCTION_POLICY in kinds
    assert EventKind.DIGEST in kinds
    # The mini corpus contains an all-stopword document that embeds to zero.
    assert EventKind.DEGENERATE in kinds

    artefacts = {dict(e.fields)["artefact"] for e in recorder.of_kind(EventKind.DIGEST)}
    assert artefacts == {"model", "vocabulary", "manifest"}


def test_the_cli_logs_nothing_unless_asked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from tfidf_stability.cli.main import main

    assert main(["build-corpus", str(CORPUS), "-o", str(tmp_path / "a.tfsx")]) == 0
    assert capsys.readouterr().err == ""
    assert logging.getLogger(ROOT_NAME).handlers == []


def test_the_cli_log_level_flag_configures_the_package_logger(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from tfidf_stability.cli.main import main

    argv = ["--log-level", "info", "build-corpus", str(CORPUS), "-o", str(tmp_path / "b.tfsx")]
    assert main(argv) == 0

    err = capsys.readouterr().err
    assert "backend_selected" in err
    assert "reduction_policy" in err
    assert CLOCK.search(err) is None
