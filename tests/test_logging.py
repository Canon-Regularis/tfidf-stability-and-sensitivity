"""Logging as a provenance channel: three properties of the package logger.

Importing installs no handler, captured output carries no timestamp because
events are hashed into the run record, and warnings are not rerouted through
logging. Rationale: docs/engineering_notes.md#logging-as-a-provenance-channel
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
    """Restore the package logger's handlers and level after each test.

    Autouse: a leaked handler makes a later test pass or fail for reasons
    unrelated to what it asserts.
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
    """A fresh interpreter importing the package has no handler and no level.

    By collection time the package is already imported and pytest's logging
    plugin has a handler on the root logger, so only a subprocess can check.
    """
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
    """Emitting with no handler installed does not raise.

    Records at ``WARNING`` and above then reach ``logging.lastResort``; a
    NullHandler would silence those diagnostics in an unconfigured run.
    """
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
    """The root logger's handlers, level and filters survive ``configure``.

    A handler list comparison alone misses two of them: `basicConfig()` or a
    `setLevel` on the root logger changes what every unrelated library emits
    while the handler list stays identical.
    """
    root = logging.getLogger()
    restore_to = root.level

    # Pinned to a known level first. Comparing against whatever the level
    # happens to be makes the test order-dependent: an earlier `configure()` in
    # this file would already have moved it, and the comparison then sees no
    # change from a second move.
    root.setLevel(logging.WARNING)
    try:
        before_handlers = list(root.handlers)
        before_filters = list(root.filters)
        before_propagate = root.propagate

        configure(stream=io.StringIO())

        assert root.handlers == before_handlers, "no handler added to or removed from the root"
        assert root.level == logging.WARNING, (
            f"the root logger's level moved to {logging.getLevelName(root.level)}, "
            f"which changes what every other library in the process emits"
        )
        assert root.filters == before_filters
        assert root.propagate is before_propagate
    finally:
        root.setLevel(restore_to)


def test_reset_leaves_a_handler_the_application_added() -> None:
    """``reset`` removes only the handler ``configure`` installed.

    Removing a handler the application added is the same overreach as
    configuring the root logger.
    """
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
    """Timestamps appear when requested; negative control for the test above.

    The no-timestamp assertion would also pass with a broken formatter.
    """
    buffer = io.StringIO()
    configure(stream=buffer, timestamps=True)
    log_event(get_logger("probe"), EventKind.BACKEND_SELECTED, backend="reference")
    assert CLOCK.search(buffer.getvalue()) is not None


def test_field_order_does_not_change_the_record() -> None:
    """Field order does not change the event or its digest.

    Two call sites recording the same decision digest alike, under the
    canonicalisation ``hash_json`` applies to configs.
    """
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
    """Reversing the event order changes the digest.

    Selecting the native backend after a fallback is a different run from
    selecting it first, so the record must distinguish them.
    """
    log = get_logger("probe")
    with capture() as forward:
        log_event(log, EventKind.BACKEND_SELECTED, backend="reference")
        log_event(log, EventKind.DEGENERATE, case="degenerate_query")
    with capture() as backward:
        log_event(log, EventKind.DEGENERATE, case="degenerate_query")
        log_event(log, EventKind.BACKEND_SELECTED, backend="reference")

    assert forward.digest() != backward.digest()


def test_a_subnormal_field_is_rendered_without_loss() -> None:
    """A subnormal float renders exactly, distinct from zero.

    A margin of 5e-324 and a margin of 0.0 separate a near-tie from an exact
    tie; a rounded rendering erases the difference.
    """
    event = Event.build(EventKind.DEGENERATE, {"margin": 5e-324})
    assert event.render() == "degenerate margin=5e-324"
    assert Event.build(EventKind.DEGENERATE, {"margin": 0.0}).render() == "degenerate margin=0.0"


def test_a_multiline_value_cannot_forge_extra_fields() -> None:
    """A value with a newline renders as one quoted field on one line.

    The native loader's fallback reason is a paragraph; one event is one
    line, or a reader cannot count events.
    """
    event = Event.build(EventKind.BACKEND_SELECTED, {"reason": "not built\nrun cmake x=1"})
    rendered = event.render()

    assert rendered == 'backend_selected reason="not built run cmake x=1"'
    # Quoted, so the embedded "x=1" stays inside one value and a reader
    # splitting the line still counts one key.
    assert shlex.split(rendered) == ["backend_selected", "reason=not built run cmake x=1"]


def test_free_text_records_stay_out_of_the_run_record() -> None:
    """Free-text log records are excluded from the captured events.

    Only the closed vocabulary is digestible; a third-party library logging a
    path or an address must not land in a hash.
    """
    with capture() as recorder:
        get_logger("probe").info("an unstructured message from somewhere")
        log_event(get_logger("probe"), EventKind.DIGEST, artefact="model", sha256="ab")

    assert [e.kind for e in recorder.events] == [EventKind.DIGEST]


def test_capture_restores_the_logger_afterwards() -> None:
    logger = logging.getLogger(ROOT_NAME)
    before, level = list(logger.handlers), logger.level
    with pytest.raises(RuntimeError, match="boom"), capture():
        raise RuntimeError("boom")
    assert logger.handlers == before
    assert logger.level == level


# ---------------------------------------------------------------------------
# Warnings still abort the suite
# ---------------------------------------------------------------------------
def test_configuring_logging_does_not_capture_warnings() -> None:
    """A warning still raises as an error after ``configure``.

    ``logging.captureWarnings`` would reroute ``showwarning`` process-wide;
    under ``filterwarnings = ["error"]`` warnings must abort the run instead.
    """
    configure(stream=io.StringIO())
    with pytest.raises(UserWarning, match="load-bearing"):
        warnings.warn("load-bearing", UserWarning, stacklevel=1)


def test_a_real_diagnostic_still_raises_with_logging_configured() -> None:
    """A tie-group diagnostic warning still raises with logging configured.

    The tie-group diagnostics are how the suite notices a degenerate tau, and
    logging must not soften them.
    """
    from tfidf_stability.ranking.tie_groups import TieGroupIndex
    from tfidf_stability.utils.validation import TauExceedsScoreRangeWarning

    configure(stream=io.StringIO())
    with pytest.raises(TauExceedsScoreRangeWarning, match="covers the entire score range"):
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
    """Building a corpus records its policy, degeneracies and digests.

    These answer three questions asked of a published number: what summed it,
    what was degenerate in the data, and what came out.
    """
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


def test_the_root_logger_is_returned_for_a_name_that_is_already_it() -> None:
    """The root name and ``None`` both resolve to the root logger itself.

    A call from the package's `__init__` passes the root name as `__name__`;
    reparenting it would give `tfidf_stability.tfidf_stability`, a name that
    matches no module.
    """
    root = logging.getLogger(ROOT_NAME)
    assert get_logger(ROOT_NAME) is root
    assert get_logger(None) is root
    assert get_logger() is root


def test_a_name_from_inside_the_package_is_used_as_written() -> None:
    """A dotted name inside the package is kept as written.

    The dotted path is what makes per-module filtering possible; rewriting it
    would flatten every module onto one name.
    """
    assert get_logger(f"{ROOT_NAME}.ranking.ranker").name == f"{ROOT_NAME}.ranking.ranker"


def test_a_name_from_outside_the_package_is_reparented_under_the_root() -> None:
    """A name outside the package is reparented under the root name.

    Nothing may emit outside the subtree `configure` controls, or a record
    would escape the handler and the formatter that makes it canonical.
    """
    assert get_logger("scripts.benchmark").name == f"{ROOT_NAME}.scripts.benchmark"


def test_json_scalars_render_as_json_not_as_python() -> None:
    """`None` and `bool` render as JSON scalars, not as Python reprs.

    `str(None)` is "None" and `str(True)` is "True"; neither is a value a JSON
    reader can take.
    """
    event = Event.build(EventKind.BACKEND_SELECTED, {"reason": None, "native": True})
    rendered = event.render()
    assert "reason=null" in rendered
    assert "native=true" in rendered
    assert "None" not in rendered
    assert "True" not in rendered


def test_an_integer_field_keeps_its_bare_form() -> None:
    """An integer field renders in its bare form.

    This is the value on which JSON and Python agree, so the rendering branch
    is shown to be about the format and not about the type.
    """
    assert "count=7" in Event.build(EventKind.DIGEST, {"count": 7}).render()


def test_a_non_ascii_field_is_quoted_but_not_escaped() -> None:
    """A non-ASCII value is quoted but not escaped.

    The log is UTF-8 and read by people; an ASCII escape sequence would make
    a document identifier or a token unrecognisable in the log.
    """
    event = Event.build(EventKind.DEGENERATE, {"token": "café au lait"})
    rendered = event.render()
    assert rendered == 'degenerate token="café au lait"'
    assert chr(92) not in rendered, "no escape sequences of any kind"


def test_of_kind_accepts_the_string_spelling_of_a_kind() -> None:
    """Both spellings of a kind select the same events.

    `EventKind` is a `str` enum and `of_kind` compares with `is`, so an uncoerced
    string selects nothing. A kind never recorded returns an empty list.
    """
    log = get_logger()
    with capture() as recorder:
        log_event(log, EventKind.DIGEST, artefact="vocabulary", value="abc")
        log_event(log, EventKind.DIGEST, artefact="model", value="def")
        log_event(log, EventKind.BACKEND_SELECTED, backend="reference")

    assert len(recorder.of_kind(EventKind.DIGEST)) == 2
    assert recorder.of_kind("digest") == recorder.of_kind(EventKind.DIGEST)
    assert recorder.of_kind("backend_selected") == recorder.of_kind(EventKind.BACKEND_SELECTED)
    assert recorder.of_kind(EventKind.DEGENERATE) == []


def test_of_kind_refuses_a_kind_that_names_nothing() -> None:
    """A misspelled kind raises ValueError instead of returning nothing."""
    log = get_logger()
    with capture() as recorder:
        log_event(log, EventKind.DIGEST, artefact="vocabulary", value="abc")

    with pytest.raises(ValueError, match="is not a valid EventKind"):
        recorder.of_kind("digests")
