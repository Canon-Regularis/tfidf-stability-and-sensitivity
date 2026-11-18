"""scripts/fetch_data.py: dataset selection, pins and the parse after download.

No test opens a connection: ``urlopen`` is replaced by a fake that serves
fixed bytes and records the URL it was asked for.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

from tfidf_stability.datasets import beir

REPO = Path(__file__).resolve().parents[1]


def _fetch_script() -> ModuleType:
    """scripts/fetch_data.py as a module. Local by house convention."""
    path = REPO / "scripts" / "fetch_data.py"
    spec = importlib.util.spec_from_file_location("fetch_data_main_under_test", path)
    assert spec is not None, "the script must be loadable as a module"
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _scifact_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("scifact/corpus.jsonl", '{"_id": "d1", "title": "t", "text": "a"}\n')
        archive.writestr("scifact/queries.jsonl", '{"_id": "q1", "text": "a"}\n')
        archive.writestr("scifact/qrels/test.tsv", "query-id\tcorpus-id\tscore\r\nq1\td1\t1\r\n")
    return buffer.getvalue()


class _Response:
    """The slice of urlopen's result that `_download` uses."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self.headers = {"Content-Length": str(len(payload))}

    def read(self, size: int) -> bytes:
        chunk, self._payload = self._payload[:size], self._payload[size:]
        return chunk

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _serve(monkeypatch: pytest.MonkeyPatch, fetch: ModuleType, payload: bytes) -> list[str]:
    """Serve ``payload`` for every request; the returned list collects the URLs."""
    asked: list[str] = []

    def urlopen(request: object) -> _Response:
        asked.append(request.full_url)  # type: ignore[attr-defined]
        return _Response(payload)

    monkeypatch.setattr(fetch.urllib.request, "urlopen", urlopen, raising=True)
    return asked


def test_a_beir_download_matching_its_pin_is_placed_and_parsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fetch = _fetch_script()
    payload = _scifact_zip()
    asked = _serve(monkeypatch, fetch, payload)
    monkeypatch.setitem(beir.PINNED_SHA256, "scifact", hashlib.sha256(payload).hexdigest())
    dest = tmp_path / "scifact.zip"

    assert fetch.main(["--dataset", "beir_scifact", "--output", str(dest)]) == 0

    assert asked == [beir.BEIR_URL.format(variant="scifact")], "the dataset picks the URL"
    assert dest.read_bytes() == payload
    out = capsys.readouterr().out
    assert "digest matches the pin" in out
    assert "parsed: 1 documents, 1 judged queries, 1 judgements" in out


def test_a_beir_download_off_its_pin_is_not_placed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fetch = _fetch_script()
    _serve(monkeypatch, fetch, _scifact_zip())
    monkeypatch.setitem(beir.PINNED_SHA256, "scifact", "0" * 64)
    dest = tmp_path / "scifact.zip"

    assert fetch.main(["--dataset", "beir_scifact", "--output", str(dest)]) == 1

    assert not dest.exists()
    err = capsys.readouterr().err
    assert "DIGEST MISMATCH" in err
    assert "The scifact archive at" in err, "the message names the dataset, not GroupLens"
    assert f"Nothing was written to {dest}" in err


def test_without_a_pin_the_line_to_paste_names_the_beir_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fetch = _fetch_script()
    payload = _scifact_zip()
    _serve(monkeypatch, fetch, payload)
    monkeypatch.delitem(beir.PINNED_SHA256, "scifact")

    assert fetch.main(["--dataset", "beir_scifact", "--output", str(tmp_path / "s.zip")]) == 0

    out = capsys.readouterr().out
    assert "src/tfidf_stability/datasets/beir.py, in PINNED_SHA256:" in out
    assert f'"scifact": "{hashlib.sha256(payload).hexdigest()}",' in out


def test_an_archive_already_present_off_its_pin_is_reported_not_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fetch = _fetch_script()
    asked = _serve(monkeypatch, fetch, b"never served")
    monkeypatch.setitem(beir.PINNED_SHA256, "nfcorpus", "0" * 64)
    dest = tmp_path / "nfcorpus.zip"
    dest.write_bytes(b"an older archive")

    assert fetch.main(["--dataset", "beir_nfcorpus", "--output", str(dest)]) == 1

    assert asked == [], "an archive already present is not downloaded again without --force"
    assert dest.read_bytes() == b"an older archive"
    err = capsys.readouterr().err
    assert "was not downloaded again and does not match the pin" in err
    assert "The nfcorpus archive at" not in err, "no upstream was contacted"


def test_a_mismatched_download_names_the_url_actually_fetched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fetch = _fetch_script()
    _serve(monkeypatch, fetch, _scifact_zip())
    monkeypatch.setitem(beir.PINNED_SHA256, "scifact", "0" * 64)
    argv = ["--dataset", "beir_scifact", "--url", "https://example.invalid/mirror.zip"]

    assert fetch.main([*argv, "--output", str(tmp_path / "s.zip")]) == 1

    assert "The scifact archive at https://example.invalid/mirror.zip" in capsys.readouterr().err


def test_an_unpinned_archive_that_does_not_parse_is_never_offered_as_a_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fetch = _fetch_script()
    _serve(monkeypatch, fetch, b"<html>proxy error</html>")
    monkeypatch.delitem(beir.PINNED_SHA256, "scifact")

    assert fetch.main(["--dataset", "beir_scifact", "--output", str(tmp_path / "s.zip")]) == 1

    captured = capsys.readouterr()
    assert "is not a valid archive: not a zip archive" in captured.err
    assert "No digest is pinned yet" not in captured.out


def test_every_registered_download_resolves_its_own_url_and_destination() -> None:
    fetch = _fetch_script()

    assert set(fetch.SOURCES) == {"movielens_small", "beir_scifact", "beir_nfcorpus"}
    assert fetch.SOURCES["movielens_small"].destination.name == "ml-latest-small.zip"
    assert fetch.SOURCES["beir_nfcorpus"].url.endswith("/nfcorpus.zip")
    assert fetch.SOURCES["beir_nfcorpus"].destination == REPO / "data" / "raw" / "beir-nfcorpus.zip"


def test_print_digest_reads_an_existing_file_without_downloading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fetch = _fetch_script()
    asked = _serve(monkeypatch, fetch, b"never served")
    dest = tmp_path / "any.zip"
    dest.write_bytes(b"bytes on disk")

    assert fetch.main(["--dataset", "beir_scifact", "--output", str(dest), "--print-digest"]) == 0

    assert asked == []
    assert capsys.readouterr().out.strip() == hashlib.sha256(b"bytes on disk").hexdigest()
