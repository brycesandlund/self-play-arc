import pytest

from modal_runner import download_archive, reserve


def test_budget_refuses_overrun_and_duplicate_jobs(tmp_path):
    ledger = tmp_path / "ledger.json"
    reserve(ledger, "one", "L40S")
    with pytest.raises(ValueError):
        reserve(ledger, "one", "L40S")
    for i in range(15):
        reserve(ledger, f"run-{i}", "L40S")
    with pytest.raises(RuntimeError):
        reserve(ledger, "over-budget", "L40S")


def test_corrupt_download_retries_without_launching_compute(tmp_path):
    import hashlib
    import io
    import zipfile
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("summary.json", '{"ok":true}')
    good = buffer.getvalue()
    class Volume:
        calls = 0
        def read_file_into_fileobj(self, path, stream):
            self.calls += 1
            stream.write(b"broken download" if self.calls == 1 else good)
    volume = Volume()
    download_archive(volume, "test", tmp_path / "run", hashlib.sha256(good).hexdigest())
    assert volume.calls == 2
    assert (tmp_path / "run" / "summary.json").read_text() == '{"ok":true}'
