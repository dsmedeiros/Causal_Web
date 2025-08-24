import json
import os
import stat

from Causal_Web.engine.stream.auth import default_session_file, write_session_bundle


def test_bundle_written_with_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    expected = default_session_file()
    bundle, path = write_session_bundle("127.0.0.1", 8765, "tok", 60)
    assert path == expected
    data = json.loads(path.read_text())
    assert data["host"] == "127.0.0.1"
    assert data["port"] == 8765
    assert data["token"] == "tok"
    if os.name != "nt":
        mode = path.stat().st_mode & 0o777
        assert mode == 0o600
