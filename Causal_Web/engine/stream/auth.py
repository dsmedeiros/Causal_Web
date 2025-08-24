from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple


def default_session_file() -> Path:
    """Return the platform-appropriate default session file path."""
    if os.name == "nt":
        base = os.getenv("LOCALAPPDATA") or Path.home() / "AppData" / "Local"
        return Path(base) / "CausalWeb" / "sessions" / "current.json"
    base = (
        os.getenv("XDG_RUNTIME_DIR")
        or os.getenv("XDG_CACHE_HOME")
        or Path.home() / ".cache"
    )
    return Path(base) / "causalweb" / "session.json"


def write_session_bundle(
    host: str,
    port: int,
    token: str,
    ttl: int,
    path: str | os.PathLike[str] | None = None,
) -> Tuple[Dict[str, Any], Path]:
    """Write the session bundle JSON with best-effort 0600 perms."""
    issued = int(time.time())
    bundle = {
        "v": 1,
        "host": host,
        "port": port,
        "token": token,
        "issued_at": issued,
        "expires_at": issued + ttl,
    }
    dest = Path(path) if path else default_session_file()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    data = json.dumps(bundle)
    if os.name == "nt":
        tmp.write_text(data)
        os.replace(tmp, dest)
    else:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(data)
        os.replace(tmp, dest)
    return bundle, dest
