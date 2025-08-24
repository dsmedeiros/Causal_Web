from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Tuple

from Causal_Web.engine.stream.auth import default_session_file


def load_session_bundle(path: str | os.PathLike[str]) -> dict:
    """Return the parsed JSON session bundle at ``path``."""

    with open(path) as f:
        return json.load(f)


def resolve_connection_info(
    ws_url: str | None = None,
    token: str | None = None,
    token_file: str | None = None,
    ws_host: str | None = None,
    ws_port: int | None = None,
) -> Tuple[str, str]:
    """Resolve WebSocket URL and token using CLI/env precedence."""
    ws_url = ws_url or os.getenv("CW_WS_URL")
    token = token or os.getenv("CW_SESSION_TOKEN")
    token_file = token_file or os.getenv("CW_SESSION_FILE")
    ws_host = ws_host or os.getenv("CW_WS_HOST", "127.0.0.1")
    ws_port = int(os.getenv("CW_WS_PORT", str(ws_port or 8765)))

    if ws_url:
        if token is None and token_file:
            bundle = load_session_bundle(token_file)
            token = bundle.get("token", "")
        return ws_url, token or ""

    if token:
        return f"ws://{ws_host}:{ws_port}", token

    bundle_path = token_file or default_session_file()
    bundle = load_session_bundle(bundle_path)
    url = f"ws://{bundle['host']}:{bundle['port']}"
    return url, bundle.get("token", "")
