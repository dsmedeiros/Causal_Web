"""Local results registry backed by SQLite.

This module offers a lightweight interface for storing and querying
experiment results.  The registry keeps generic run metadata alongside
optional internal statistics emitted by the MCTS-H optimizer.  Results
are stored in a single SQLite database so thousands of runs can be mined
quickly without scanning individual JSON files.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    run_id TEXT PRIMARY KEY,
    optimizer TEXT,
    promotion_rate REAL,
    residual REAL,
    proxy_full_corr REAL,
    mcts_nodes_expanded INTEGER,
    mcts_promotions INTEGER,
    mcts_bins_created INTEGER,
    mcts_frontier INTEGER,
    mcts_params TEXT
);
CREATE INDEX IF NOT EXISTS idx_results_optimizer ON results (optimizer);
CREATE INDEX IF NOT EXISTS idx_results_promotion ON results (promotion_rate);
CREATE INDEX IF NOT EXISTS idx_results_residual ON results (residual);
"""


# ---------------------------------------------------------------------------


def connect(path: Path) -> sqlite3.Connection:
    """Return a connection to ``path`` initialising the schema if required."""

    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.executescript(SCHEMA)
    return conn


# ---------------------------------------------------------------------------


def insert(
    conn: sqlite3.Connection,
    run_id: str,
    optimizer: str,
    promotion_rate: float,
    residual: float,
    proxy_full_corr: float,
    *,
    mcts_nodes_expanded: int = 0,
    mcts_promotions: int = 0,
    mcts_bins_created: int = 0,
    mcts_frontier: int = 0,
    mcts_params: Optional[dict[str, Any]] = None,
) -> None:
    """Insert a run result into the registry."""

    data = {
        "run_id": run_id,
        "optimizer": optimizer,
        "promotion_rate": float(promotion_rate),
        "residual": float(residual),
        "proxy_full_corr": float(proxy_full_corr),
        "mcts_nodes_expanded": int(mcts_nodes_expanded),
        "mcts_promotions": int(mcts_promotions),
        "mcts_bins_created": int(mcts_bins_created),
        "mcts_frontier": int(mcts_frontier),
        "mcts_params": json.dumps(mcts_params or {}),
    }
    cols = ",".join(data.keys())
    placeholders = ":" + ",:".join(data.keys())
    conn.execute(
        f"INSERT OR REPLACE INTO results ({cols}) VALUES ({placeholders})", data
    )
    conn.commit()


# ---------------------------------------------------------------------------


def query(
    conn: sqlite3.Connection,
    *,
    optimizer: Optional[str] = None,
    promotion_min: Optional[float] = None,
    promotion_max: Optional[float] = None,
    residual_max: Optional[float] = None,
    proxy_full_corr_min: Optional[float] = None,
) -> list[sqlite3.Row]:
    """Return rows matching the supplied filters."""

    clauses: list[str] = []
    params: list[Any] = []
    if optimizer:
        clauses.append("optimizer = ?")
        params.append(optimizer)
    if promotion_min is not None:
        clauses.append("promotion_rate >= ?")
        params.append(float(promotion_min))
    if promotion_max is not None:
        clauses.append("promotion_rate <= ?")
        params.append(float(promotion_max))
    if residual_max is not None:
        clauses.append("residual < ?")
        params.append(float(residual_max))
    if proxy_full_corr_min is not None:
        clauses.append("proxy_full_corr >= ?")
        params.append(float(proxy_full_corr_min))
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    q = f"SELECT * FROM results{where}"
    conn.row_factory = sqlite3.Row
    return list(conn.execute(q, params))


__all__ = ["connect", "insert", "query"]
