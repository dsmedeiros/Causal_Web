"""Database initialization utilities."""

from __future__ import annotations

from typing import Any

import psycopg2

SCHEMA_SQL = """
-- Table 1: events
CREATE TABLE IF NOT EXISTS events (
    log_id TEXT PRIMARY KEY,
    tick INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    correlation_id TEXT,
    event_type TEXT NOT NULL,
    subject_id TEXT,
    payload JSONB
);
CREATE INDEX IF NOT EXISTS idx_events_tick ON events (tick);
CREATE INDEX IF NOT EXISTS idx_events_type ON events (event_type);
CREATE INDEX IF NOT EXISTS idx_events_subject ON events (subject_id);
CREATE INDEX IF NOT EXISTS idx_events_payload ON events USING GIN (payload);

-- Table 2: tick_events
CREATE TABLE IF NOT EXISTS tick_events (
    log_id TEXT PRIMARY KEY,
    tick INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    correlation_id TEXT,
    event_type TEXT NOT NULL,
    subject_id TEXT,
    payload JSONB
);
CREATE INDEX IF NOT EXISTS idx_tick_events_tick ON tick_events (tick);
CREATE INDEX IF NOT EXISTS idx_tick_events_subject ON tick_events (subject_id);
CREATE INDEX IF NOT EXISTS idx_tick_events_payload ON tick_events USING GIN (payload);

-- Table 3: node_state_history
CREATE TABLE IF NOT EXISTS node_state_history (
    log_id TEXT PRIMARY KEY,
    tick INTEGER NOT NULL,
    node_id TEXT NOT NULL,
    is_classicalized BOOLEAN,
    coherence REAL,
    decoherence REAL,
    frequency REAL,
    payload JSONB
);
CREATE INDEX IF NOT EXISTS idx_node_state_tick_node ON node_state_history (tick, node_id);
CREATE INDEX IF NOT EXISTS idx_node_state_payload ON node_state_history USING GIN (payload);

-- Table 4: bridge_state_history
CREATE TABLE IF NOT EXISTS bridge_state_history (
    log_id TEXT PRIMARY KEY,
    tick INTEGER NOT NULL,
    bridge_id TEXT NOT NULL,
    strength REAL,
    trust_score REAL,
    status TEXT,
    payload JSONB
);
CREATE INDEX IF NOT EXISTS idx_bridge_state_tick_bridge ON bridge_state_history (tick, bridge_id);
CREATE INDEX IF NOT EXISTS idx_bridge_state_payload ON bridge_state_history USING GIN (payload);

-- Table 5: system_state_history
CREATE TABLE IF NOT EXISTS system_state_history (
    log_id TEXT PRIMARY KEY,
    tick INTEGER NOT NULL,
    node_count INTEGER,
    edge_count INTEGER,
    avg_coherence REAL,
    payload JSONB
);
CREATE INDEX IF NOT EXISTS idx_system_state_tick ON system_state_history (tick);
CREATE INDEX IF NOT EXISTS idx_system_state_payload ON system_state_history USING GIN (payload);

-- Table 6: runs
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    description TEXT,
    seed INTEGER,
    tick_limit INTEGER,
    topology_type TEXT,
    was_generated BOOLEAN,
    node_count INTEGER,
    edge_count INTEGER,
    archive_path TEXT
);

-- Table 7: causal_analysis_results
CREATE TABLE IF NOT EXISTS causal_analysis_results (
    log_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    result_type TEXT NOT NULL,
    payload JSONB
);
CREATE INDEX IF NOT EXISTS idx_analysis_run_id ON causal_analysis_results (run_id);
CREATE INDEX IF NOT EXISTS idx_analysis_payload ON causal_analysis_results USING GIN (payload);
"""


def initialize_database(config: dict[str, Any]) -> None:
    """Create required PostgreSQL tables and indexes if they do not exist.

    Parameters
    ----------
    config:
        Mapping containing connection parameters such as ``host``, ``port``,
        ``user``, ``password`` and ``dbname``.
    """

    conn = None
    try:
        conn = psycopg2.connect(**config)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
    except Exception as exc:  # pragma: no cover - networked operation
        raise RuntimeError("Failed to initialize database") from exc
    finally:
        if conn is not None:
            conn.close()
