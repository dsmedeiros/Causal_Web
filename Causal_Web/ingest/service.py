"""Asynchronous ingestion of run logs into PostgreSQL."""

from __future__ import annotations

import asyncio
import io
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable

import psycopg2
import psycopg2.extras
import zstandard as zstd

from ..config import Config

MANIFEST_NAME = "ingestion_manifest.json"

LOG_TABLE_MAP = {
    # --- Events Table ---
    "event_log": "events",
    "node_emergence_log": "events",
    "collapse_chain_log": "events",
    "bridge_rupture_log": "events",
    "bridge_reformation_log": "events",
    "law_drift_log": "events",
    # --- Tick Events Table ---
    "tick_emission_log": "tick_events",
    "tick_propagation_log": "tick_events",
    "tick_delivery_log": "tick_events",
    "tick_drop_log": "tick_events",
    "propagation_failure_log": "tick_events",
    # --- Node State History Table ---
    "node_state_log": "node_state_history",
    "coherence_log": "node_state_history",
    "decoherence_log": "node_state_history",
    "classicalization_map": "node_state_history",
    # --- Bridge State History Table ---
    "bridge_state": "bridge_state_history",
    "bridge_decay_log": "bridge_state_history",
    # --- System State History Table ---
    "structural_growth_log": "system_state_history",
}


def _iter_json(path: str) -> Iterable[dict[str, Any]]:
    """Yield JSON objects from ``path`` supporting optional ``.zst`` compression."""

    if path.endswith(".zst"):
        with open(path, "rb") as fh:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                wrapper = io.TextIOWrapper(reader, encoding="utf-8")
                for line in wrapper:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
    else:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _group_records(path: str, field: str) -> Dict[str, list[dict[str, Any]]]:
    """Return records from ``path`` grouped by ``field``."""

    grouped: Dict[str, list[dict[str, Any]]] = {}
    if not os.path.exists(path):
        return grouped
    for rec in _iter_json(path):
        key = rec.get(field)
        if key is None:
            continue
        grouped.setdefault(key, []).append(rec)
    return grouped


async def _write_records(
    conn_params: Dict[str, Any], table: str, records: Iterable[dict[str, Any]]
) -> None:
    """Insert ``records`` into ``table`` using ``psycopg2``."""

    def _insert() -> None:
        conn = psycopg2.connect(**conn_params)
        try:
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur,
                    f"INSERT INTO {table} (payload) VALUES %s",
                    [(json.dumps(r),) for r in records],
                )
            conn.commit()
        finally:
            conn.close()

    await asyncio.to_thread(_insert)


def _load_manifest(path: str) -> Dict[str, Any]:
    """Return the JSON manifest at ``path`` if it exists."""

    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def _save_manifest(path: str, data: Dict[str, Any]) -> None:
    """Write ``data`` as JSON to ``path``."""

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)


async def ingest_run(run_dir: str) -> None:
    """Ingest all logs for ``run_dir`` into PostgreSQL."""

    manifest_path = os.path.join(Config.ingest_dir, MANIFEST_NAME)
    manifest = _load_manifest(manifest_path)
    run_id = os.path.basename(run_dir)
    if run_id in manifest:
        return

    log_dir = os.path.join(run_dir, "logs")
    if not os.path.isdir(log_dir):
        return

    tasks = []

    events = _group_records(os.path.join(log_dir, "events_log.jsonl"), "event_type")
    ticks = _group_records(os.path.join(log_dir, "ticks_log.jsonl"), "label")
    phen = _group_records(os.path.join(log_dir, "phenomena_log.jsonl"), "label")

    for key, records in {**events, **ticks, **phen}.items():
        table = LOG_TABLE_MAP.get(key)
        if not table or not records:
            continue
        tasks.append(_write_records(Config.database, table, records))

    if tasks:
        await asyncio.gather(*tasks)

    manifest[run_id] = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "ingested",
    }
    _save_manifest(manifest_path, manifest)


async def ingest_runs() -> None:
    """Scan ``Config.runs_dir`` for new runs and ingest them."""

    if not os.path.isdir(Config.runs_dir):
        return
    for entry in sorted(os.listdir(Config.runs_dir)):
        run_path = os.path.join(Config.runs_dir, entry)
        await ingest_run(run_path)
