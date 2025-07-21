"""Asynchronous ingestion of run logs into PostgreSQL."""

from __future__ import annotations

import asyncio
import io
import json
import os
from datetime import datetime
from typing import AsyncIterator, Dict, Iterable

import psycopg2
import psycopg2.extras
import zstandard as zstd

from ..config import Config

MANIFEST_NAME = "ingestion_manifest.json"

LOG_TABLE_MAP = {
    # --- Events Table ---
    "event_log.jsonl": "events",
    "node_emergence_log.jsonl": "events",
    "collapse_chain_log.jsonl": "events",
    "bridge_rupture_log.jsonl": "events",
    "bridge_reformation_log.jsonl": "events",
    "law_drift_log.jsonl": "events",
    # --- Tick Events Table ---
    "tick_emission_log.jsonl": "tick_events",
    "tick_propagation_log.jsonl": "tick_events",
    "tick_delivery_log.jsonl": "tick_events",
    "tick_drop_log.jsonl": "tick_events",
    "propagation_failure_log.jsonl": "tick_events",
    # --- Node State History Table ---
    "node_state_log.jsonl": "node_state_history",
    "coherence_log.jsonl": "node_state_history",
    "decoherence_log.jsonl": "node_state_history",
    "law_wave_log.jsonl": "node_state_history",
    "classicalization_map.jsonl": "node_state_history",
    # --- Bridge State History Table ---
    "bridge_state_log.jsonl": "bridge_state_history",
    "bridge_decay_log.jsonl": "bridge_state_history",
    # --- System State History Table ---
    # Assuming "system_state_log.jsonl" is the new name for "structural_growth_log.jsonl"
    "system_state_log.jsonl": "system_state_history",
}


def _iter_json(path: str) -> Iterable[dict]:
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


async def _write_records(
    conn_params: Dict, table: str, records: Iterable[dict]
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


def _load_manifest(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def _save_manifest(path: str, data: Dict) -> None:
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
    for name in os.listdir(log_dir):
        table = LOG_TABLE_MAP.get(name.replace(".zst", ""))
        if not table:
            continue
        records = list(_iter_json(os.path.join(log_dir, name)))
        if not records:
            continue
        tasks.append(_write_records(Config.database, table, records))

    if tasks:
        await asyncio.gather(*tasks)

    manifest[run_id] = {
        "timestamp": datetime.utcnow().isoformat(),
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
