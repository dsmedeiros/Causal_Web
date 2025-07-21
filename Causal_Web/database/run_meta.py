import json
import os
from datetime import datetime
from typing import Any, Dict

import psycopg2

from ..config import Config


def _graph_metadata(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {
            "topology_type": None,
            "was_generated": False,
            "node_count": 0,
            "edge_count": 0,
        }
    with open(path) as fh:
        data = json.load(fh)
    meta = data.get("metadata", {})
    nodes = data.get("nodes", {})
    if isinstance(nodes, dict):
        node_count = len(nodes)
    else:
        node_count = len(nodes)
    edges = data.get("edges", {})
    if isinstance(edges, dict):
        edge_count = sum(
            len(v) if isinstance(v, list) else len(getattr(v, "keys", lambda: [])())
            for v in edges.values()
        )
    else:
        edge_count = len(edges)
    return {
        "topology_type": meta.get("topology"),
        "was_generated": bool(meta.get("generated")),
        "node_count": node_count,
        "edge_count": edge_count,
    }


def record_run(
    run_id: str, config_path: str, graph_path: str, archive_path: str
) -> None:
    metadata = _graph_metadata(graph_path)
    with open(config_path) as fh:
        cfg = json.load(fh)
    data = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "description": cfg.get("description"),
        "seed": cfg.get("random_seed"),
        "tick_limit": cfg.get("tick_limit"),
        "topology_type": metadata["topology_type"],
        "was_generated": metadata["was_generated"],
        "node_count": metadata["node_count"],
        "edge_count": metadata["edge_count"],
        "archive_path": archive_path,
    }
    conn = psycopg2.connect(**Config.database)
    try:
        with conn.cursor() as cur:
            cols = ",".join(data.keys())
            placeholders = ",".join(["%s"] * len(data))
            cur.execute(
                f"INSERT INTO runs ({cols}) VALUES ({placeholders}) ON CONFLICT (run_id) DO NOTHING",
                list(data.values()),
            )
        conn.commit()
    finally:
        conn.close()
