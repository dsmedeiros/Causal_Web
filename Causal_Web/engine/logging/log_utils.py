"""Helpers for runtime logging and exports."""

from __future__ import annotations

import json
import os
from typing import Optional

from ...config import Config
from ..models.graph import CausalGraph
from .logger import log_json, logger, log_manager
from ..services.sim_services import GlobalDiagnosticsService
from ..models.logging import StructuralGrowthLog, StructuralGrowthPayload

# The global graph instance is injected at runtime
_graph: CausalGraph | None = None


def attach_graph(graph: CausalGraph) -> None:
    global _graph
    _graph = graph


def log_curvature_per_tick(global_tick: int) -> None:
    assert _graph is not None
    log = {}
    for edge in _graph.edges:
        src = _graph.get_node(edge.source)
        tgt = _graph.get_node(edge.target)
        if not src or not tgt:
            continue
        df = abs(src.law_wave_frequency - tgt.law_wave_frequency)
        curved = edge.adjusted_delay(
            src.law_wave_frequency,
            tgt.law_wave_frequency,
            getattr(Config, "delay_density_scaling", 1.0),
            graph=_graph,
        )
        log[f"{edge.source}->{edge.target}"] = {
            "delta_f": round(df, 4),
            "curved_delay": round(curved, 4),
        }
    log_json(Config.output_path("curvature_log.json"), {str(global_tick): log})


def log_bridge_states(global_tick: int) -> None:
    assert _graph is not None
    snapshot = {
        b.bridge_id: {
            "active": b.active,
            "last_activation": b.last_activation,
            "last_rupture_tick": b.last_rupture_tick,
            "last_reform_tick": b.last_reform_tick,
            "coherence_at_reform": b.coherence_at_reform,
            "trust_score": b.trust_score,
            "reinforcement": b.reinforcement_streak,
        }
        for b in _graph.bridges
    }
    log_json(Config.output_path("bridge_state_log.json"), {str(global_tick): snapshot})


def log_meta_node_ticks(global_tick: int) -> None:
    assert _graph is not None
    events = {}
    for meta_id, meta in _graph.meta_nodes.items():
        member_ticks = []
        for nid in meta.member_ids:
            node = _graph.get_node(nid)
            if node is None:
                continue
            if any(t.time == global_tick for t in node.tick_history):
                member_ticks.append(nid)
        if member_ticks:
            events[meta_id] = member_ticks
        if events:
            log_json(
                Config.output_path("meta_node_tick_log.json"),
                {str(global_tick): events},
            )


def snapshot_graph(global_tick: int) -> Optional[str]:
    assert _graph is not None
    interval = getattr(Config, "snapshot_interval", 0)
    if interval and global_tick % interval == 0:
        path_dir = os.path.join(Config.output_dir, "runtime_graph_snapshots")
        os.makedirs(path_dir, exist_ok=True)
        path = os.path.join(path_dir, f"graph_{global_tick}.json")
        with open(path, "w") as f:
            json.dump(_graph.to_dict(), f, indent=2)
        return path
    return None


def log_metrics_per_tick(global_tick: int) -> None:
    from ..services.sim_services import NodeMetricsService

    assert _graph is not None
    NodeMetricsService(_graph).log_metrics(global_tick)


def export_curvature_map() -> None:
    assert _graph is not None
    grid = []
    try:
        with open(Config.output_path("curvature_log.json")) as f:
            for line in f:
                data = json.loads(line.strip())
                tick, edges = next(iter(data.items()))
                records = [
                    {
                        "source": k.split("->")[0],
                        "target": k.split("->")[1],
                        "delay": v["curved_delay"],
                    }
                    for k, v in edges.items()
                ]
                grid.append({"tick": int(tick), "edges": records})
    except FileNotFoundError:
        return

    with open(Config.output_path("curvature_map.json"), "w") as f:
        json.dump(grid, f, indent=2)
    print(f"✅ Curvature map exported to {Config.output_path('curvature_map.json')}")


def export_global_diagnostics() -> None:
    assert _graph is not None
    GlobalDiagnosticsService(_graph).export()


def export_regional_maps() -> None:
    assert _graph is not None
    regions = {}
    for nid, node in _graph.nodes.items():
        region = nid[0]
        regions.setdefault(region, []).append(node)

    regional_pressure = {}
    for reg, nodes in regions.items():
        pressure = sum(n.collapse_pressure for n in nodes) / len(nodes)
        regional_pressure[reg] = round(pressure, 3)

    matrix = {}
    for edge in _graph.edges:
        src_r = edge.source[0]
        tgt_r = edge.target[0]
        key = f"{src_r}->{tgt_r}"
        matrix[key] = matrix.get(key, 0) + 1

    with open(Config.output_path("regional_pressure_map.json"), "w") as f:
        json.dump(regional_pressure, f, indent=2)
    with open(Config.output_path("cluster_influence_matrix.json"), "w") as f:
        json.dump(matrix, f, indent=2)
    print("✅ Regional influence maps exported")


def write_output() -> None:
    assert _graph is not None
    logger.stop()
    with open(Config.output_path("tick_trace.json"), "w") as f:
        json.dump(_graph.to_dict(), f, indent=2)
    print(f"✅ Tick trace saved to {Config.output_path('tick_trace.json')}")

    inspection = _graph.inspect_superpositions()
    if Config.is_log_enabled("inspection_log.json"):
        with open(Config.output_path("inspection_log.json"), "w") as f:
            json.dump(inspection, f, indent=2)
        print(
            f"✅ Superposition inspection saved to {Config.output_path('inspection_log.json')}"
        )

    export_curvature_map()
    export_regional_maps()
    export_global_diagnostics()
