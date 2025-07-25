from __future__ import annotations

from typing import Any, Dict, List, TypedDict

# Reusable typed mappings for graph JSON files

NodeData = TypedDict(
    "NodeData",
    {
        "id": str,
        "x": float,
        "y": float,
        "frequency": float,
        "refractory_period": float,
        "base_threshold": float,
        "phase": float,
        "origin_type": str,
        "generation_tick": int,
        "parent_ids": List[str],
        "goals": Dict[str, Any],
    },
    total=False,
)

EdgeData = TypedDict(
    "EdgeData",
    {
        "from": str,
        "to": str,
        "delay": float,
        "attenuation": float,
        "density": float,
        "phase_shift": float,
        "weight": float,
    },
    total=False,
)

BridgeData = TypedDict(
    "BridgeData",
    {
        "from": str,
        "to": str,
        "bridge_type": str,
        "phase_offset": float,
        "drift_tolerance": float | None,
        "decoherence_limit": float | None,
        "initial_strength": float,
        "medium_type": str,
        "mutable": bool,
    },
    total=False,
)

MetaNodeData = TypedDict(
    "MetaNodeData",
    {
        "members": List[str],
        "constraints": Dict[str, Any],
        "type": str,
        "origin": str,
        "collapsed": bool,
        "x": float,
        "y": float,
    },
    total=False,
)

ObserverData = TypedDict(
    "ObserverData",
    {
        "id": str,
        "monitors": List[str],
        "frequency": float,
        "target_nodes": List[str],
        "x": float,
        "y": float,
    },
    total=False,
)

GraphDict = TypedDict(
    "GraphDict",
    {
        "nodes": Dict[str, NodeData] | List[NodeData],
        "edges": List[EdgeData],
        "bridges": List[BridgeData],
        "tick_sources": List[Dict[str, Any]],
        "observers": List[ObserverData],
        "meta_nodes": Dict[str, MetaNodeData],
    },
    total=False,
)
