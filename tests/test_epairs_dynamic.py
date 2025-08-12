"""Tests for the dynamic ε-pair utilities."""

import logging
import numpy as np

from Causal_Web.engine.engine_v2.adapter import EngineAdapter
from Causal_Web.engine.engine_v2.epairs import Bridge, EPairs
from Causal_Web.engine.engine_v2.state import Packet


def _make_manager():
    return EPairs(
        delta_ttl=2,
        ancestry_prefix_L=4,
        theta_max=0.1,
        sigma0=1.0,
        lambda_decay=0.5,
        sigma_reinforce=0.2,
        sigma_min=0.1,
    )


def test_seed_binding_creates_bridge():
    mgr = _make_manager()
    # Seeds share a 4 bit prefix and close theta values
    edges = {"dst": [3], "d_eff": [1]}
    mgr.emit(
        origin=1,
        h_value=0xD00000000000000E,
        theta=0.10,
        depth_emit=0,
        edge_ids=[0],
        edges=edges,
    )
    mgr.emit(
        origin=2,
        h_value=0xD000000000000001,
        theta=0.15,
        depth_emit=0,
        edge_ids=[0],
        edges=edges,
    )
    assert (1, 2) in mgr.bridges
    bridge = mgr.bridges[(1, 2)]
    assert bridge.sigma == 1.0
    assert bridge.edge_id < 0
    assert 2 in mgr.adjacency.get(1, [])
    assert 1 in mgr.adjacency.get(2, [])


def test_bridge_reinforcement_and_decay():
    mgr = _make_manager()
    mgr.bridges[(1, 2)] = Bridge(0.3)
    from array import array

    mgr.adjacency[1] = array("i", [2])
    mgr.adjacency[2] = array("i", [1])
    mgr.decay_all()
    mgr.reinforce(1, 2)
    # sigma: (1-0.5)*0.3 + 0.2 = 0.35 > sigma_min -> bridge persists
    assert mgr.bridges[(1, 2)].sigma == (1 - 0.5) * 0.3 + 0.2
    # decay below minimum removes the bridge
    mgr.lambda_decay = 0.8
    mgr.sigma_reinforce = 0.0
    mgr.decay_all()
    assert (1, 2) not in mgr.bridges


def test_bridge_lifecycle_events(monkeypatch):
    events = []

    def fake_log_record(category, label, *, value=None, **kwargs):
        events.append((label, value))

    monkeypatch.setattr(
        "Causal_Web.engine.engine_v2.epairs.log_record", fake_log_record
    )

    mgr = _make_manager()
    edges = {"dst": [3], "d_eff": [1]}
    mgr.emit(
        origin=1,
        h_value=0xD00000000000000E,
        theta=0.10,
        depth_emit=0,
        edge_ids=[0],
        edges=edges,
    )
    mgr.emit(
        origin=2,
        h_value=0xD000000000000001,
        theta=0.15,
        depth_emit=0,
        edge_ids=[0],
        edges=edges,
    )
    assert any(
        lbl == "bridge_created" and {"src", "dst", "sigma", "bridge_id"} <= value.keys()
        for lbl, value in events
    )
    mgr.lambda_decay = 1.0
    mgr.decay_all()
    assert any(
        lbl == "bridge_removed" and {"src", "dst", "sigma", "bridge_id"} <= value.keys()
        for lbl, value in events
    )


def test_bridge_full_lifecycle_removal_event(monkeypatch):
    """Bridge creation, reinforcement and decay emit a removal event.

    This exercises the complete lifecycle: a bridge is created, reinforced to
    keep it alive, then allowed to decay immediately.  The manager should log
    both the creation and removal events and the bridge should be deleted from
    internal state.  It locks behaviour so cleanup is observable rather than
    silent.
    """

    events = []

    def fake_log_record(category, label, *, value=None, **kwargs):
        if label in {"bridge_created", "bridge_removed"}:
            events.append(label)

    monkeypatch.setattr(
        "Causal_Web.engine.engine_v2.epairs.log_record", fake_log_record
    )

    mgr = _make_manager()
    mgr._create_bridge(1, 2)
    mgr.reinforce(1, 2)
    mgr.lambda_decay = 1.0
    mgr.decay_all()

    assert events == ["bridge_created", "bridge_removed"]
    assert (1, 2) not in mgr.bridges


def test_bridge_id_stability():
    mgr = _make_manager()
    mgr._create_bridge(1, 2)
    mgr._create_bridge(3, 4)
    id1 = mgr.bridges[(1, 2)].edge_id
    id2 = mgr.bridges[(3, 4)].edge_id
    assert id1 != id2
    mgr.reinforce(1, 2)
    assert mgr.bridges[(1, 2)].edge_id == id1
    mgr._remove_bridge(1, 2)
    mgr._create_bridge(1, 2)
    assert mgr.bridges[(1, 2)].edge_id != id1


def test_seed_ttl_propagates_and_expires():
    mgr = EPairs(
        delta_ttl=2,
        ancestry_prefix_L=4,
        theta_max=0.1,
        sigma0=1.0,
        lambda_decay=0.5,
        sigma_reinforce=0.2,
        sigma_min=0.1,
    )
    edges = {"dst": [2, 3, 4], "d_eff": [1, 1, 1]}
    mgr.emit(
        origin=1,
        h_value=0xD000000000000000,
        theta=0.1,
        depth_emit=0,
        edge_ids=[0],
        edges=edges,
    )
    mgr.carry(2, depth_curr=1, edge_ids=[1], edges=edges)
    assert 2 not in mgr.seeds
    assert mgr.seeds[3][0].expiry_depth == 2
    mgr.carry(3, depth_curr=2, edge_ids=[2], edges=edges)
    assert 3 not in mgr.seeds
    assert 4 not in mgr.seeds


def test_seed_expires_with_large_d_eff():
    mgr = _make_manager()
    edges = {"dst": [2], "d_eff": [5]}
    mgr.emit(
        origin=1,
        h_value=0xD000000000000000,
        theta=0.1,
        depth_emit=0,
        edge_ids=[0],
        edges=edges,
    )
    assert 2 not in mgr.seeds


def test_seed_logging(monkeypatch):
    events = []

    def fake_log_record(category, label, *, value=None, **kwargs):
        events.append((label, value))

    monkeypatch.setattr(
        "Causal_Web.engine.engine_v2.epairs.log_record", fake_log_record
    )

    mgr = EPairs(
        delta_ttl=1,
        ancestry_prefix_L=4,
        theta_max=0.1,
        sigma0=1.0,
        lambda_decay=0.5,
        sigma_reinforce=0.2,
        sigma_min=0.1,
    )

    # expiry drop
    edges1 = {"dst": [2], "d_eff": [1]}
    mgr.emit(
        origin=1,
        h_value=0xD000000000000000,
        theta=0.0,
        depth_emit=0,
        edge_ids=[0],
        edges=edges1,
    )
    edges2 = {"dst": [3], "d_eff": [1]}
    mgr.carry(2, depth_curr=1, edge_ids=[0], edges=edges2)

    # prefix mismatch seeds coexist
    edges3 = {"dst": [5], "d_eff": [1]}
    mgr.emit(
        origin=4,
        h_value=0xF000000000000000,
        theta=0.0,
        depth_emit=0,
        edge_ids=[0],
        edges=edges3,
    )
    mgr.emit(
        origin=5,
        h_value=0x0000000000000000,
        theta=0.0,
        depth_emit=0,
        edge_ids=[0],
        edges=edges3,
    )
    assert len(mgr.seeds.get(5, [])) == 2

    # angle mismatch drop
    edges4 = {"dst": [7], "d_eff": [1]}
    mgr.emit(
        origin=6,
        h_value=0xA000000000000000,
        theta=0.0,
        depth_emit=0,
        edge_ids=[0],
        edges=edges4,
    )
    mgr.emit(
        origin=7,
        h_value=0xA00000000000000F,
        theta=1.0,
        depth_emit=0,
        edge_ids=[0],
        edges=edges4,
    )

    assert any(lbl == "seed_emitted" for lbl, _ in events)
    assert any(
        lbl == "seed_dropped" and val["reason"] == "expired" for lbl, val in events
    )
    assert not any(
        lbl == "seed_dropped" and val["reason"] == "prefix" for lbl, val in events
    )
    assert any(
        lbl == "seed_dropped" and val["reason"] == "angle" for lbl, val in events
    )


def test_seed_chain_respects_d_eff():
    mgr = EPairs(
        delta_ttl=6,
        ancestry_prefix_L=4,
        theta_max=0.1,
        sigma0=1.0,
        lambda_decay=0.5,
        sigma_reinforce=0.2,
        sigma_min=0.1,
    )

    edges1 = {"dst": [2], "d_eff": [3]}
    mgr.emit(
        origin=1,
        h_value=0,
        theta=0.0,
        depth_emit=0,
        edge_ids=[0],
        edges=edges1,
    )

    assert 2 in mgr.seeds
    assert mgr.seeds[2][0].expiry_depth == 6

    edges2 = {"dst": [3], "d_eff": [4]}
    mgr.carry(2, depth_curr=3, edge_ids=[0], edges=edges2)

    assert 3 not in mgr.seeds


def test_seed_list_capped(monkeypatch, caplog):
    events = []

    def fake_log_record(category, label, *, value=None, **kwargs):
        events.append((label, value))

    monkeypatch.setattr(
        "Causal_Web.engine.engine_v2.epairs.log_record", fake_log_record
    )

    mgr = EPairs(
        delta_ttl=4,
        ancestry_prefix_L=4,
        theta_max=0.1,
        sigma0=1.0,
        lambda_decay=0.5,
        sigma_reinforce=0.2,
        sigma_min=0.1,
        max_seeds_per_site=2,
    )
    edges = {"dst": [5], "d_eff": [1]}
    with caplog.at_level(logging.WARNING):
        mgr.emit(
            origin=1,
            h_value=0x0000000000000000,
            theta=0.0,
            depth_emit=0,
            edge_ids=[0],
            edges=edges,
        )
        mgr.emit(
            origin=2,
            h_value=0x1000000000000000,
            theta=0.0,
            depth_emit=0,
            edge_ids=[0],
            edges=edges,
        )
        mgr.emit(
            origin=3,
            h_value=0x2000000000000000,
            theta=0.0,
            depth_emit=0,
            edge_ids=[0],
            edges=edges,
        )

    assert [s.origin for s in mgr.seeds[5]] == [2, 3]
    assert any("max seeds per site" in rec.message for rec in caplog.records)
    assert sum(1 for rec in caplog.records if "max seeds per site" in rec.message) == 1
    assert any(
        lbl == "seed_dropped" and val["reason"] == "overflow" for lbl, val in events
    )


def test_bridge_delay_median_used_in_scheduler(monkeypatch):
    adapter = EngineAdapter()
    graph = {"nodes": [{"id": "0"}, {"id": "1"}], "edges": []}
    adapter.build_graph(graph)

    delays = [2, 6]
    median = int(np.median(delays))
    adapter._epairs._create_bridge(0, 1, d_bridge=median)

    bridge = adapter._epairs.bridges[(0, 1)]
    assert bridge.d_bridge == median

    pushed: list[tuple[int, int, int]] = []
    original_push = adapter._scheduler.push

    def capture(depth: int, dst: int, edge_id: int, packet: Packet) -> None:
        pushed.append((depth, dst, edge_id))
        original_push(depth, dst, edge_id, packet)

    adapter._scheduler.push = capture  # type: ignore[assignment]

    adapter._scheduler.push(0, 0, 0, Packet(src=-1, dst=0, payload=None))
    adapter.run_until_next_window_or(limit=10)

    assert any(depth == median and dst == 1 for depth, dst, _ in pushed)


def test_bridge_delay_reads_live_d_eff():
    adapter = EngineAdapter()
    graph = {
        "nodes": [{"id": "0"}, {"id": "1"}],
        "edges": [
            {"from": "0", "to": "1", "delay": 1.0},
            {"from": "1", "to": "0", "delay": 1.0},
        ],
    }
    adapter.build_graph(graph)
    adapter._arrays.edges["d_eff"][0] = 9
    adapter._epairs._create_bridge(0, 1)
    bridge = adapter._epairs.bridges[(0, 1)]
    expected = int(np.median([9, adapter._arrays.edges["d_eff"][1]]))
    assert bridge.d_bridge == expected


def test_bridge_delay_updates_after_rho_change():
    adapter = EngineAdapter()
    graph = {
        "nodes": [{"id": "0"}, {"id": "1"}],
        "edges": [
            {"from": "0", "to": "1", "delay": 1.0},
            {"from": "1", "to": "0", "delay": 1.0},
        ],
    }
    adapter.build_graph(graph)

    adapter._epairs._create_bridge(0, 1)
    bridge = adapter._epairs.bridges[(0, 1)]
    assert bridge.d_bridge == 1

    adapter._epairs._remove_bridge(0, 1)
    adapter._arrays.edges["d_eff"][0] = 9
    adapter._epairs._create_bridge(0, 1)
    bridge = adapter._epairs.bridges[(0, 1)]
    expected = int(np.median([9, adapter._arrays.edges["d_eff"][1]]))
    assert bridge.d_bridge == expected
