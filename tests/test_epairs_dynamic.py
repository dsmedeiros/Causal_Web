"""Tests for the dynamic ε-pair utilities."""

from Causal_Web.engine.engine_v2.epairs import Bridge, EPairs


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
        h_value=0b1101_1110,
        theta=0.10,
        depth_emit=0,
        edge_ids=[0],
        edges=edges,
    )
    mgr.emit(
        origin=2,
        h_value=0b1101_0001,
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
    mgr.adjacency[1] = [2]
    mgr.adjacency[2] = [1]
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
        h_value=0b1101_1110,
        theta=0.10,
        depth_emit=0,
        edge_ids=[0],
        edges=edges,
    )
    mgr.emit(
        origin=2,
        h_value=0b1101_0001,
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
        h_value=0b1101_0000,
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
        h_value=0b1101_0000,
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
        h_value=0b1101_0000,
        theta=0.0,
        depth_emit=0,
        edge_ids=[0],
        edges=edges1,
    )
    edges2 = {"dst": [3], "d_eff": [1]}
    mgr.carry(2, depth_curr=1, edge_ids=[0], edges=edges2)

    # prefix mismatch drop
    edges3 = {"dst": [5], "d_eff": [1]}
    mgr.emit(
        origin=4,
        h_value=0b1111_0000,
        theta=0.0,
        depth_emit=0,
        edge_ids=[0],
        edges=edges3,
    )
    mgr.emit(
        origin=5,
        h_value=0b0000_0000,
        theta=0.0,
        depth_emit=0,
        edge_ids=[0],
        edges=edges3,
    )

    # angle mismatch drop
    edges4 = {"dst": [7], "d_eff": [1]}
    mgr.emit(
        origin=6,
        h_value=0b1010_0000,
        theta=0.0,
        depth_emit=0,
        edge_ids=[0],
        edges=edges4,
    )
    mgr.emit(
        origin=7,
        h_value=0b1010_1111,
        theta=1.0,
        depth_emit=0,
        edge_ids=[0],
        edges=edges4,
    )

    assert any(lbl == "seed_emitted" for lbl, _ in events)
    assert any(
        lbl == "seed_dropped" and val["reason"] == "expired" for lbl, val in events
    )
    assert any(
        lbl == "seed_dropped" and val["reason"] == "prefix" for lbl, val in events
    )
    assert any(
        lbl == "seed_dropped" and val["reason"] == "angle" for lbl, val in events
    )
