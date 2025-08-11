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
    mgr.emit(origin=1, h_value=0b1101_1110, theta=0.10, neighbours=[3])
    mgr.emit(origin=2, h_value=0b1101_0001, theta=0.15, neighbours=[3])
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

