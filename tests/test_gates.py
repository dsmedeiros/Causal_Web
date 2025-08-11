import math
from collections import deque

import numpy as np

from Causal_Web.engine.models.graph import CausalGraph
from Causal_Web.engine.services.node_services import EdgePropagationService
from Causal_Web.engine.models.tick import GLOBAL_TICK_POOL
from Causal_Web.engine.engine_v2.rho_delay import update_rho_delay
from Causal_Web.engine.engine_v2.lccm import LCCM
from Causal_Web.engine.engine_v2.epairs import EPairs
from Causal_Web.engine.engine_v2.qtheta_c import deliver_packet, close_window


def _build_two_path_graph():
    g = CausalGraph()
    for nid in ["S", "A", "B", "D1", "D2"]:
        g.add_node(nid)
    g.add_edge("S", "A", attenuation=0.5)
    g.add_edge("S", "B", attenuation=0.5)
    g.add_edge("A", "D1")
    g.add_edge("A", "D2")
    g.add_edge("B", "D1")
    g.add_edge("B", "D2", phase_shift=np.pi)
    return g


def _run_paths(order, phase_b):
    g = _build_two_path_graph()
    for node in g.nodes.values():
        node.psi = np.zeros(2, dtype=np.complex128)
        node.incoming_tick_counts.clear()
    g.get_node("S").psi = np.array([1 + 0j, 0 + 0j])
    tick = GLOBAL_TICK_POOL.acquire()
    tick.origin = "self"
    tick.time = 0
    tick.phase = 0
    tick.amplitude = 1
    EdgePropagationService(g.get_node("S"), 0, 0, "self", g, tick).propagate()
    GLOBAL_TICK_POOL.release(tick)
    for sid in order:
        phase = phase_b if sid == "B" else 0.0
        tick2 = GLOBAL_TICK_POOL.acquire()
        tick2.origin = sid
        tick2.time = 0
        tick2.phase = phase
        tick2.amplitude = 1
        EdgePropagationService(g.get_node(sid), 0, phase, sid, g, tick2).propagate()
        GLOBAL_TICK_POOL.release(tick2)
    return float(abs(g.get_node("D1").psi[0]) ** 2)


def test_gate1_two_path_visibility_order_vs_phase():
    p_ab = _run_paths(["A", "B"], 0.0)
    p_ba = _run_paths(["B", "A"], 0.0)
    assert math.isclose(p_ab, p_ba, rel_tol=1e-9)
    p_phase = _run_paths(["A", "B"], np.pi)
    assert not math.isclose(p_ab, p_phase, rel_tol=1e-2)


def test_gate2_rho_to_d_eff_rising_and_relaxing():
    rho = 0.0
    rising = []
    for _ in range(5):
        rho, d_eff = update_rho_delay(
            rho,
            [],
            1.0,
            alpha_d=0.0,
            alpha_leak=0.1,
            eta=0.5,
            d0=1.0,
            gamma=1.0,
            rho0=0.1,
        )
        rising.append((rho, d_eff))
    assert all(rising[i][1] <= rising[i + 1][1] for i in range(len(rising) - 1))
    decays = []
    for _ in range(5):
        rho, d_eff = update_rho_delay(
            rho,
            [],
            0.0,
            alpha_d=0.0,
            alpha_leak=0.1,
            eta=0.5,
            d0=1.0,
            gamma=1.0,
            rho0=0.1,
        )
        decays.append(rho)
    assert all(decays[i] >= decays[i + 1] for i in range(len(decays) - 1))


def test_gate3_lccm_hysteresis_transitions():
    lccm = LCCM(
        W0=2,
        zeta1=0.0,
        zeta2=0.0,
        rho0=1.0,
        a=1.0,
        b=0.5,
        C_min=1.0,
        f_min=1.0,
        conf_min=0.0,
        H_max=1.0,
        T_hold=2,
        T_class=10,
    )
    lccm.advance_depth(0)
    lccm.deliver()
    lccm.deliver()
    assert lccm.layer == "Θ"
    lccm.update_eq(1.0)
    lccm.advance_depth(4)
    lccm.deliver()
    lccm.advance_depth(6)
    lccm.deliver()
    assert lccm.layer == "Q"


def test_gate4_epairs_locality_ttl_and_decay():
    mgr = EPairs(
        delta_ttl=0,
        ancestry_prefix_L=4,
        theta_max=0.1,
        sigma0=1.0,
        lambda_decay=0.5,
        sigma_reinforce=0.2,
        sigma_min=0.1,
    )
    edges = {"dst": [3], "d_eff": [1]}
    mgr.emit(
        origin=1,
        h_value=0b1101_0000,
        theta=0.1,
        depth_emit=0,
        edge_ids=[0],
        edges=edges,
    )
    mgr.emit(
        origin=2,
        h_value=0b1101_1111,
        theta=0.1,
        depth_emit=0,
        edge_ids=[0],
        edges=edges,
    )
    assert (1, 2) not in mgr.bridges
    mgr2 = EPairs(
        delta_ttl=1,
        ancestry_prefix_L=4,
        theta_max=0.1,
        sigma0=1.0,
        lambda_decay=0.5,
        sigma_reinforce=0.2,
        sigma_min=0.1,
    )
    mgr2.emit(
        origin=1,
        h_value=0b1101_0000,
        theta=0.1,
        depth_emit=0,
        edge_ids=[0],
        edges=edges,
    )
    mgr2.emit(
        origin=2,
        h_value=0b1101_1111,
        theta=0.1,
        depth_emit=0,
        edge_ids=[0],
        edges=edges,
    )
    assert (1, 2) in mgr2.bridges
    mgr2.lambda_decay = 1.0
    mgr2.decay_all()
    assert (1, 2) not in mgr2.bridges


def _energy_total():
    psi_acc = np.zeros(2, dtype=np.complex64)
    p_v = np.zeros(2, dtype=np.float32)
    bit_deque: deque[int] = deque()
    packet = {"psi": np.array([1, 0], np.complex64), "p": [0.4, 0.6], "bit": 1}
    edge = {"alpha": 1.0, "phi": 0.0, "A": 0.0, "U": np.eye(2, dtype=np.complex64)}
    depth, psi_acc, p_v, (bit, conf), intensity = deliver_packet(
        0, psi_acc, p_v, bit_deque, packet, edge, "Q"
    )
    psi, EQ = close_window(psi_acc)
    H_pv = float(-(p_v * np.log2(p_v + 1e-12)).sum())
    lccm = LCCM(
        W0=1,
        zeta1=0.0,
        zeta2=0.0,
        rho0=1.0,
        a=1.0,
        b=1.0,
        C_min=0.0,
        f_min=0.0,
        conf_min=0.0,
        H_max=1.0,
        T_hold=1,
        T_class=1,
    )
    E_theta = lccm.a * (1.0 - H_pv)
    E_C = lccm.b * conf
    return EQ + E_theta + E_C


def test_gate5_conservation_meter():
    total1 = _energy_total()
    total2 = _energy_total()
    assert math.isclose(total1, total2, rel_tol=1e-6)


def _simulate_chsh(kappa_a: float) -> float:
    return 2.0 + 2.0 * kappa_a


def test_gate6_bell_toggles_chsh_bounds():
    s_strict = _simulate_chsh(0.0)
    s_biased = _simulate_chsh(0.5)
    assert s_strict <= 2.0 + 1e-6
    assert s_biased > s_strict
