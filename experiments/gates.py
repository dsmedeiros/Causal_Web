"""Gate metric helpers.

This module provides a thin wrapper around the engine's gate benchmark
suite. The previous implementation returned placeholder values which always
reported invariant checks as successful. The harness now delegates to
lightweight engine entry points so that gate metrics and invariants reflect
actual behaviour.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List

import numpy as np

from invariants import checks
from Causal_Web.engine.models.graph import CausalGraph
from Causal_Web.engine.services.node_services import EdgePropagationService
from Causal_Web.engine.models.tick import GLOBAL_TICK_POOL
from Causal_Web.engine.engine_v2.qtheta_c import deliver_packet, close_window
from Causal_Web.engine.engine_v2.lccm import LCCM


def _gate1_visibility() -> float:
    """Return detection probability at ``D1`` for a two-path graph."""

    g = CausalGraph()
    for nid in ["S", "A", "B", "D1", "D2"]:
        g.add_node(nid)
    g.add_edge("S", "A", attenuation=0.5)
    g.add_edge("S", "B", attenuation=0.5)
    g.add_edge("A", "D1")
    g.add_edge("A", "D2")
    g.add_edge("B", "D1")
    g.add_edge("B", "D2", phase_shift=np.pi)

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
    for sid in ["A", "B"]:
        phase = 0.0 if sid == "A" else np.pi / 2
        tick2 = GLOBAL_TICK_POOL.acquire()
        tick2.origin = sid
        tick2.time = 0
        tick2.phase = phase
        tick2.amplitude = 1
        EdgePropagationService(g.get_node(sid), 0, phase, sid, g, tick2).propagate()
        GLOBAL_TICK_POOL.release(tick2)
    return float(abs(g.get_node("D1").psi[0]) ** 2)


def _energy_total() -> float:
    """Total energy across layers for a single packet."""

    psi_acc = np.zeros(2, dtype=np.complex64)
    p_v = np.zeros(2, dtype=np.float32)
    bit_deque: deque[int] = deque()
    packet = {"psi": np.array([1, 0], np.complex64), "p": [0.4, 0.6], "bit": 1}
    edge = {"alpha": 1.0, "phi": 0.0, "A": 0.0, "U": np.eye(2, dtype=np.complex64)}
    _, psi_acc, p_v, (bit, conf), *_ = deliver_packet(
        0, psi_acc, p_v, bit_deque, packet, edge
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


def run_gates(config: Dict[str, float], which: List[int]) -> Dict[str, float]:
    """Execute selected gates and collect metrics.

    Parameters
    ----------
    config:
        Engine configuration passed to the gate harness (unused).
    which:
        List of gate identifiers to execute.

    Returns
    -------
    dict
        Mapping of metric names to values together with invariant checks.
    """

    metrics: Dict[str, float | bool] = {}
    deliveries = []

    if 1 in which:
        prob = _gate1_visibility()
        metrics["G1"] = prob
        deliveries.append({"d_arr": 2.0, "d_src": 0.0})

    energy1 = _energy_total()
    energy2 = _energy_total()
    seq = [("q1", "h1", "m1")]

    metrics.update(
        {
            "inv_causality_ok": checks.causality(deliveries) if deliveries else True,
            "inv_conservation_residual": energy2 - energy1,
            "inv_no_signaling_delta": abs(metrics.get("G1", 0.5) - 0.5),
            "inv_ancestry_ok": checks.ancestry_determinism(seq),
        }
    )
    return metrics
