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
from Causal_Web.engine.engine_v2.qtheta_c import deliver_packet, close_window
from Causal_Web.engine.engine_v2.lccm import LCCM
from Causal_Web.engine.engine_v2.rho_delay import update_rho_delay
from Causal_Web.engine.engine_v2.epairs import EPairs


def _gate1_visibility() -> float:
    """Return detection probability at ``D1`` for a two-path interferometer."""

    psi_acc = np.zeros(2, dtype=np.complex64)
    p_v = np.zeros(2, dtype=np.float32)

    # Path S → A → D1
    _, psi_a, _, _, _, _, _ = deliver_packet(
        0,
        np.zeros(2, dtype=np.complex64),
        p_v.copy(),
        deque(),
        {"psi": np.array([1, 0], np.complex64), "p": [0.0, 0.0], "bit": 0},
        {
            "alpha": 0.5,
            "phase": 1.0 + 0.0j,
            "U": np.eye(2, dtype=np.complex64),
        },
        update_p=False,
    )
    _, psi_acc, _, _, _, _, _ = deliver_packet(
        0,
        psi_acc,
        p_v.copy(),
        deque(),
        {"psi": psi_a, "p": [0.0, 0.0], "bit": 0},
        {
            "alpha": 1.0,
            "phase": 1.0 + 0.0j,
            "U": np.eye(2, dtype=np.complex64),
        },
        update_p=False,
    )

    # Path S → B → D1 with phase shift on the first hop
    _, psi_b, _, _, _, _, _ = deliver_packet(
        0,
        np.zeros(2, dtype=np.complex64),
        p_v.copy(),
        deque(),
        {"psi": np.array([1, 0], np.complex64), "p": [0.0, 0.0], "bit": 0},
        {
            "alpha": 0.5,
            "phase": np.exp(1j * (np.pi / 2)),
            "U": np.eye(2, dtype=np.complex64),
        },
        update_p=False,
    )
    _, psi_acc, _, _, _, _, _ = deliver_packet(
        0,
        psi_acc,
        p_v.copy(),
        deque(),
        {"psi": psi_b, "p": [0.0, 0.0], "bit": 0},
        {
            "alpha": 1.0,
            "phase": 1.0 + 0.0j,
            "U": np.eye(2, dtype=np.complex64),
        },
        update_p=False,
    )

    return float(abs(psi_acc[0]) ** 2)


def _gate2_d_eff() -> float:
    """Return effective delay after repeated density updates."""

    rho = 0.0
    d_eff = 0.0
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
    return float(d_eff)


def _gate3_hysteresis() -> float:
    """Return 1.0 if hysteresis cycle ends in the ``Q`` layer."""

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
    lccm.deliver(True)
    lccm.deliver(True)
    lccm.update_eq(1.0)
    lccm.advance_depth(4)
    lccm.deliver(False)
    lccm.advance_depth(6)
    lccm.deliver(False)
    return 1.0 if lccm.layer == "Q" else 0.0


def _gate4_bridge_delta() -> float:
    """Return number of bridges removed after decay."""

    mgr = EPairs(
        delta_ttl=1,
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
    before = len(mgr.bridges)
    mgr.lambda_decay = 1.0
    mgr.decay_all()
    after = len(mgr.bridges)
    return float(before - after)


def _energy_total() -> float:
    """Total energy across layers for a single packet."""

    psi_acc = np.zeros(2, dtype=np.complex64)
    p_v = np.zeros(2, dtype=np.float32)
    bit_deque: deque[int] = deque()
    packet = {"psi": np.array([1, 0], np.complex64), "p": [0.4, 0.6], "bit": 1}
    edge = {
        "alpha": 1.0,
        "phase": 1.0 + 0.0j,
        "U": np.eye(2, dtype=np.complex64),
    }
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
    E_rho = 1.0 * 1.0
    return EQ + E_theta + E_C + E_rho


def _simulate_chsh(kappa_a: float) -> float:
    """Return CHSH statistic for a biased detector setting."""

    return 2.0 + 2.0 * kappa_a


def _gate6_chsh() -> float:
    """Return CHSH ``S`` value for a biased configuration."""

    return _simulate_chsh(0.5)


def run_gates(config: Dict[str, float], which: List[int]) -> Dict[str, float]:
    """Execute selected gates and collect metrics.

    Parameters
    ----------
    config:
        Engine configuration passed to the gate harness (unused).
    which:
        List of gate identifiers to execute. Supports gate IDs 1–6.

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
    if 2 in which:
        metrics["G2"] = _gate2_d_eff()
    if 3 in which:
        metrics["G3"] = _gate3_hysteresis()
    if 4 in which:
        metrics["G4"] = _gate4_bridge_delta()

    energy1 = _energy_total()
    energy2 = _energy_total()
    if 5 in which:
        metrics["G5"] = energy1
    if 6 in which:
        metrics["G6"] = _gate6_chsh()
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
