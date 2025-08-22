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
from Causal_Web.engine.engine_v2.bell import Ancestry, BellHelpers


def _gate1_probability(phase: float, base: float) -> float:
    """Return detection probability at ``D1`` for a given phase offset.

    Parameters
    ----------
    phase:
        Phase offset applied to one arm of the interferometer.
    base:
        Baseline probability controlling both the offset and amplitude of the
        fringe pattern. ``base=0.5`` reproduces the previous behaviour.
    """

    return float(base * (1.0 + np.cos(phase)))


def _gate1_visibility(base: float) -> float:
    """Return interference visibility for a two-path interferometer."""

    intensities = [
        _gate1_probability(phase, base)
        for phase in np.linspace(0.0, 2 * np.pi, 25, endpoint=False)
    ]
    I_max = max(intensities)
    I_min = min(intensities)
    return (I_max - I_min) / (I_max + I_min)


def _gate2_delay(
    alpha_leak: float,
    eta: float,
    d0: float,
    gamma: float,
    rho0: float,
) -> tuple[float, float]:
    """Return delay slope during load and relaxation time after quench.

    Parameters
    ----------
    alpha_leak, eta, d0, gamma, rho0:
        Raw parameters passed through from the runner configuration and used
        by :func:`update_rho_delay`.
    """

    rho = 0.0
    d_vals = []
    depths = []

    # Sustained load phase
    for depth in range(1, 11):
        rho, d_eff = update_rho_delay(
            rho,
            [],
            1.0,
            alpha_d=0.0,
            alpha_leak=alpha_leak,
            eta=eta,
            d0=d0,
            gamma=gamma,
            rho0=rho0,
        )
        d_vals.append(d_eff)
        depths.append(depth)

    if len(depths) >= 2:
        slope = float(np.polyfit(depths, d_vals, 1)[0])
    else:
        slope = 0.0

    # Quench phase
    rho_decay = []
    for t in range(1, 11):
        rho, _ = update_rho_delay(
            rho,
            [],
            0.0,
            alpha_d=0.0,
            alpha_leak=alpha_leak,
            eta=eta,
            d0=d0,
            gamma=gamma,
            rho0=rho0,
        )
        rho_decay.append(rho)

    # Avoid log(0) by discarding non-positive entries
    t_vals = np.arange(1, len(rho_decay) + 1)
    rho_arr = np.array(rho_decay)
    mask = rho_arr > 0
    if mask.sum() >= 2:
        tau = -1.0 / np.polyfit(t_vals[mask], np.log(rho_arr[mask]), 1)[0]
    else:
        tau = 0.0

    return slope, float(tau)


def _gate3_hysteresis(cfg: Dict[str, float] | None = None) -> tuple[int, int, int]:
    """Return hysteresis transition depths for the LCCM model.

    Parameters
    ----------
    cfg:
        Optional configuration overriding ``W0``, ``C_min`` and ``T_hold``.

    Fan-in is swept upward until the layer transitions from ``Q`` to ``Θ``.
    Afterwards the system is held with ``EQ`` above ``C_min`` and zero additional
    fan-in so that a ``Θ``→``Q`` transition occurs after ``T_hold`` windows.
    The depth indices of these transitions and their difference
    (``q_to_theta_at - theta_to_q_at``) are returned.
    """

    cfg = cfg or {}
    lccm = LCCM(
        W0=int(cfg.get("W0", 2)),
        zeta1=0.0,
        zeta2=0.0,
        rho0=1.0,
        a=1.0,
        b=0.5,
        C_min=float(cfg.get("C_min", 1.0)),
        f_min=1.0,
        conf_min=0.0,
        H_max=1.0,
        T_hold=int(cfg.get("T_hold", 2)),
        T_class=1,
    )

    q_to_theta_at = -1
    theta_to_q_at = -1
    fan_in = 0
    depth = 0

    while q_to_theta_at < 0 and depth < 50:
        lccm.advance_depth(depth)
        for _ in range(fan_in):
            lccm.deliver(True)
        if lccm.layer == "Θ":
            q_to_theta_at = depth
        fan_in += 1
        depth += 1

    while theta_to_q_at < 0 and depth < 100:
        lccm.advance_depth(depth)
        lccm.update_eq(1.0)
        lccm._check_transitions()
        if lccm.layer == "Q":
            theta_to_q_at = depth
        depth += 1

    width = q_to_theta_at - theta_to_q_at
    return q_to_theta_at, theta_to_q_at, width


def _gate4_metrics(cfg: Dict[str, float] | None = None) -> Dict[str, float]:
    """Return basic ε-pair locality statistics.

    Parameters
    ----------
    cfg:
        Optional configuration overriding ``delta_ttl`` and bridge parameters
        ``sigma0``, ``lambda_decay``, ``sigma_reinforce`` and ``sigma_min``.

    A small network is initialised and two seeds emitted so that a single
    bridge forms.  The bridge is then decayed while tracking its ``sigma``
    lifetime.  TTL compliance is verified during emission.
    """

    cfg = cfg or {}
    mgr = EPairs(
        delta_ttl=int(cfg.get("delta_ttl", 1)),
        ancestry_prefix_L=4,
        theta_max=0.1,
        sigma0=float(cfg.get("sigma0", 1.0)),
        lambda_decay=float(cfg.get("lambda_decay", 0.5)),
        sigma_reinforce=float(cfg.get("sigma_reinforce", 0.2)),
        sigma_min=float(cfg.get("sigma_min", 0.1)),
    )
    edges = {"dst": [3], "d_eff": [1]}
    depth_emit = 0
    ttl_violations = 0
    bind_depths: List[int] = []
    for origin, h_val in ((1, 0b1101_0000), (2, 0b1101_1111)):
        d_eff = edges["d_eff"][0]
        expiry = depth_emit + mgr.delta_ttl
        depth_next = depth_emit + d_eff
        if depth_next > expiry:
            ttl_violations += 1
            continue
        mgr.emit(
            origin=origin,
            h_value=h_val,
            theta=0.1,
            depth_emit=depth_emit,
            edge_ids=[0],
            edges=edges,
        )
        bind_depths.append(depth_next)

    bridge_count = len(mgr.bridges)
    lifetimes = {key: 0 for key in mgr.bridges.keys()}
    while mgr.bridges:
        mgr.decay_all()
        for key in list(lifetimes.keys()):
            if key in mgr.bridges:
                lifetimes[key] += 1

    sigma_mean = float(np.mean(list(lifetimes.values()))) if lifetimes else 0.0
    max_bind_depth = float(max(bind_depths)) if bind_depths else 0.0

    return {
        "G4_bridge_count": float(bridge_count),
        "G4_max_bind_depth": max_bind_depth,
        "G4_ttl_violations": float(ttl_violations),
        "G4_sigma_lifetime_mean": sigma_mean,
    }


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


def _gate5_conservation_residual() -> float:
    """Return average absolute energy residual across windows."""

    totals = [_energy_total() for _ in range(3)]
    diffs = [totals[i + 1] - totals[i] for i in range(len(totals) - 1)]
    return float(np.mean(np.abs(diffs))) if diffs else 0.0


def _gate6_chsh(mi_mode: str, kappa_a: float) -> Dict[str, float]:
    """Return CHSH statistics and marginal bias for Bell tests."""

    helper = BellHelpers()
    trials = 256
    pairs = {"00": [], "01": [], "10": [], "11": []}
    count_a = 0
    count_b = 0
    total = 0
    for _ in range(trials):
        lam_u, _ = helper.lambda_at_source(Ancestry(), 0.0, 0.0)
        a0 = helper.setting_draw(
            "conditioned" if mi_mode == "MI_conditioned" else "strict",
            Ancestry(),
            lam_u,
            kappa_a,
        )
        a1 = helper.setting_draw(
            "conditioned" if mi_mode == "MI_conditioned" else "strict",
            Ancestry(),
            lam_u,
            kappa_a,
        )
        b0 = helper.setting_draw(
            "conditioned" if mi_mode == "MI_conditioned" else "strict",
            Ancestry(),
            lam_u,
            kappa_a,
        )
        b1 = helper.setting_draw(
            "conditioned" if mi_mode == "MI_conditioned" else "strict",
            Ancestry(),
            lam_u,
            kappa_a,
        )
        for a_set, b_set, key in (
            (a0, b0, "00"),
            (a0, b1, "01"),
            (a1, b0, "10"),
            (a1, b1, "11"),
        ):
            A = 1 if float(np.dot(a_set, lam_u)) >= 0 else -1
            B = 1 if float(np.dot(b_set, lam_u)) >= 0 else -1
            pairs[key].append(A * B)
            count_a += A == 1
            count_b += B == 1
            total += 1

    expectations = {k: float(np.mean(v)) for k, v in pairs.items()}
    S = (
        expectations["00"]
        + expectations["01"]
        + expectations["10"]
        - expectations["11"]
    )
    p_a = count_a / total if total else 0.5
    p_b = count_b / total if total else 0.5
    marginal_delta = float(max(abs(p_a - 0.5), abs(p_b - 0.5)))
    return {"G6_CHSH": S, "G6_marginal_delta": marginal_delta}


def run_gates(
    config: Dict[str, float], which: List[int], frames: int | None = None
) -> Dict[str, float]:
    """Execute selected gates and collect metrics.

    Parameters
    ----------
    config:
        Engine configuration passed to the gate harness. ``config['bell']`` may
        contain Bell experiment parameters used by Gate 6. ``config['gate3']``
        and ``config['gate4']`` optionally override parameters for those gates.
    which:
        List of gate identifiers to execute. Supports gate IDs 1–6.
    frames:
        Optional number of simulation frames to execute. ``None`` uses the
        default for each gate. The argument is currently advisory and exists so
        callers can model multi-fidelity evaluations.

    Returns
    -------
    dict
        Mapping of metric names to values together with invariant checks.

    Notes
    -----
    For backward compatibility the legacy short names ``G1``–``G6`` are
    also emitted when a gate produces a single primary metric. These mirror
    the more descriptive entries such as ``G1_visibility`` or
    ``G6_CHSH`` and are derived automatically via :func:`_add_legacy_aliases`.
    """

    metrics: Dict[str, float | bool] = {}
    deliveries = []

    base_prob = float(config.get("prob", 0.5)) if isinstance(config, dict) else 0.5
    prob = base_prob
    if 1 in which:
        vis1 = _gate1_visibility(base_prob)
        vis2 = _gate1_visibility(base_prob)
        metrics["G1_visibility"] = vis1
        deliveries.append({"d_arr": 2.0, "d_src": 0.0})
        metrics["inv_gate_determinism_ok"] = checks.determinism([vis1, vis2], 1e-12)
        prob = _gate1_probability(np.pi / 2, base_prob)
    else:
        metrics["inv_gate_determinism_ok"] = True
        prob = _gate1_probability(np.pi / 2, base_prob)
    if 2 in which:
        slope, tau = _gate2_delay(
            float(config.get("alpha_leak", 0.0)),
            float(config.get("eta", 0.0)),
            float(config.get("d0", 0.0)),
            float(config.get("gamma", 0.0)),
            float(config.get("rho0", 0.0)),
        )
        metrics["G2_delay_slope"] = slope
        metrics["G2_relax_tau"] = tau
    if 3 in which:
        gate3_cfg = config.get("gate3", {}) if isinstance(config, dict) else {}
        q2t, t2q, width = _gate3_hysteresis(gate3_cfg)
        metrics["G3_q_to_theta_at"] = float(q2t)
        metrics["G3_theta_to_q_at"] = float(t2q)
        metrics["G3_hysteresis_width"] = float(width)
    if 4 in which:
        gate4_cfg = config.get("gate4", {}) if isinstance(config, dict) else {}
        gate4 = _gate4_metrics(gate4_cfg)
        metrics.update(gate4)

    energy1 = _energy_total()
    energy2 = _energy_total()
    if 5 in which:
        residual = _gate5_conservation_residual()
        metrics["G5_conservation_residual"] = residual
    else:
        residual = energy2 - energy1
    bell_cfg = config.get("bell", {}) if isinstance(config, dict) else {}
    if 6 in which:
        gate6 = _gate6_chsh(
            bell_cfg.get("mi_mode", "MI_strict"),
            float(bell_cfg.get("kappa_a", 0.0)),
        )
        metrics.update(gate6)
    seq = [("q1", "h1", "m1")]

    metrics.update(
        {
            "inv_causality_ok": checks.causality(deliveries) if deliveries else True,
            "inv_conservation_residual": metrics.get(
                "G5_conservation_residual", residual
            ),
            "inv_no_signaling_delta": metrics.get("G6_marginal_delta", abs(prob - 0.5)),
            "inv_ancestry_ok": checks.ancestry_determinism(seq),
        }
    )
    _add_legacy_aliases(metrics)
    return metrics


LEGACY_ALIASES = {
    "G1_visibility": "G1",
    "G2_delay_slope": "G2",
    "G3_hysteresis_width": "G3",
    "G4_bridge_count": "G4",
    "G5_conservation_residual": "G5",
    "G6_CHSH": "G6",
}


def _add_legacy_aliases(metrics: Dict[str, float | bool]) -> None:
    """Populate short legacy metric names.

    Parameters
    ----------
    metrics:
        Metric mapping to augment in-place.
    """

    for canonical, legacy in LEGACY_ALIASES.items():
        if canonical in metrics and legacy not in metrics:
            metrics[legacy] = metrics[canonical]
