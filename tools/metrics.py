from __future__ import annotations

"""Diagnostic metrics for common simulations."""

from dataclasses import dataclass
import math
import random
import uuid
import tempfile
from pathlib import Path

import numpy as np

from Causal_Web.engine.models.graph import CausalGraph
from Causal_Web.engine.services.entanglement_service import EntanglementService
from Causal_Web.engine.logging.logger import log_json, logger
from Causal_Web.analysis.bell import compute_bell_statistics
from Causal_Web.config import Config
from Causal_Web.engine.services.node_services import EdgePropagationService
from Causal_Web.engine.models.tick import GLOBAL_TICK_POOL
from Causal_Web.engine.tick_engine.tick_router import TickRouter


@dataclass
class TwinResult:
    """Results of a twin-paradox simulation."""

    tau_home: float
    tau_traveller: float
    ratio: float
    analytic: float


def bell_score(epsilon: bool, runs: int = 200, seed: int | None = None) -> float:
    """Estimate the CHSH ``S`` value using ``\u03b5`` edges.

    Parameters
    ----------
    epsilon:
        When ``True`` the two nodes are linked by ``\u03b5`` edges enabling
        super-classical correlations.
    runs:
        Number of measurement pairs to generate.
    seed:
        Optional seed for the random number generator.

    Returns
    -------
    float
        The calculated CHSH ``S`` value.
    """

    rng = random.Random(seed)
    with tempfile.TemporaryDirectory() as tmp:
        original_dir = Config.output_dir
        try:
            Config.output_dir = tmp
            g = CausalGraph()
            g.add_node("A")
            g.add_node("B")
            g.add_edge("A", "B", epsilon=epsilon, partner_id="pair")
            g.add_edge("B", "A", epsilon=epsilon, partner_id="pair")
            ent_id = "E"
            settings_a = [0.0, math.pi / 2]
            settings_b = [math.pi / 4, 3 * math.pi / 4]

            def emit(tick: int, a_setting: float, b_setting: float) -> None:
                node_a = g.get_node("A")
                outcome_a = rng.choice([1, -1])
                node_a.psi = (
                    np.array([1, 0], np.complex128)
                    if outcome_a == 1
                    else np.array([0, 1], np.complex128)
                )
                EntanglementService.collapse_epsilon(g, node_a, tick)
                p_same = 0.5 * (1 + math.cos(a_setting - b_setting))
                outcome_b = outcome_a if rng.random() < p_same else -outcome_a
                log_json(
                    "entangled",
                    "measurement",
                    {
                        "tick_id": str(uuid.uuid4()),
                        "observer_id": "A",
                        "entangled_id": ent_id,
                        "measurement_setting": a_setting,
                        "binary_outcome": outcome_a,
                    },
                    tick=tick,
                )
                log_json(
                    "entangled",
                    "measurement",
                    {
                        "tick_id": str(uuid.uuid4()),
                        "observer_id": "B",
                        "entangled_id": ent_id,
                        "measurement_setting": b_setting,
                        "binary_outcome": outcome_b,
                    },
                    tick=tick,
                )

            emit(0, 0.0, math.pi / 4)
            emit(1, math.pi / 2, 3 * math.pi / 4)
            for i in range(2, runs + 2):
                emit(i, rng.choice(settings_a), rng.choice(settings_b))
            logger.flush()
            s_val, _ = compute_bell_statistics(Path(tmp) / "entangled_log.jsonl")
            return s_val
        finally:
            Config.output_dir = original_dir


def interference_visibility(
    fan_in: int, runs: int = 200, seed: int | None = None
) -> float:
    """Return interference visibility for a double-slit style graph.

    Parameters
    ----------
    fan_in:
        Threshold for decoherence. ``0`` keeps coherent phases.
    runs:
        Number of simulation runs to average.
    seed:
        Optional random seed.

    Returns
    -------
    float
        Visibility defined as ``(I_max - I_min) / (I_max + I_min)``.
    """

    Config.N_DECOH = fan_in
    rng = random.Random(seed)
    np.random.seed(seed or 0)

    def _build_graph() -> CausalGraph:
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

    g = _build_graph()
    counts = np.zeros(2)
    for _ in range(runs):
        for node_id, node in g.nodes.items():
            node.psi = (
                np.array([1 + 0j, 0 + 0j])
                if node_id == "S"
                else np.zeros(2, dtype=np.complex128)
            )
            node.incoming_tick_counts.clear()
        tick = GLOBAL_TICK_POOL.acquire()
        tick.origin = "self"
        tick.time = 0
        tick.phase = 0
        tick.amplitude = 1
        EdgePropagationService(g.get_node("S"), 0, 0, "self", g, tick).propagate()
        GLOBAL_TICK_POOL.release(tick)
        if fan_in:
            for sid in ["A", "B"]:
                node = g.get_node(sid)
                node.incoming_tick_counts[0] = fan_in
                TickRouter.record_fanin(node, 0)
        for sid in ["A", "B"]:
            phase = rng.uniform(0, 2 * np.pi) if fan_in else 0.0
            tick2 = GLOBAL_TICK_POOL.acquire()
            tick2.origin = sid
            tick2.time = 0
            tick2.phase = phase
            tick2.amplitude = 1
            EdgePropagationService(g.get_node(sid), 0, phase, sid, g, tick2).propagate()
            GLOBAL_TICK_POOL.release(tick2)
        counts[0] += abs(g.get_node("D1").psi[0]) ** 2
        counts[1] += abs(g.get_node("D2").psi[0]) ** 2
        for nid in ["A", "B", "D1", "D2"]:
            g.get_node(nid).psi[:] = 0
    probs = counts / runs
    i_max, i_min = probs.max(), probs.min()
    return float((i_max - i_min) / (i_max + i_min) if (i_max + i_min) else 0.0)


def tau_ratio(velocity: float, total_time: float = 10.0, dt: float = 1.0) -> TwinResult:
    """Compute proper time ratio for travelling and stationary twins.

    Parameters
    ----------
    velocity:
        Constant speed of the travelling twin.
    total_time:
        Total coordinate time of the round trip.
    dt:
        Scheduler time step.

    Returns
    -------
    TwinResult
        Proper times for each twin along with the measured and analytic ratios.
    """

    from Causal_Web.analysis.twin import run_demo

    tau_home, tau_traveller = run_demo(total_time=total_time, dt=dt, velocity=velocity)
    ratio = tau_traveller / tau_home if tau_home else 0.0
    analytic = math.sqrt(1 - velocity**2)
    return TwinResult(tau_home, tau_traveller, ratio, analytic)
