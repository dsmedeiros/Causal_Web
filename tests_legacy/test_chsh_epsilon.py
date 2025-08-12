import math
import random
import uuid

import numpy as np

from Causal_Web.engine.models.graph import CausalGraph
from Causal_Web.engine.services.entanglement_service import EntanglementService
from Causal_Web.engine.logging.logger import log_json, logger
from Causal_Web.analysis.bell import compute_bell_statistics
from Causal_Web.config import Config


def test_chsh_epsilon(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "output_dir", tmp_path)
    g = CausalGraph()
    g.add_node("A")
    g.add_node("B")
    g.add_edge("A", "B", epsilon=True, partner_id="pair")
    g.add_edge("B", "A", epsilon=True, partner_id="pair")

    rng = random.Random(0)
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

    for i in range(2, 202):
        emit(
            i,
            rng.choice(settings_a),
            rng.choice(settings_b),
        )

    logger.flush()
    s, _, _ = compute_bell_statistics(tmp_path / "entangled_log.jsonl")
    assert s > 2.6

