import math
import os
import tempfile
from Causal_Web.engine.node import Node
from Causal_Web.config import Config


def test_coherence_and_decoherence_calculation():
    node = Node("A")
    node.pending_superpositions[1] = [0.0, math.pi / 2]
    coh = node.compute_coherence_level(1)
    dec = node.compute_decoherence_field(1)
    assert 0 <= coh <= 1
    assert dec >= 0


def test_update_classical_state(tmp_path):
    old_dir = Config.output_dir
    Config.output_dir = tmp_path
    node = Node("A")
    node.update_classical_state(0.5, tick_time=1, graph=None, threshold=0.4, streak_required=1)
    assert node.is_classical
    Config.output_dir = old_dir
