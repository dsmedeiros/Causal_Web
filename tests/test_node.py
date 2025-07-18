import math
import os
import tempfile
from Causal_Web.engine.node import Node
from Causal_Web.engine.graph import CausalGraph
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
    node.update_classical_state(
        0.5, tick_time=1, graph=None, threshold=0.4, streak_required=1
    )
    assert node.is_classical
    Config.output_dir = old_dir


def test_tick_threshold(tmp_path):
    old_dir = Config.output_dir
    old_thresh = getattr(Config, "tick_threshold", 1)
    Config.output_dir = tmp_path
    Config.tick_threshold = 2
    node = Node("A")
    node.schedule_tick(1, 0.0)
    fire, _, reason = node.should_tick(1)
    assert not fire
    assert reason == "count_threshold"
    node.schedule_tick(1, math.pi / 2)
    fire, phase, reason = node.should_tick(1)
    assert fire
    assert reason in {"threshold", "merged"}
    Config.tick_threshold = old_thresh
    Config.output_dir = old_dir


def test_configurable_refractory_period():
    old_period = getattr(Config, "refractory_period", 2)
    Config.refractory_period = 3
    g = CausalGraph()
    g.add_node("A")
    assert g.get_node("A").refractory_period == 3
    node = Node("B")
    assert node.refractory_period == 3
    Config.refractory_period = old_period


def test_tick_decay_factor_affects_firing():
    old_decay = getattr(Config, "tick_decay_factor", 1.0)
    old_thresh = getattr(Config, "tick_threshold", 1)
    Config.tick_decay_factor = 0.5
    Config.tick_threshold = 1
    node = Node("A")
    Config.current_tick = 0
    node.schedule_tick(1, 0.0)
    Config.current_tick = 1
    fire, _, reason = node.should_tick(1)
    assert not fire
    assert reason == "count_threshold"
    Config.tick_decay_factor = old_decay
    Config.tick_threshold = old_thresh
