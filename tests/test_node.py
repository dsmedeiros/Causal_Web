import math
import os
import tempfile
from Causal_Web.engine.node import Node
from Causal_Web.engine.graph import CausalGraph
from Causal_Web.config import Config
from Causal_Web.engine import tick_engine


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


def test_global_firing_limit_blocks_nodes():
    old_limit = getattr(Config, "total_max_concurrent_firings", 0)
    Config.total_max_concurrent_firings = 1
    g = CausalGraph()
    g.add_node("A")
    g.add_node("B")
    g.nodes["A"].schedule_tick(1, 0.0)
    g.nodes["B"].schedule_tick(1, 0.0)
    tick_engine.reset_firing_limits()
    Config.current_tick = 1
    g.nodes["A"].maybe_tick(1, g)
    g.nodes["B"].maybe_tick(1, g)
    fired = sum(len(n.tick_history) for n in g.nodes.values())
    assert fired == 1
    Config.total_max_concurrent_firings = old_limit


def test_cluster_firing_limit_blocks_nodes():
    old_limit = getattr(Config, "max_concurrent_firings_per_cluster", 0)
    Config.max_concurrent_firings_per_cluster = 1
    g = CausalGraph()
    g.add_node("A")
    g.add_node("B")
    for n in g.nodes.values():
        n.cluster_ids[0] = 0
    g.nodes["A"].schedule_tick(1, 0.0)
    g.nodes["B"].schedule_tick(1, 0.0)
    tick_engine.reset_firing_limits()
    Config.current_tick = 1
    g.nodes["A"].maybe_tick(1, g)
    g.nodes["B"].maybe_tick(1, g)
    fired = sum(len(n.tick_history) for n in g.nodes.values())
    assert fired == 1
    Config.max_concurrent_firings_per_cluster = old_limit


def test_event_horizon_drop(tmp_path):
    old_dir = Config.output_dir
    old_delay = getattr(Config, "max_cumulative_delay", 25)
    old_coh = getattr(Config, "min_coherence_threshold", 0.2)
    Config.output_dir = tmp_path
    Config.max_cumulative_delay = 2
    Config.min_coherence_threshold = 0.0

    g = CausalGraph()
    g.add_node("A")
    g.add_node("B")
    g.add_edge("A", "B", delay=3)
    Config.current_tick = 0
    g.nodes["A"].apply_tick(0, 0.0, g)
    assert not g.nodes["B"].incoming_phase_queue

    Config.max_cumulative_delay = old_delay
    Config.min_coherence_threshold = old_coh
    Config.output_dir = old_dir


def test_decoherence_drop(tmp_path):
    old_dir = Config.output_dir
    old_delay = getattr(Config, "max_cumulative_delay", 25)
    old_coh = getattr(Config, "min_coherence_threshold", 0.2)
    Config.output_dir = tmp_path
    Config.max_cumulative_delay = 10
    Config.min_coherence_threshold = 0.9

    g = CausalGraph()
    g.add_node("A1")
    g.add_node("A2")
    g.add_node("B")
    g.add_edge("A1", "B", delay=1)
    g.add_edge("A2", "B", delay=1)
    Config.current_tick = 0
    g.nodes["A1"].apply_tick(0, 0.0, g)
    g.nodes["A2"].apply_tick(0, math.pi, g)
    Config.current_tick = 1
    tick_engine.reset_firing_limits()
    g.nodes["B"].maybe_tick(1, g)
    assert len(g.nodes["B"].tick_history) == 0
    assert g.nodes["B"].tick_drop_counts.get("decoherence", 0) == 2

    Config.max_cumulative_delay = old_delay
    Config.min_coherence_threshold = old_coh
    Config.output_dir = old_dir
