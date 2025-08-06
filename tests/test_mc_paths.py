from __future__ import annotations

import random

import networkx as nx

from Causal_Web.engine.analysis.mc_paths import (
    enumerate_path_integral,
    monte_carlo_path_integral,
)


def build_test_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    edges = [
        ("s", "a", {"delay": 1, "phase": 0.1, "atten": 0.9}),
        ("a", "t", {"delay": 1, "phase": 0.2, "atten": 0.8}),
        ("s", "b", {"delay": 2, "phase": 0.0, "atten": 0.7}),
        ("b", "t", {"delay": 1, "phase": 0.3, "atten": 0.6}),
        ("a", "b", {"delay": 1, "phase": 0.5, "atten": 0.5}),
        ("b", "a", {"delay": 1, "phase": 0.4, "atten": 0.5}),
    ]
    g.add_edges_from(edges)
    return g


def test_mc_estimator_matches_full_enumeration():
    g = build_test_graph()
    exact = enumerate_path_integral(g, "s", "t")
    n_paths = sum(1 for _ in nx.all_simple_paths(g, "s", "t"))
    estimate = monte_carlo_path_integral(
        g,
        "s",
        "t",
        k=n_paths,
        samples=5000,
        rng=random.Random(1),
    )
    rel_err = abs(estimate - exact) / abs(exact)
    assert rel_err < 0.02
