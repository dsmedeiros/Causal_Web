import json
import os
from pathlib import Path

import numpy as np

from experiments import OptimizerQueueManager
from experiments.ga import GeneticAlgorithm
from experiments.optim import MCTS_H, build_priors
from experiments.optim.priors import DiscretePrior


def test_progressive_widening():
    priors = {
        "a": DiscretePrior([0, 1], [0.5, 0.5]),
        "b": DiscretePrior([0, 1], [0.5, 0.5]),
    }
    cfg = {"alpha_pw": 0.5, "k_pw": 2.0, "rng_seed": 0}
    opt = MCTS_H(["a", "b"], priors, cfg)
    for _ in range(20):
        cfgs = opt.suggest(1)
        opt.observe([{"config": cfgs[0], "fitness": 0.0}])
    limit = cfg["k_pw"] * (opt.root.N ** cfg["alpha_pw"])
    assert len(opt.root.children) <= int(limit + 1e-9)


def test_backprop_updates():
    priors = {"a": DiscretePrior([0, 1], [0.5, 0.5])}
    opt = MCTS_H(["a"], priors, {"rng_seed": 0})
    cfgs = opt.suggest(1)
    opt.observe([{"config": cfgs[0], "fitness": 1.0}])
    assert opt.root.Q == -1.0
    assert opt.root.N == 1


def test_build_priors_discrete():
    rows = [{"a": 0}, {"a": 1}, {"a": 1}]
    priors = build_priors(rows)
    assert "a" in priors
    rng = np.random.default_rng(0)
    samples = [priors["a"].sample(rng) for _ in range(100)]
    assert samples.count(1.0) > samples.count(0.0)


def test_build_priors_quantile_bins():
    rows = [{"a": 0.0}, {"a": 0.5}, {"a": 1.0}]
    priors = build_priors(rows, bins=2)
    assert "a" in priors
    rng = np.random.default_rng(0)
    samples = {priors["a"].sample(rng) for _ in range(20)}
    assert len(samples) == 2


def test_transposition_reuses_nodes():
    priors = {
        "a": DiscretePrior([0], [1.0]),
        "b": DiscretePrior([0], [1.0]),
    }
    opt = MCTS_H(["a", "b"], priors, {"rng_seed": 0})
    first = opt.suggest(1)[0]
    nodes_before = opt._nodes
    second = opt.suggest(1)[0]
    assert first == second
    assert nodes_before == opt._nodes


def test_proxy_promotion_requeues_config():
    priors = {"a": DiscretePrior([0, 1], [0.5, 0.5])}
    opt = MCTS_H(["a"], priors, {"rng_seed": 0, "promote_threshold": 0.5})
    cfg = opt.suggest(1)[0]
    opt.observe([{"config": cfg, "fitness_proxy": 0.4}])
    cfg2 = opt.suggest(1)[0]
    assert cfg2 == cfg
    opt.observe([{"config": cfg2, "fitness": 0.2}])
    assert opt.root.N > 0


def test_quantile_promotion():
    priors = {"a": DiscretePrior([0, 1], [0.5, 0.5])}
    opt = MCTS_H(["a"], priors, {"rng_seed": 0, "promote_quantile": 0.5})
    cfg1 = opt.suggest(1)[0]
    opt.observe([{"config": cfg1, "fitness_proxy": 1.0}])
    cfg2 = opt.suggest(1)[0]
    opt.observe([{"config": cfg2, "fitness_proxy": 0.2}])
    cfg3 = opt.suggest(1)[0]
    assert cfg3 == cfg2


def test_quantile_window_promotion():
    priors = {"a": DiscretePrior([0], [1.0])}
    opt = MCTS_H(
        ["a"], priors, {"rng_seed": 0, "promote_quantile": 0.5, "promote_window": 2}
    )
    cfg = opt.suggest(1)[0]
    opt.observe([{"config": cfg, "fitness_proxy": 1.0}])
    opt._pending_full.clear()
    cfg = opt.suggest(1)[0]
    opt.observe([{"config": cfg, "fitness_proxy": 0.8}])
    opt._pending_full.clear()
    cfg = opt.suggest(1)[0]
    opt.observe([{"config": cfg, "fitness_proxy": 0.2}])
    opt._pending_full.clear()
    cfg = opt.suggest(1)[0]
    opt.observe([{"config": cfg, "fitness_proxy": 0.6}])
    assert not opt._pending_full


def test_state_roundtrip(tmp_path):
    priors = {"a": DiscretePrior([0, 1], [0.5, 0.5])}
    cfg = {"rng_seed": 0}
    opt = MCTS_H(["a"], priors, cfg)
    first = opt.suggest(1)[0]
    opt.observe([{"config": first, "fitness": 0.1}])
    state = tmp_path / "mcts.json"
    opt.save(state)
    next1 = opt.suggest(1)[0]
    opt2 = MCTS_H.load(state, priors)
    next2 = opt2.suggest(1)[0]
    assert next1 == next2


def test_multi_objective_scalarisation():
    priors = {"a": DiscretePrior([0], [1.0])}
    cfg = {"rng_seed": 0, "multi_objective": True}
    opt = MCTS_H(["a"], priors, cfg)
    cfgs = opt.suggest(1)
    clone = np.random.default_rng()
    clone.bit_generator.state = opt.rng.bit_generator.state
    expected = -float(np.dot(clone.dirichlet([1.0, 1.0]), [1.0, 2.0]))
    opt.observe(
        [
            {
                "config": cfgs[0],
                "objectives": {"f1": 1.0, "f2": 2.0},
            }
        ]
    )
    assert opt.root.N == 1
    assert np.isclose(opt.root.Q, expected)


def _base_config():
    return {
        "W0": 1.0,
        "Delta": 1.0,
        "alpha_leak": 1.0,
        "alpha_d": 1.0,
        "lambda_decay": 1.0,
        "sigma_reinforce": 1.0,
        "a": 1.0,
        "b": 1.0,
        "eta": 1.0,
    }


def test_optimizer_queue_promotion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    priors = {"Delta_over_W0": DiscretePrior([0.2], [1.0])}
    opt = MCTS_H(["Delta_over_W0"], priors, {"rng_seed": 0, "promote_threshold": 1.0})

    def fitness_fn(metrics, inv, groups):
        return groups["Delta_over_W0"]

    mgr = OptimizerQueueManager(_base_config(), [1], fitness_fn, opt)
    res1 = mgr.run_next()
    assert res1 and res1.status == "proxy"
    res2 = mgr.run_next()
    assert res2 and res2.status == "full"
    data = json.loads(Path("experiments/top_k.json").read_text())
    assert data["rows"][0]["groups"]["Delta_over_W0"] == 0.2
    manifest = json.loads(
        (Path("experiments") / res2.path / "manifest.json").read_text()
    )
    assert manifest.get("mcts_run_id")


def test_mcts_beats_ga(tmp_path, monkeypatch):
    """MCTS-H matches GA fitness with fewer full evaluations."""

    base = _base_config()
    optimum = 0.8

    ga_dir = tmp_path / "ga"
    mcts_dir = tmp_path / "mcts"
    ga_dir.mkdir()
    mcts_dir.mkdir()

    from Causal_Web.config import Config

    monkeypatch.chdir(tmp_path)

    # GA baseline
    os.chdir(ga_dir)
    Config.output_dir = str(ga_dir)
    group_ranges = {"Delta_over_W0": (0.0, 1.0)}
    ga_evals = 0

    def ga_fitness(metrics, inv, groups, toggles):
        nonlocal ga_evals
        ga_evals += 1
        return -abs(groups["Delta_over_W0"] - optimum)

    ga = GeneticAlgorithm(
        base, group_ranges, {}, [], ga_fitness, population_size=4, seed=0
    )
    best = ga.run(5)
    ga_best = abs(best.groups["Delta_over_W0"] - optimum)

    # MCTS-H run under the same budget
    os.chdir(mcts_dir)
    Config.output_dir = str(mcts_dir)
    priors = {"Delta_over_W0": DiscretePrior([0.2, optimum], [0.5, 0.5])}
    opt = MCTS_H(["Delta_over_W0"], priors, {"rng_seed": 0, "promote_threshold": 0.1})

    def mcts_fitness(metrics, inv, groups):
        return abs(groups["Delta_over_W0"] - optimum)

    mgr = OptimizerQueueManager(
        base, [1], mcts_fitness, opt, proxy_frames=1, full_frames=1
    )
    full = 0
    best_err = float("inf")
    for _ in range(ga_evals):
        res = mgr.run_next()
        assert res is not None
        if res.status == "full":
            full += 1
            best_err = min(best_err, res.fitness)

    os.chdir(tmp_path)

    assert full < ga_evals
    assert best_err <= ga_best
