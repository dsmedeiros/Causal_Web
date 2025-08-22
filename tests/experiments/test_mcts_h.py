import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from experiments import OptimizerQueueManager
from experiments.ga import GeneticAlgorithm
from experiments.optim import MCTS_H, build_priors
from experiments.optim.priors import DiscretePrior, GaussianPrior


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


def test_build_priors_robust_gaussian():
    rows = [{"a": 0.0}, {"a": 1.0}, {"a": 10.0}]
    priors = build_priors(rows)
    assert isinstance(priors["a"], GaussianPrior)
    assert priors["a"].mu == pytest.approx(1.0)
    assert priors["a"].sigma == pytest.approx(1.4826, rel=1e-3)


def test_online_prior_update():
    priors = build_priors([{"a": 0.0}])
    opt = MCTS_H(["a"], priors, {"rng_seed": 0})
    opt._update_priors({"a": 0.5}, 0.1)
    assert isinstance(opt.priors["a"], GaussianPrior)
    assert opt.priors["a"].mu == pytest.approx(0.25)
    assert opt.priors["a"].sigma == pytest.approx(0.37065, rel=1e-3)


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


def test_proxy_non_promotion_penalises_leaf():
    priors = {"a": DiscretePrior([0, 1], [0.5, 0.5])}
    opt = MCTS_H(["a"], priors, {"rng_seed": 0, "promote_threshold": 0.5})
    cfg = opt.suggest(1)[0]
    opt.observe([{"config": cfg, "fitness_proxy": 0.9}])
    assert not opt._pending_full
    assert opt.root.children[0].Q <= -1e8


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


def test_optimizer_queue_multi_objective_archive(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    priors = {"a": DiscretePrior([0.0], [1.0])}
    opt = MCTS_H(["a"], priors, {"multi_objective": True})

    def fit_fn(metrics, inv, groups):
        val = groups["a"]
        return (val, val**2)

    mgr = OptimizerQueueManager(
        _base_config(), [1], fit_fn, opt, proxy_frames=1, full_frames=1
    )
    res1 = mgr.run_next()
    assert res1 and res1.status == "proxy"
    res2 = mgr.run_next()
    assert res2 and res2.status == "full"
    hof = json.loads(Path("experiments/hall_of_fame.json").read_text())
    entry = hof["archive"][0]
    assert entry.get("origin") == "mcts"
    assert set(entry["objectives"].keys()) == {"f0", "f1"}


def _slow_gates(raw, gates, frames=1):
    time.sleep(0.05)
    return {}


@pytest.mark.parametrize("backend", ["process", "ray"])
def test_optimizer_queue_parallel(tmp_path, monkeypatch, backend):
    monkeypatch.chdir(tmp_path)
    priors = {"a": DiscretePrior([0.0, 0.5], [0.5, 0.5])}

    def fitness_fn(metrics, inv, groups):
        return groups["a"]

    base = _base_config()

    evals = 8
    if backend == "process":
        monkeypatch.setattr("experiments.queue.run_gates", _slow_gates)
        opt_seq = MCTS_H(["a"], priors, {"rng_seed": 0})
        mgr_seq = OptimizerQueueManager(base, [1], fitness_fn, opt_seq)
        start = time.perf_counter()
        for _ in range(evals):
            mgr_seq.run_next()
        seq_time = time.perf_counter() - start

    opt_par1 = MCTS_H(["a"], priors, {"rng_seed": 0})
    mgr_par1 = OptimizerQueueManager(base, [1], fitness_fn, opt_par1)
    start = time.perf_counter()
    if backend == "ray":
        ray = pytest.importorskip("ray")
        repo = Path(__file__).resolve().parents[2]
        ray.init(runtime_env={"env_vars": {"PYTHONPATH": str(repo)}})
        res1 = mgr_par1.run_parallel(evals, parallel=2, use_ray=True)
        par_time = time.perf_counter() - start
        ray.shutdown()
    else:
        res1 = mgr_par1.run_parallel(evals, parallel=2, use_processes=True)
        par_time = time.perf_counter() - start

    opt_par2 = MCTS_H(["a"], priors, {"rng_seed": 0})
    mgr_par2 = OptimizerQueueManager(base, [1], fitness_fn, opt_par2)
    if backend == "ray":
        ray.init(runtime_env={"env_vars": {"PYTHONPATH": str(repo)}})
        res2 = mgr_par2.run_parallel(evals, parallel=2, use_ray=True)
        ray.shutdown()
    else:
        res2 = mgr_par2.run_parallel(evals, parallel=2, use_processes=True)

    assert [r.config for r in res1] == [r.config for r in res2]
    if backend == "process":
        assert par_time <= seq_time


def test_optimizer_queue_parallel_rungs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    priors = {"a": DiscretePrior([0.0, 0.5], [0.5, 0.5])}

    def fitness_fn(metrics, inv, groups):
        return groups["a"]

    base = _base_config()
    monkeypatch.setattr("experiments.queue.run_gates", _slow_gates)
    evals = 6

    opt_par1 = MCTS_H(["a"], priors, {"rng_seed": 0})
    mgr_par1 = OptimizerQueueManager(
        base, [1], fitness_fn, opt_par1, full_frames=10, rung_fractions=[0.5, 1.0]
    )
    res1 = mgr_par1.run_parallel(evals, parallel=2, use_processes=True)

    opt_par2 = MCTS_H(["a"], priors, {"rng_seed": 0})
    mgr_par2 = OptimizerQueueManager(
        base, [1], fitness_fn, opt_par2, full_frames=10, rung_fractions=[0.5, 1.0]
    )
    res2 = mgr_par2.run_parallel(evals, parallel=2, use_processes=True)

    assert [r.config for r in res1] == [r.config for r in res2]
    assert [r.status for r in res1] == [r.status for r in res2]
    stats1 = mgr_par1.rung_stats()
    stats2 = mgr_par2.rung_stats()
    assert stats1["rung_counts"] == stats2["rung_counts"]
    assert stats1["promotion_fractions"] == stats2["promotion_fractions"]


@pytest.mark.parametrize("frame_time", [0.001, 0.002])
def test_mcts_h_asha_scheduler(tmp_path, monkeypatch, frame_time):
    """ASHA reduces full evaluations while retaining fitness across workloads."""

    monkeypatch.chdir(tmp_path)
    priors = {"Delta_over_W0": DiscretePrior([0.0, 0.5], [0.5, 0.5])}

    def fitness_fn(metrics, inv, groups):
        return groups["Delta_over_W0"]

    base = _base_config()

    def fake_run_gates(raw, gates, frames=1):
        time.sleep(frames * frame_time)
        return {}

    monkeypatch.setattr("experiments.queue.run_gates", fake_run_gates)

    # Baseline without ASHA: all evaluations are full
    start = time.perf_counter()
    opt_base = MCTS_H(["Delta_over_W0"], priors, {"rng_seed": 0})
    mgr_base = OptimizerQueueManager(base, [1], fitness_fn, opt_base, full_frames=50)
    full_base = 0
    best_base = 1.0
    for _ in range(8):
        res = mgr_base.run_next()
        assert res
        if res.status == "full":
            full_base += 1
            best_base = min(best_base, res.fitness or 1.0)
    base_time = time.perf_counter() - start

    # With ASHA rungs 20%, 40%, 100%
    start = time.perf_counter()
    opt_asha = MCTS_H(["Delta_over_W0"], priors, {"rng_seed": 0})
    mgr_asha = OptimizerQueueManager(
        base,
        [1],
        fitness_fn,
        opt_asha,
        full_frames=50,
        rung_fractions=[0.2, 0.4, 1.0],
    )
    full_asha = 0
    best_asha = 1.0
    for _ in range(8):
        res = mgr_asha.run_next()
        assert res
        if res.status == "full":
            full_asha += 1
            best_asha = min(best_asha, res.fitness or 1.0)
    asha_time = time.perf_counter() - start

    assert full_asha <= max(1, full_base // 2)
    assert best_asha <= best_base + 1e-9
    assert asha_time <= base_time
    stats = mgr_asha.rung_stats()
    assert stats["rung_counts"][0] >= stats["rung_counts"][1] >= stats["rung_counts"][2]
    assert "rung_times" in stats and len(stats["rung_times"]) == len(
        stats["rung_counts"]
    )


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


def test_generation_promotion_rate_improvement():
    priors = {"a": DiscretePrior([0, 1], [0.5, 0.5])}
    opt = MCTS_H(["a"], priors, {"rng_seed": 0, "promote_threshold": 0.5})
    cfg = opt.suggest(1)[0]
    opt.observe([{"config": cfg, "fitness_proxy": 0.6}])
    m1 = opt.metrics()
    assert m1["promotion_rate_gen"] == pytest.approx(0.0)
    cfg = opt.suggest(1)[0]
    opt.observe([{"config": cfg, "fitness_proxy": 0.4}])
    m2 = opt.metrics()
    assert m2["promotion_rate_gen"] == pytest.approx(1.0)
    assert m2["promotion_rate_improvement"] == pytest.approx(1.0)


def test_generation_best_so_far_improvement():
    priors = {"a": DiscretePrior([0], [1.0])}
    opt = MCTS_H(["a"], priors, {"rng_seed": 0})
    cfg = opt.suggest(1)[0]
    opt.observe([{"config": cfg, "fitness": 1.0}])
    cfg = opt.suggest(1)[0]
    opt.observe([{"config": cfg, "fitness": 0.5}])
    m = opt.metrics()
    assert m["best_so_far"] == pytest.approx(0.5)
    assert m["best_so_far_improvement"] == pytest.approx(0.5)
