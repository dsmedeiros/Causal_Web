import pathlib
import time
import json
import yaml

from experiments.ga import GeneticAlgorithm


def test_ga_converges(tmp_path: pathlib.Path) -> None:
    base = {"W0": 1.0, "alpha_leak": 1.0, "lambda_decay": 1.0, "b": 1.0, "prob": 0.5}
    group_ranges = {"Delta_over_W0": (0.0, 1.0)}
    toggles: dict[str, list[int]] = {}

    def fitness(metrics, invariants, groups, toggles):
        return -abs(groups["Delta_over_W0"] - 0.8)

    ga = GeneticAlgorithm(
        base,
        group_ranges,
        toggles,
        [],
        fitness,
        population_size=6,
        seed=1,
    )
    best = ga.run(8)
    assert abs(best.groups["Delta_over_W0"] - 0.8) < 0.25
    out = tmp_path / "best.yaml"
    ga.promote_best(out)
    assert out.exists()


def test_ga_artifacts_link_runs(tmp_path: pathlib.Path, monkeypatch) -> None:
    base = {"W0": 1.0}
    group_ranges = {"x": (0.0, 1.0)}
    toggles: dict[str, list[int]] = {}

    def fitness(metrics, invariants, groups, toggles):
        return -abs(groups["x"] - 0.3)

    from Causal_Web.config import Config

    monkeypatch.chdir(tmp_path)
    Config.output_dir = str(tmp_path)
    (tmp_path / "delta_log.jsonl").write_text("{}\n")
    ga = GeneticAlgorithm(
        base, group_ranges, toggles, [], fitness, population_size=2, seed=0
    )
    ga.run(1)
    ga.save_artifacts("experiments/top_k.json", "experiments/hall_of_fame.json")
    data = json.loads((tmp_path / "experiments/top_k.json").read_text())
    row = data["rows"][0]
    run_dir = tmp_path / "experiments" / row["path"]
    assert (run_dir / "config.json").exists()
    result = json.loads((run_dir / "result.json").read_text())
    assert "fitness" in result
    assert (run_dir / "delta_log.jsonl").exists()
    # hall-of-fame paths point to the same run directories
    hof = json.loads((tmp_path / "experiments/hall_of_fame.json").read_text())
    hof_dir = tmp_path / "experiments" / hof["archive"][0]["path"]
    assert (hof_dir / "config.json").exists()
    assert (hof_dir / "result.json").exists()
    assert (hof_dir / "delta_log.jsonl").exists()


def test_ga_promote_uses_run_config(tmp_path: pathlib.Path, monkeypatch) -> None:
    base = {"W0": 1.0}
    group_ranges = {"x": (0.0, 1.0)}
    toggles: dict[str, list[int]] = {}

    def fitness(metrics, invariants, groups, toggles):
        return -abs(groups["x"] - 0.2)

    from Causal_Web.config import Config

    monkeypatch.chdir(tmp_path)
    Config.output_dir = str(tmp_path)
    (tmp_path / "delta_log.jsonl").write_text("{}\n")
    ga = GeneticAlgorithm(
        base, group_ranges, toggles, [], fitness, population_size=2, seed=0
    )
    ga.run(1)
    best = max(ga.population, key=ga._score)
    run_dir = pathlib.Path("experiments") / (best.run_path or "")
    (run_dir / "config.json").write_text(json.dumps({"y": 5}))
    out = tmp_path / "best.yaml"
    ga.promote_best(out)
    data = yaml.safe_load(out.read_text())
    assert data == {"y": 5}


def test_pareto_front() -> None:
    base = {"W0": 1.0}
    group_ranges = {"x": (0.0, 1.0)}
    toggles: dict[str, list[int]] = {}

    def fitness(metrics, invariants, groups, toggles):
        x = groups["x"]
        return (x, 1 - x)

    ga = GeneticAlgorithm(
        base, group_ranges, toggles, [], fitness, population_size=4, seed=0
    )
    for g in ga.population:
        ga._evaluate(g)
    front = ga.pareto_front()
    assert len(front) >= 2


def test_infeasible_excluded_from_front(monkeypatch) -> None:
    base = {"W0": 1.0}
    group_ranges = {"x": (0.0, 1.0)}
    toggles: dict[str, list[int]] = {}

    def fitness(metrics, invariants, groups, toggles):
        x = groups["x"]
        return (x, 1 - x)

    ga = GeneticAlgorithm(
        base, group_ranges, toggles, [], fitness, population_size=4, seed=0
    )
    bad = ga.population[0]
    real_eval = ga._evaluate

    def fake_eval(genome):  # type: ignore[override]
        if genome is bad:
            genome.fitness = None
            genome.invariants = {"inv_causality_ok": False}
            return None
        return real_eval(genome)

    monkeypatch.setattr(ga, "_evaluate", fake_eval)
    ga.step()
    assert bad not in ga.pareto_front()


def test_pareto_archive_shrinks() -> None:
    base = {"W0": 1.0}
    group_ranges = {"x": (0.0, 1.0)}
    toggles: dict[str, list[int]] = {}

    def fitness(metrics, invariants, groups, toggles):
        x = groups["x"]
        return (x, 1 - x)

    ga = GeneticAlgorithm(
        base, group_ranges, toggles, [], fitness, population_size=6, seed=1
    )
    ga.step()
    size1 = len(ga.pareto_front())
    ga.step()
    size2 = len(ga.pareto_front())
    assert size2 <= size1


def test_epsilon_dominance_prunes_archive() -> None:
    base = {"W0": 1.0}
    group_ranges = {"x": (0.0, 1.0)}
    toggles: dict[str, list[int]] = {}

    def fitness(metrics, invariants, groups, toggles):
        x = groups["x"]
        return (x, 1 - x)

    ga = GeneticAlgorithm(
        base,
        group_ranges,
        toggles,
        [],
        fitness,
        population_size=10,
        seed=0,
        archive_eps=0.6,
    )
    ga.step()
    assert len(ga.pareto_front()) <= 3


def test_promote_pareto(tmp_path: pathlib.Path) -> None:
    base = {"W0": 1.0}
    group_ranges = {"x": (0.0, 1.0)}
    toggles: dict[str, list[int]] = {}

    def fitness(metrics, invariants, groups, toggles):
        x = groups["x"]
        return (x, 1 - x)

    ga = GeneticAlgorithm(
        base, group_ranges, toggles, [], fitness, population_size=4, seed=0
    )
    ga.step()
    out = tmp_path / "pareto.yaml"
    ga.promote_pareto(0, out)
    assert out.exists()


def test_checkpoint_resume(tmp_path: pathlib.Path) -> None:
    base = {"W0": 1.0}
    group_ranges = {"x": (0.0, 1.0)}
    toggles: dict[str, list[int]] = {}

    def fitness(metrics, invariants, groups, toggles):
        return -abs(groups["x"] - 0.5)

    ga_ref = GeneticAlgorithm(
        base, group_ranges, toggles, [], fitness, population_size=4, seed=123
    )
    best_ref = ga_ref.run(3)

    ga = GeneticAlgorithm(
        base, group_ranges, toggles, [], fitness, population_size=4, seed=123
    )
    ga.run(2)
    ckpt = tmp_path / "ga.pkl"
    ga.save_checkpoint(ckpt)

    ga_loaded = GeneticAlgorithm.load_checkpoint(ckpt, fitness)
    best_loaded = ga_loaded.run(1)

    assert ga_loaded.history == ga_ref.history
    assert best_loaded.groups == best_ref.groups


def test_threadsafe_eval_deterministic() -> None:
    """Stress test thread-safe evaluation and deterministic metrics."""

    import asyncio
    import threading
    import random

    base = {"W0": 1.0}
    group_ranges = {"x": (0.0, 1.0)}
    toggles: dict[str, list[int]] = {}

    def fitness(metrics, invariants, groups, toggles):
        return metrics["m"]

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

    ga = GeneticAlgorithm(
        base,
        group_ranges,
        toggles,
        [],
        fitness,
        population_size=1,
        seed=0,
        client=None,
        loop=loop,
    )

    class DummyClient:
        def __init__(self, ga: GeneticAlgorithm) -> None:
            self.ga = ga

        async def send(self, msg: dict) -> None:
            rid = msg["ExperimentControl"]["id"]
            seed = msg["ExperimentControl"]["config"]["seed"]
            rng = random.Random(seed)
            metric = rng.random()
            await asyncio.sleep(0)
            ga.handle_status(
                {
                    "id": rid,
                    "state": "finished",
                    "metrics": {"m": metric},
                    "invariants": {
                        "inv_causality_ok": True,
                        "inv_ancestry_ok": True,
                        "inv_conservation_residual": 0.0,
                        "inv_no_signaling_delta": 0.0,
                    },
                }
            )

    ga._client = DummyClient(ga)

    genome = ga.population[0]
    first = ga.evaluate_blocking(genome)
    for _ in range(200):
        res = ga.evaluate_blocking(genome)
        assert res["status"] == "ok"
        assert res["metrics"] == first["metrics"]

    loop.call_soon_threadsafe(loop.stop)
    thread.join()


def test_run_index_resume_skips_duplicates(tmp_path: pathlib.Path, monkeypatch) -> None:
    base = {"W0": 1.0}
    group_ranges = {"x": (0.0, 1.0)}
    toggles: dict[str, list[int]] = {}

    def fitness(metrics, invariants, groups, toggles):
        return -abs(groups["x"] - 0.3)

    monkeypatch.chdir(tmp_path)
    ga = GeneticAlgorithm(
        base, group_ranges, toggles, [], fitness, population_size=1, seed=0
    )
    genome = ga.population[0]
    ga._evaluate(genome)

    manifests = list((tmp_path / "experiments" / "runs").rglob("manifest.json"))
    assert len(manifests) == 1

    # Simulate crash by removing index and creating a new GA
    (tmp_path / "experiments" / "runs" / "index.json").unlink()
    ga2 = GeneticAlgorithm(
        base, group_ranges, toggles, [], fitness, population_size=1, seed=0
    )
    genome2 = ga2.population[0]
    ga2._evaluate(genome2)

    manifests2 = list((tmp_path / "experiments" / "runs").rglob("manifest.json"))
    assert len(manifests2) == 1


def test_force_rerun_creates_new_manifest(tmp_path: pathlib.Path, monkeypatch) -> None:
    base = {"W0": 1.0}
    group_ranges = {"x": (0.0, 1.0)}
    toggles: dict[str, list[int]] = {}

    def fitness(metrics, invariants, groups, toggles):
        return -abs(groups["x"] - 0.3)

    monkeypatch.chdir(tmp_path)
    ga = GeneticAlgorithm(
        base, group_ranges, toggles, [], fitness, population_size=1, seed=0
    )
    genome = ga.population[0]
    ga._evaluate(genome)

    ga_force = GeneticAlgorithm(
        base,
        group_ranges,
        toggles,
        [],
        fitness,
        population_size=1,
        seed=0,
        force=True,
    )
    genome2 = ga_force.population[0]
    ga_force._evaluate(genome2)

    manifests = list((tmp_path / "experiments" / "runs").rglob("manifest.json"))
    assert len(manifests) == 2


def test_checkpoint_persists_pending(tmp_path: pathlib.Path) -> None:
    """Pending evaluations are saved and resubmitted on load."""

    import asyncio
    import threading
    import random

    base = {"W0": 1.0}
    group_ranges = {"x": (0.0, 1.0)}
    toggles: dict[str, list[int]] = {}

    def fitness(metrics, invariants, groups, toggles):
        return metrics["m"]

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

    ga = GeneticAlgorithm(
        base,
        group_ranges,
        toggles,
        [],
        fitness,
        population_size=1,
        seed=0,
        client=None,
        loop=loop,
    )

    class DummyClient:
        def __init__(self) -> None:
            self.sent: list[dict] = []

        async def send(self, msg: dict) -> None:
            self.sent.append(msg)

    client = DummyClient()
    ga._client = client
    genome = ga.population[0]

    asyncio.run_coroutine_threadsafe(
        ga._async_eval_genome(genome, ga._seed_for_genome(genome)), loop
    )

    time.sleep(0.01)
    ckpt = tmp_path / "ga.pkl"
    ga.save_checkpoint(ckpt)

    ga_loaded = GeneticAlgorithm.load_checkpoint(
        ckpt, fitness, client=client, loop=loop
    )
    time.sleep(0.01)

    assert len(client.sent) == 2

    msg = client.sent[-1]
    rid = msg["ExperimentControl"]["id"]
    rng = random.Random(msg["ExperimentControl"]["config"]["seed"])
    metric = rng.random()
    ga_loaded.handle_status(
        {
            "id": rid,
            "state": "finished",
            "metrics": {"m": metric},
            "invariants": {
                "inv_causality_ok": True,
                "inv_ancestry_ok": True,
                "inv_conservation_residual": 0.0,
                "inv_no_signaling_delta": 0.0,
            },
        }
    )

    assert not ga_loaded._pending

    loop.call_soon_threadsafe(loop.stop)
    thread.join()
