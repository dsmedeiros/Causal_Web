import pathlib

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
    import time

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
