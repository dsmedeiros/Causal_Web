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
