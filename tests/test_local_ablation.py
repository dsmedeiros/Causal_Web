from experiments.ablation import local_ablation
import pytest


def test_local_ablation_1d():
    best = {"x": 1.0, "y": 2.0}

    def evaluate(cfg):
        return (cfg["x"] - 1.0) ** 2 + cfg["y"]

    res = local_ablation(best, evaluate, ["x"], span=1.0, steps=3)
    assert res.params == ["x"]
    assert res.values[0] == pytest.approx([0.0, 1.0, 2.0])
    expected = [3.0, 2.0, 3.0]
    assert [s[0] for s in res.scores] == pytest.approx(expected)


def test_local_ablation_2d():
    best = {"x": 1.0, "y": 2.0}

    def evaluate(cfg):
        return cfg["x"] + cfg["y"]

    res = local_ablation(best, evaluate, ["x", "y"], span=0.5, steps=2)
    assert res.params == ["x", "y"]
    assert res.values[0] == pytest.approx([0.5, 1.5])
    assert res.values[1] == pytest.approx([1.0, 3.0])
    assert res.scores == [[1.5, 3.5], [2.5, 4.5]]
