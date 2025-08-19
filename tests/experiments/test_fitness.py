from experiments.fitness import scalar_fitness, vector_fitness
import pytest


def test_scalar_fitness_monotone() -> None:
    metrics = {"target_success": 0.8, "coherence": 0.9, "G6_CHSH": 2.2}
    invariants = {
        "inv_conservation_residual": 0.2,
        "inv_no_signaling_delta": 0.1,
        "inv_causality_ok": True,
        "inv_ancestry_ok": True,
    }
    fit1 = scalar_fitness(
        metrics, invariants, residual_threshold=1.0, ns_delta_threshold=1.0
    )
    invariants["inv_conservation_residual"] = 0.1
    fit2 = scalar_fitness(
        metrics, invariants, residual_threshold=1.0, ns_delta_threshold=1.0
    )
    assert fit2 > fit1


def test_scalar_fitness_rejects_invalid() -> None:
    metrics = {"target_success": 0.0, "coherence": 0.0, "G6_CHSH": float("nan")}
    invariants = {
        "inv_conservation_residual": 0.0,
        "inv_no_signaling_delta": 0.0,
        "inv_causality_ok": True,
        "inv_ancestry_ok": True,
    }
    with pytest.raises(ValueError):
        scalar_fitness(metrics, invariants)


def test_vector_fitness_guardrails() -> None:
    metrics = {"target_success": 0.5, "G6_CHSH": 2.1}
    invariants = {
        "inv_conservation_residual": 0.2,
        "inv_no_signaling_delta": 0.1,
        "inv_causality_ok": True,
        "inv_ancestry_ok": True,
    }
    vec = vector_fitness(
        metrics, invariants, residual_threshold=1.0, ns_delta_threshold=1.0
    )
    assert len(vec) == 3
    assert all(0.0 <= v <= 1.0 for v in vec)
    metrics["G6_CHSH"] = float("nan")
    with pytest.raises(ValueError):
        vector_fitness(metrics, invariants)
