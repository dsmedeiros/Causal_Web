import pytest

from experiments import gates


def test_run_gates_invariants_ok():
    metrics = gates.run_gates({}, [1])
    assert metrics["inv_causality_ok"] is True
    assert metrics["inv_conservation_residual"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["inv_no_signaling_delta"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["inv_ancestry_ok"] is True
    assert metrics["inv_gate_determinism_ok"] is True


def test_run_gates_detects_bad_causality(monkeypatch):
    original = gates.checks.causality

    def bad_causality(_deliveries):
        bad = [{"d_arr": 0.0, "d_src": 1.0}]
        return original(bad)

    monkeypatch.setattr(gates.checks, "causality", bad_causality)
    metrics = gates.run_gates({}, [1])
    assert metrics["inv_causality_ok"] is False


def test_run_gates_detects_conservation_residual(monkeypatch):
    vals = iter([1.0, 3.5])
    monkeypatch.setattr(gates, "_energy_total", lambda: next(vals))
    metrics = gates.run_gates({}, [1])
    assert metrics["inv_conservation_residual"] == pytest.approx(2.5)


def test_run_gates_detects_no_signaling(monkeypatch):
    monkeypatch.setattr(gates, "_gate1_probability", lambda phase: 0.8)
    metrics = gates.run_gates({}, [1])
    assert metrics["inv_no_signaling_delta"] == pytest.approx(0.3)


def test_run_gates_detects_bad_ancestry(monkeypatch):
    original = gates.checks.ancestry_determinism

    def bad_ancestry(_seq):
        bad_seq = [("q1", "h1", "m1"), ("q1", "h2", "m2")]
        return original(bad_seq)

    monkeypatch.setattr(gates.checks, "ancestry_determinism", bad_ancestry)
    metrics = gates.run_gates({}, [1])
    assert metrics["inv_ancestry_ok"] is False


def test_run_gates_detects_non_determinism(monkeypatch):
    vals = iter([0.4, 0.6])
    monkeypatch.setattr(gates, "_gate1_visibility", lambda: next(vals))
    metrics = gates.run_gates({}, [1])
    assert metrics["inv_gate_determinism_ok"] is False
