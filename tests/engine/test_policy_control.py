"""Tests for PolicyControl integration with the engine runtime."""

from Causal_Web.engine.engine_v2.adapter import EngineAdapter


def test_policy_control_applies_actions() -> None:
    engine = EngineAdapter()
    engine.residual = 10.0

    # Single action via "action"
    engine.handle_control({"PolicyControl": {"action": "toggle_theta_reset"}})
    assert engine.theta_reset is True
    status = engine.experiment_status()
    assert status and status["residual"] == 5.0

    # Multiple actions via "actions"
    engine.handle_control(
        {"PolicyControl": {"actions": ["boost_eps_emit", "clamp_Wmax"]}}
    )
    assert engine.eps_emit == 1.0
    assert engine.Wmax == 62.0
    status = engine.experiment_status()
    assert status and status["residual"] == 0.0
