from experiments import DOEQueueManager


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


def test_lhs_enqueue_and_run():
    mgr = DOEQueueManager(_base_config(), [1])
    mgr.enqueue_lhs({"Delta_over_W0": (0.5, 1.0)}, samples=2)
    assert len(mgr.runs) == 2
    mgr.run_all()
    for _, status in mgr.runs:
        assert status.state == "finished"
        assert status.invariants["inv_causality_ok"]


def test_grid_enqueue():
    mgr = DOEQueueManager(_base_config(), [1])
    mgr.enqueue_grid({"Delta_over_W0": (0.0, 1.0)}, {"Delta_over_W0": 3})
    assert len(mgr.runs) == 3
