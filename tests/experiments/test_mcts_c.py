"""Verify the MCTS-C policy planner outperforms simple baselines."""

import random

from experiments.policy import (
    MCTS_C,
    boost_eps_emit,
    clamp_Wmax,
    flip_MI_mode,
    toggle_theta_reset,
)


class ToyEnv:
    """Minimal environment exposing policy flags and a residual."""

    def __init__(self, base: float = 10.0) -> None:
        self._base = base
        self.theta_reset = False
        self.eps_emit = 0.0
        self.Wmax = 64.0
        self.MI_mode = False

    @property
    def residual(self) -> float:
        res = self._base
        if self.theta_reset:
            res *= 0.5
        res = max(res - self.eps_emit, 0.0)
        res = max(res - max(0.0, 64.0 - self.Wmax), 0.0)
        if self.MI_mode:
            res += 4.0
        return res

    def clone(self) -> "ToyEnv":
        clone = ToyEnv(self._base)
        clone.theta_reset = self.theta_reset
        clone.eps_emit = self.eps_emit
        clone.Wmax = self.Wmax
        clone.MI_mode = self.MI_mode
        return clone


def _run_plan(env: ToyEnv, plan):
    for act in plan:
        act(env)
    return env.residual


def test_mcts_c_beats_baselines() -> None:
    random.seed(2)
    actions = [toggle_theta_reset, boost_eps_emit, clamp_Wmax, flip_MI_mode]
    planner = MCTS_C(actions=actions, horizon=3, iterations=200)
    env0 = ToyEnv(10.0)

    baseline = _run_plan(env0.clone(), [])

    heur_env = env0.clone()
    for _ in range(3):
        clamp_Wmax(heur_env)
    heuristic = heur_env.residual

    plan = planner.plan(env0.clone())
    mcts_env = env0.clone()
    residual = _run_plan(mcts_env, plan)

    assert residual < heuristic < baseline
