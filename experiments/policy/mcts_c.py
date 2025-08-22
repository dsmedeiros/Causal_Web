"""Receding-horizon policy planner using Monte Carlo Tree Search."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import List, Sequence

from .actions import ACTION_SET, Action


@dataclass
class Node:
    """Tree node holding state statistics."""

    state: object
    depth: int
    parent: "Node | None" = None
    action: Action | None = None
    value: float = 0.0
    visits: int = 0
    children: dict[str, "Node"] = field(default_factory=dict)


class MCTS_C:
    """Plan sequences of interventions with progressive widening."""

    def __init__(
        self,
        actions: Sequence[Action] | None = None,
        horizon: int = 2,
        gamma: float = 0.95,
        iterations: int = 100,
        c_ucb: float = 1.0,
        alpha_pw: float = 0.5,
        k_pw: float = 1.0,
    ) -> None:
        self.actions = (
            list(actions) if actions is not None else list(ACTION_SET.values())
        )
        self.horizon = max(2, min(5, horizon))
        self.gamma = gamma
        self.iterations = iterations
        self.c_ucb = c_ucb
        self.alpha_pw = alpha_pw
        self.k_pw = k_pw

    # ------------------------------------------------------------------
    def plan(self, env: object) -> List[Action]:
        """Run a tree search from ``env`` and return an action sequence."""

        root = Node(copy.deepcopy(env), depth=0)
        for _ in range(self.iterations):
            leaf, state = self._select(root)
            reward = self._rollout(state, leaf.depth)
            self._backpropagate(leaf, reward)
        return self._best_plan(root)

    # ------------------------------------------------------------------
    def _select(self, node: Node) -> tuple[Node, object]:
        """Descend the tree applying UCB and progressive widening."""

        state = copy.deepcopy(node.state)
        while node.depth < self.horizon:
            limit = max(1, int(self.k_pw * (node.visits**self.alpha_pw)))
            if len(node.children) < min(limit, len(self.actions)):
                action = random.choice(
                    [a for a in self.actions if a.__name__ not in node.children]
                )
                new_state = copy.deepcopy(state)
                action(new_state)
                child = Node(new_state, node.depth + 1, node, action)
                node.children[action.__name__] = child
                return child, new_state
            if not node.children:
                break
            node = max(
                node.children.values(),
                key=lambda n: n.value / (n.visits or 1)
                + self.c_ucb * math.sqrt(math.log(node.visits + 1) / (n.visits + 1)),
            )
            state = copy.deepcopy(node.state)
        return node, state

    # ------------------------------------------------------------------
    def _rollout(self, env: object, depth: int) -> float:
        """Sample a trajectory from ``env`` to the planning horizon."""

        total = 0.0
        discount = 1.0
        for _ in range(depth, self.horizon):
            total += discount * (-getattr(env, "residual", 0.0))
            action = random.choice(self.actions)
            action(env)
            discount *= self.gamma
        return total

    # ------------------------------------------------------------------
    def _backpropagate(self, node: Node, reward: float) -> None:
        """Propagate ``reward`` up to the root."""

        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
            reward *= self.gamma

    # ------------------------------------------------------------------
    def _best_plan(self, root: Node) -> List[Action]:
        """Extract the best action sequence from the search tree."""

        plan: List[Action] = []
        node = root
        while node.children and len(plan) < self.horizon:
            node = max(node.children.values(), key=lambda n: n.visits)
            if node.action is None:
                break
            plan.append(node.action)
        return plan


__all__ = ["MCTS_C"]
