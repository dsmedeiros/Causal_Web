from __future__ import annotations

"""Expose the MCTS-C policy planner to QML panels."""

import asyncio
from typing import Dict, List, Optional

from ..qt import QObject, Property, Signal, Slot

from experiments.policy import ACTION_ICONS, ACTION_SET, MCTS_C
from ..ipc import Client


class _Env:
    """Toy environment with policy flags and a residual signal."""

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

    def clone(self) -> "_Env":
        clone = _Env(self._base)
        clone.theta_reset = self.theta_reset
        clone.eps_emit = self.eps_emit
        clone.Wmax = self.Wmax
        clone.MI_mode = self.MI_mode
        return clone


class PolicyModel(QObject):
    """Run receding-horizon plans over discrete engine toggles."""

    planChanged = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._horizon = 2
        self._enabled: Dict[str, bool] = {name: True for name in ACTION_SET}
        self._plan: List[str] = []
        self._client: Optional[Client] = None

    # ------------------------------------------------------------------
    @Property("QStringList", constant=True)
    def actionNames(self) -> List[str]:
        return list(ACTION_SET.keys())

    # ------------------------------------------------------------------
    @Property(int, notify=planChanged)
    def horizon(self) -> int:
        return self._horizon

    @horizon.setter
    def horizon(self, value: int) -> None:
        value = int(max(2, min(5, value)))
        if value != self._horizon:
            self._horizon = value
            self.planChanged.emit()

    # ------------------------------------------------------------------
    @Property(str, notify=planChanged)
    def planSummary(self) -> str:
        """Return the planned sequence as a human-readable string."""

        return ", ".join(self._plan)

    # ------------------------------------------------------------------
    @Property("QStringList", notify=planChanged)
    def planSteps(self) -> List[str]:
        """Expose the raw action names for timeline overlays."""

        return list(self._plan)

    # ------------------------------------------------------------------
    @Property("QStringList", notify=planChanged)
    def planIcons(self) -> List[str]:
        """Return a list of icon codes matching ``planSteps``."""

        return [ACTION_ICONS.get(name, "") for name in self._plan]

    # ------------------------------------------------------------------
    @Slot(str, bool)
    def setActionEnabled(self, name: str, enabled: bool) -> None:
        if name in self._enabled:
            self._enabled[name] = enabled

    # ------------------------------------------------------------------
    @Slot(str, result=str)
    def iconFor(self, name: str) -> str:
        """Lookup an icon for ``name`` to use in delegates."""

        return ACTION_ICONS.get(name, "")

    # ------------------------------------------------------------------
    @Slot()
    def plan(self) -> None:
        actions = [ACTION_SET[n] for n, e in self._enabled.items() if e]
        env = _Env()
        planner = MCTS_C(actions=actions, horizon=self._horizon, iterations=200)
        seq = planner.plan(env)
        self._plan = [a.__name__ for a in seq]
        self.planChanged.emit()

    # ------------------------------------------------------------------
    def set_client(self, client: Optional[Client]) -> None:
        """Attach a WebSocket ``client`` for dispatching actions."""

        self._client = client

    # ------------------------------------------------------------------
    @Slot()
    def apply(self) -> None:
        """Send the planned action sequence to the engine runtime."""

        if self._client and self._plan:
            asyncio.create_task(
                self._client.send({"PolicyControl": {"actions": self._plan}})
            )
