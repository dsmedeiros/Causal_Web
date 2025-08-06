from __future__ import annotations

from typing import Tuple

from ..engine.models.node import Node
from ..engine import scheduler


def run_demo(
    total_time: float = 10.0, dt: float = 1.0, velocity: float = 0.8
) -> Tuple[float, float]:
    """Simulate twin-paradox style time dilation.

    Parameters
    ----------
    total_time:
        Duration of the journey in coordinate time.
    dt:
        Time step used by the scheduler.
    velocity:
        Constant speed of the travelling twin in spatial units per ``dt``.

    Returns
    -------
    Tuple[float, float]
        Proper times ``(tau_home, tau_traveller)`` accumulated by each twin.
    """

    home = Node("home")
    traveller = Node("traveller")
    half_steps = int(total_time / (2 * dt))
    for _ in range(half_steps):
        traveller.x += velocity * dt
        scheduler.step([home, traveller], dt)
    for _ in range(half_steps):
        traveller.x -= velocity * dt
        scheduler.step([home, traveller], dt)
    return home.tau, traveller.tau


if __name__ == "__main__":
    tau_home, tau_traveller = run_demo()
    asym = (tau_home - tau_traveller) / tau_home if tau_home else 0.0
    print(
        f"home tau: {tau_home:.3f} traveller tau: {tau_traveller:.3f} (asymmetry {asym*100:.1f}%)"
    )
