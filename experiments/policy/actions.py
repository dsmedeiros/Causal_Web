"""Discrete engine intervention actions for policy planning.

Each action mutates simple engine flags instead of touching the residual
directly. The engine is then responsible for recomputing the residual based
on the updated state.
"""

from __future__ import annotations

from typing import Callable, Protocol


class Env(Protocol):
    """Minimal environment protocol for policy actions."""

    theta_reset: bool
    eps_emit: float
    Wmax: float
    MI_mode: bool


Action = Callable[[Env], None]


def toggle_theta_reset(env: Env) -> None:
    """Toggle the ``theta_reset`` flag."""

    env.theta_reset = not env.theta_reset


def boost_eps_emit(env: Env) -> None:
    """Increase the ``eps_emit`` boost by one unit."""

    env.eps_emit += 1.0


def clamp_Wmax(env: Env) -> None:
    """Reduce the ``Wmax`` limit by two units."""

    env.Wmax = max(env.Wmax - 2.0, 0.0)


def flip_MI_mode(env: Env) -> None:
    """Toggle the adversarial ``MI_mode`` flag."""

    env.MI_mode = not env.MI_mode


ACTION_SET: dict[str, Action] = {
    "toggle_theta_reset": toggle_theta_reset,
    "boost_eps_emit": boost_eps_emit,
    "clamp_Wmax": clamp_Wmax,
    "flip_MI_mode": flip_MI_mode,
}

# Simplified visual identifiers for timeline overlays.
ACTION_ICONS: dict[str, str] = {
    "toggle_theta_reset": "θ",
    "boost_eps_emit": "ε",
    "clamp_Wmax": "W",
    "flip_MI_mode": "MI",
}

__all__ = [
    "ACTION_SET",
    "ACTION_ICONS",
    "toggle_theta_reset",
    "boost_eps_emit",
    "clamp_Wmax",
    "flip_MI_mode",
]
