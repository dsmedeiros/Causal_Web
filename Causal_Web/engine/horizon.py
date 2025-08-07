"""Toy horizon thermodynamics model for Hawking radiation.

This module contains a lightweight model that probabilistically emits entangled
Hawking pairs from nodes designated as *interior* to a horizon.  The emitted
quanta are tracked to build a crude Page curve for the exterior radiation's
entropy.  The entropy rises until roughly half of the interior quanta have been
emitted and then falls back to zero as the horizon evaporates.

The public API exposes a singleton horizon model and a few helpers for
integration with the scheduler:

``get_horizon()``
    Access the global :class:`HorizonThermodynamics` instance.
``register_interior(node, energy)``
    Mark a node as inside the horizon with an energy budget.
``step(nodes)``
    Perform Hawking emission for a collection of nodes for one tick.
"""

from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List

from ..config import Config
from .models.node import Node


class HorizonThermodynamics:
    """Simulate Hawking-pair emission and track exterior entropy.

    Parameters
    ----------
    temperature:
        Hawking temperature :math:`T_H` controlling the emission probability.
    delta_e:
        Energy carried by a single quantum; defaults to ``1.0``.
    """

    def __init__(self, temperature: float, delta_e: float = 1.0) -> None:
        self.temperature = temperature
        self.delta_e = delta_e
        self.interior_energy: Dict[str, float] = {}
        self.emitted_quanta = 0
        self.total_quanta = 0.0
        self.outside_entropy: List[float] = []

    def register(self, node: Node, energy: float) -> None:
        """Mark ``node`` as inside the horizon with available ``energy``.

        Parameters
        ----------
        node:
            Interior node eligible to emit Hawking quanta.
        energy:
            Energy budget that can be converted into emitted quanta.
        """

        self.interior_energy[node.id] = energy
        self.total_quanta += energy / self.delta_e

    def step(self, nodes: Iterable[Node]) -> None:
        """Attempt Hawking emission for ``nodes`` and update entropy.

        Parameters
        ----------
        nodes:
            Nodes to process this tick. Only nodes previously registered as
            interior contribute energy.
        """

        prob = math.exp(-self.delta_e / self.temperature)
        for node in nodes:
            energy = self.interior_energy.get(node.id, 0.0)
            if energy <= 0.0:
                continue
            if random.random() < prob:
                energy -= self.delta_e
                self.emitted_quanta += 1
            self.interior_energy[node.id] = max(0.0, energy)
        self._update_entropy()

    def _update_entropy(self) -> None:
        out = self.emitted_quanta
        ent = min(out, self.total_quanta - out)
        self.outside_entropy.append(ent)

    @property
    def entropy(self) -> float:
        """Return the most recent outside entropy value."""

        if not self.outside_entropy:
            return 0.0
        return self.outside_entropy[-1]


_horizon: HorizonThermodynamics | None = None


def get_horizon() -> HorizonThermodynamics:
    """Return the global :class:`HorizonThermodynamics` instance.

    The instance is created lazily using ``Config.hawking_temperature`` for its
    temperature.
    """

    global _horizon
    if _horizon is None:
        temp = getattr(Config, "hawking_temperature", 1.0)
        _horizon = HorizonThermodynamics(temp)
    return _horizon


def register_interior(node: Node, energy: float) -> None:
    """Register ``node`` as interior using the global horizon model.

    Parameters
    ----------
    node:
        Node to register as interior.
    energy:
        Energy available for Hawking emission.
    """

    get_horizon().register(node, energy)


def step(nodes: Iterable[Node]) -> None:
    """Advance the global horizon model for ``nodes``.

    Parameters
    ----------
    nodes:
        Nodes to process for potential Hawking emission during this tick.
    """

    get_horizon().step(nodes)
