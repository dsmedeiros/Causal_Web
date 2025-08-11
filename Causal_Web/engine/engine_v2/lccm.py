"""Local causal consistency helpers for engine v2.

This module implements the windowing mathematics and layer transition rules
for the Local Causal Consistency Model (LCCM).  The implementation is a
lightweight container used by the experimental engine to reason about packet
delivery counts within rolling windows and to transition between quantum (Q),
decohered (Θ) and classical (C) layers with simple hysteresis timers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class LCCM:
    """Track window indices and layer transitions for a vertex.

    Parameters defining the window function and transition thresholds are
    provided at construction time.  ``deg`` is the vertex's **incident degree**—
    the sum of its in-degree and out-degree.  ``W(v)`` uses this combined
    degree to size the rolling window so that both fan-in and fan-out pressure
    influence window growth.  ``rho_mean`` is the mean local density used in
    ``W(v)`` and ``conf_min`` is the minimum bit-majority confidence required
    for Θ→C transitions.  The coefficients ``k_theta`` and ``k_c`` weight the
    ``E_Θ`` and ``E_C`` meters reported at window closure.  The public
    attributes ``depth``, ``window_idx`` and ``layer`` expose the current state
    for callers.
    """

    W0: int
    zeta1: float
    zeta2: float
    rho0: float
    a: float
    b: float
    C_min: float
    f_min: float
    conf_min: float
    H_max: float
    T_hold: int
    T_class: int
    k_theta: float = 1.0
    k_c: float = 0.5
    deg: int = 0  # incident degree (in + out) used in W(v)
    rho_mean: float = 0.0
    depth: int = 0
    window_idx: int = 0
    layer: str = "Q"

    _lambda: int = 0
    _eq: float = 0.0
    _eq_hold: int = 0
    _bit_fraction: float = 0.0
    _entropy: float = 0.0
    _confidence: float = 0.0
    _class_timer: int = 0

    # ------------------------------------------------------------------
    def _window_size(self) -> int:
        term = self.zeta1 * math.log(1 + self.deg) + self.zeta2 * math.log(
            1 + self.rho_mean / self.rho0
        )
        return self.W0 + math.floor(term)

    # Public API -------------------------------------------------------
    def advance_depth(self, new_depth: int) -> None:
        """Update ``depth`` and recompute ``window_idx``.

        ``new_depth`` must be monotonically increasing.  When the window index
        changes the fan-in counter Λ is reset.
        """

        if new_depth < self.depth:
            raise ValueError("depth must be monotonically increasing")
        self.depth = new_depth
        w = self._window_size()
        idx = new_depth // w
        if idx != self.window_idx:
            self.window_idx = idx
            self._lambda = 0

    def deliver(self) -> None:
        """Record a packet delivery at the current depth."""

        self._lambda += 1
        self._check_transitions()

    def update_eq(self, value: float) -> None:
        """Update the EQ metric used for Θ→Q transitions."""

        self._eq = value

    def update_classical_metrics(
        self, bit_fraction: float, entropy: float, confidence: float
    ) -> None:
        """Update metrics used for Θ→C transitions.

        Parameters
        ----------
        bit_fraction:
            Fraction of ``1`` bits observed within the current window.
        entropy:
            Shannon entropy of the classical distribution ``p``.
        confidence:
            Majority-vote confidence of the recent bits.
        """

        self._bit_fraction = bit_fraction
        self._entropy = entropy
        self._confidence = confidence

    # Internal helpers -------------------------------------------------
    def _check_transitions(self) -> None:
        w = self._window_size()
        if self.layer == "Q":
            if self._lambda >= self.a * w:
                self.layer = "Θ"
                self._eq_hold = 0
                self._class_timer = 0
        elif self.layer == "Θ":
            # Recoherence toward Q
            if self._lambda <= self.b * w and self._eq >= self.C_min:
                self._eq_hold += 1
            else:
                self._eq_hold = 0
            if self._eq_hold >= self.T_hold:
                self.layer = "Q"
                self._eq_hold = 0
                self._class_timer = 0
                self._lambda = 0
                return

            # Classical dominance toward C
            if (
                self._bit_fraction >= self.f_min
                and self._confidence >= self.conf_min
                and self._entropy <= self.H_max
            ):
                self._class_timer += 1
            else:
                self._class_timer = 0
            if self._class_timer >= self.T_class:
                self.layer = "C"
                self._class_timer = 0
                self._eq_hold = 0


__all__ = ["LCCM"]
