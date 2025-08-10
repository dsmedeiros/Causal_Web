"""Bell experiment helpers for the v2 engine.

This module implements a minimal Bell experiment simulator with
ancestry-aware hidden variables and measurement setting draws that can
be either strictly independent or weakly conditioned on the shared
``lambda``.  The implementation is intentionally lightweight but
captures the interfaces required by the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np


@dataclass
class Ancestry:
    """Track ancestry information for a packet.

    Parameters
    ----------
    h:
        Rolling 256-bit hash stored as four ``uint64`` values.
    m:
        Three dimensional phase-moment vector.
    """

    h: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.uint64))
    m: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))


class BellHelpers:
    """Utility methods for Bell experiment simulations."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Helpers
    def _unit_vector(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm == 0:
            vec = self._rng.normal(size=3)
            norm = np.linalg.norm(vec)
        return vec / norm

    def _random_unit(self) -> np.ndarray:
        return self._unit_vector(self._rng.normal(size=3))

    def _sample_vmf(self, mu: np.ndarray, kappa: float) -> np.ndarray:
        """Sample from an approximate 3D von Mises–Fisher distribution.

        This uses a simple normal perturbation which is sufficient for the
        simulation goals of this project and avoids heavy dependencies.
        """

        if kappa <= 0:
            return self._random_unit()
        sample = self._rng.normal(size=3) + kappa * mu
        return self._unit_vector(sample)

    def _rotate(self, u: np.ndarray, zeta: int) -> np.ndarray:
        """Rotate ``u`` using a simple component roll keyed by ``zeta``."""

        return np.roll(u, int(zeta % 3))

    def _ancestry_overlap(self, a: Ancestry, b: Ancestry) -> float:
        """Return fraction of matching ``h`` segments between ancestries."""

        return float(np.sum(a.h == b.h)) / 4.0

    # ------------------------------------------------------------------
    # Public API
    def lambda_at_source(
        self, ancestry: Ancestry, beta_m: float, beta_h: float
    ) -> Tuple[np.ndarray, int]:
        """Create the shared hidden variable ``lambda`` for a new pair.

        Parameters
        ----------
        ancestry:
            Source ancestry values ``(h_S, m_S)``.
        beta_m:
            Blend factor for phase moments.
        beta_h:
            Blend factor for hash contribution to ``zeta``.
        """

        u_rand = self._random_unit()
        u = beta_m * ancestry.m + (1.0 - beta_m) * u_rand
        u = self._unit_vector(u)

        h_int = int.from_bytes(ancestry.h.tobytes(), "little")
        zeta_rand = int(self._rng.integers(0, 2**32))
        zeta = int(beta_h * h_int) ^ zeta_rand
        return u, zeta

    def setting_draw(
        self,
        mode: str,
        ancestry: Ancestry,
        lam_u: np.ndarray,
        kappa_a: float,
    ) -> np.ndarray:
        """Draw a measurement setting vector ``a_D``.

        Parameters
        ----------
        mode:
            Either ``"strict"`` for measurement independence or
            ``"conditioned"`` to bias settings toward the shared ancestry.
        ancestry:
            Detector ancestry values ``(h_D, m_D)``.
        lam_u:
            Shared hidden direction from :func:`lambda_at_source`.
        kappa_a:
            Concentration parameter for ``MI_conditioned`` draws.
        """

        if mode == "strict":
            return self._random_unit()

        mu = self._unit_vector(ancestry.m + lam_u)
        return self._sample_vmf(mu, kappa_a)

    def contextual_readout(
        self,
        mode: str,
        a_D: np.ndarray,
        detector_ancestry: Ancestry,
        lam_u: np.ndarray,
        zeta: int,
        kappa_xi: float,
        source_ancestry: Ancestry,
        kappa_a: float,
        batch: int,
    ) -> Tuple[int, Dict[str, float]]:
        """Compute the detector readout and associated log values."""

        rotated = self._rotate(lam_u, zeta)
        noise = self._rng.normal(scale=1.0 / max(kappa_xi, 1e-9))
        outcome = 1 if float(np.dot(a_D, rotated) + noise) > 0 else -1

        log = {
            "mode": mode,
            "kappa_a": kappa_a,
            "kappa_xi": kappa_xi,
            "L": self._ancestry_overlap(detector_ancestry, source_ancestry),
            "batch": batch,
        }
        return outcome, log
