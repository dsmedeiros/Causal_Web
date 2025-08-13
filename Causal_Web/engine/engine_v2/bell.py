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


def _splitmix64(x: np.uint64) -> np.uint64:
    """Return SplitMix64 hash of ``x``.

    Parameters
    ----------
    x:
        64-bit input value.

    Returns
    -------
    np.uint64
        Hashed output value.
    """

    z = x + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def _splitmix_vec3(seed: np.ndarray) -> np.ndarray:
    """Return three ``uint32`` values derived from ``seed``.

    The input ``seed`` is reduced to a single ``uint64`` and expanded via
    sequential SplitMix64 applications.
    """

    s = np.uint64(int(np.bitwise_xor.reduce(seed)))
    return np.array(
        [_splitmix64(s + np.uint64(i)) & np.uint64(0xFFFFFFFF) for i in range(3)],
        dtype=np.uint32,
    )


@dataclass
class Ancestry:
    """Track ancestry information for a packet.

    Parameters
    ----------
    h:
        Rolling hash stored as four ``uint64`` segments.
    m:
        Three dimensional phase-moment vector.
    """

    h: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.uint64))
    m: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))


class BellHelpers:
    """Utility methods for Bell experiment simulations."""

    def __init__(
        self, cfg: Dict[str, float] | None = None, seed: int | None = None
    ) -> None:
        """Initialise the helper with an optional RNG ``seed`` and config."""

        self._cfg = cfg or {}
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

    def _rotate(self, u: np.ndarray, h: np.ndarray, zeta: float) -> np.ndarray:
        """Rotate ``u`` about a deterministic axis by ``2πζα_R``.

        The axis is derived locally from ``h`` using :func:`_splitmix_vec3`
        ensuring strict locality. The rotation angle is scaled by the
        ``alpha_R`` configuration parameter and supports both continuous and
        discrete ``zeta`` values.
        """

        ax_u32 = _splitmix_vec3(h)
        axis = ax_u32.astype(np.int64) - 2**31
        axis = self._unit_vector(axis.astype(float))

        alpha_R = self._cfg.get("alpha_R", 1.0)
        if self._cfg.get("zeta_mode", "float") == "int_mod_k":
            k = self._cfg.get("k_mod", 3)
            theta = 2 * np.pi * (zeta / max(k, 1)) * alpha_R
        else:
            theta = 2 * np.pi * float(zeta) * alpha_R

        cos_a = np.cos(theta)
        sin_a = np.sin(theta)
        return (
            u * cos_a + np.cross(axis, u) * sin_a + axis * np.dot(axis, u) * (1 - cos_a)
        )

    def _ancestry_overlap(self, a: Ancestry, b: Ancestry) -> float:
        """Return fraction of matching ``h`` segments between ancestries."""

        return float(np.sum(a.h == b.h)) / 4.0

    # ------------------------------------------------------------------
    # Public API
    def lambda_at_source(
        self, ancestry: Ancestry, beta_m: float, beta_h: float
    ) -> Tuple[np.ndarray, float]:
        """Create the shared hidden variable ``lambda`` for a new pair.

        Parameters
        ----------
        ancestry:
            Source ancestry values ``(h_S, m_S)``.
        beta_m:
            Blend factor for phase moments.
        beta_h:
            Blend factor for hash contribution to ``zeta``.

        Returns
        -------
        tuple
            ``(u, zeta)`` where ``u`` is a unit direction vector and ``zeta``
            is derived from the ancestry hash. When
            When ``cfg['zeta_mode']`` is ``"float"`` (default) ``zeta``
            lies in ``[0, 1)``. If ``"int_mod_k"`` is selected ``zeta`` is an
            integer in ``[0, k_mod)``.
        """

        u_rand = self._random_unit()
        u = beta_m * ancestry.m + (1.0 - beta_m) * u_rand
        u = self._unit_vector(u)

        zeta_u64 = _splitmix64(ancestry.h[0])
        zeta_hash = np.float64(zeta_u64) / np.float64(2**64)
        zeta_rand = self._rng.random()
        zeta_float = beta_h * zeta_hash + (1.0 - beta_h) * zeta_rand

        if self._cfg.get("zeta_mode", "float") == "int_mod_k":
            k = self._cfg.get("k_mod", 3)
            zeta_val: float | int = int(zeta_u64 % max(k, 1))
        else:
            zeta_val = float(zeta_float % 1.0)

        return u, float(zeta_val)

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
        zeta: float,
        kappa_xi: float,
        source_ancestry: Ancestry,
        kappa_a: float,
        batch: int,
    ) -> Tuple[int, Dict[str, float]]:
        """Compute the detector readout and associated log values."""

        rotated = self._rotate(lam_u, detector_ancestry.h, zeta)
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
