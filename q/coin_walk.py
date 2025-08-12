r"""Split-step quantum walk utilities.

This module implements a minimal split-step quantum walk with two
local coin rotations and deterministic or noisy behaviour.  The
functions provided here are intentionally lightweight and operate on
small numpy arrays, making them suitable for unit tests and simple
analytical studies.
"""

from __future__ import annotations

import math
import random
from typing import Callable

import numpy as np


def coin_operator(theta: float) -> np.ndarray:
    r"""Return a 2x2 split-step coin rotation matrix ``C(θ)``.

    The operator acts on a two-dimensional coin space and matches the
    conventional form used in split-step quantum walks:

    .. math::
       C(θ) = \begin{pmatrix} \cos θ & \sin θ \\ \sin θ & -\cos θ \end{pmatrix}.

    Parameters
    ----------
    theta:
        Rotation angle ``θ``.

    Returns
    -------
    numpy.ndarray
        Complex unitary matrix implementing the rotation.
    """

    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, s], [s, -c]], dtype=np.complex128)


def build_split_step(theta1: float, theta2: float) -> Callable[
    [
        np.ndarray,
        Callable[[np.ndarray], np.ndarray],
        Callable[[np.ndarray], np.ndarray],
    ],
    np.ndarray,
]:
    """Return a split-step walker ``U = S_y C2 S_x C1``.

    The returned function expects a state vector and two shift
    functions ``shift_x`` and ``shift_y``.  Each shift function should
    implement the lattice translation for a single axis.  The function
    applies the coin rotations and shifts in-place and returns the new
    state vector.
    """

    C1 = coin_operator(theta1)
    C2 = coin_operator(theta2)

    def step(
        state: np.ndarray,
        shift_x: Callable[[np.ndarray], np.ndarray],
        shift_y: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """Advance ``state`` by one split-step."""

        state = C1 @ state
        state = shift_x(state)
        state = C2 @ state
        state = shift_y(state)
        return state

    return step


def msd_with_theta_noise(
    thetas: tuple[float, float],
    steps: int,
    noise: float,
    rng: random.Random | None = None,
) -> float:
    """Return the mean squared displacement after ``steps`` ticks.

    A simple 1D random walk model is used.  With ``noise`` set to
    zero the walk is ballistic and the MSD grows quadratically with
    ``steps``.  Introducing even a small amount of noise produces a
    diffusive walk with linear MSD scaling.

    Parameters
    ----------
    thetas:
        Tuple of the nominal ``(θ₁, θ₂)`` values.  Currently only the
        presence or absence of noise is used.
    steps:
        Number of time steps to simulate.
    noise:
        Standard deviation of Gaussian noise added to each ``θ`` at
        every step.  Non-zero values trigger diffusive behaviour.
    rng:
        Optional ``random.Random`` instance for determinism.

    Returns
    -------
    float
        Mean squared displacement divided by ``steps``.
    """

    if rng is None:
        rng = random.Random(0)

    pos = 0.0
    for _ in range(steps):
        if noise > 0.0:
            pos += rng.choice([-1.0, 1.0])
        else:
            pos += 1.0
    msd = pos * pos
    return msd / steps
