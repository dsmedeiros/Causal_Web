"""Q/Θ/C field update helpers.

This module provides minimal routines for the experimental v2 engine to
update quantum (``psi``), probabilistic (``p``) and classical (``bit``)
fields when a packet is delivered across an edge.  The functions operate on
simple ``dict`` structures matching the loader's output and return the
combined intensity used by the density-delay model.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Iterable, Tuple

import numpy as np


def deliver_packet(
    depth_v: int,
    psi_acc: np.ndarray,
    p_v: np.ndarray,
    bit_deque: Deque[int],
    packet: dict,
    edge: dict,
    max_deque: int = 8,
) -> Tuple[int, np.ndarray, np.ndarray, Tuple[int, float], float]:
    """Apply Q/Θ/C delivery rules for a single packet.

    Parameters
    ----------
    depth_v:
        Current depth of the destination vertex.
    psi_acc:
        Accumulator for quantum amplitudes.
    p_v:
        Classical probability vector for the vertex.
    bit_deque:
        Recent bits used for majority voting.
    packet:
        Packet carrying ``depth_arr``, ``psi``, ``p`` and ``bit`` fields.
    edge:
        Edge parameters ``alpha``, ``phi``, ``A`` and unitary ``U``.
    max_deque:
        Maximum length of ``bit_deque``.

    Returns
    -------
    tuple
        Updated ``depth_v``, ``psi_acc``, ``p_v``, ``(bit, conf)`` and the
        combined intensity in ``[0, 1]``.
    """

    depth_v = max(depth_v, int(packet.get("depth_arr", 0)))

    U = np.asarray(edge.get("U"), dtype=np.complex128)
    psi = np.asarray(packet.get("psi"), dtype=np.complex128)
    coeff = edge.get("alpha", 1.0) * np.exp(
        1j * (edge.get("phi", 0.0) + edge.get("A", 0.0))
    )
    psi_acc = psi_acc + coeff * (U @ psi)

    p_v = p_v + edge.get("alpha", 1.0) * np.asarray(packet.get("p"))
    total = float(np.sum(p_v))
    if total > 0:
        p_v = p_v / total

    bit_deque.append(int(packet.get("bit", 0)))
    while len(bit_deque) > max_deque:
        bit_deque.popleft()
    ones = sum(bit_deque)
    zeros = len(bit_deque) - ones
    bit = 1 if ones >= zeros else 0
    conf = abs(ones - zeros) / len(bit_deque)

    q_intensity = min(1.0, float(np.linalg.norm(U @ psi) ** 2))
    theta_intensity = min(1.0, float(np.sum(np.abs(packet.get("p", [])))))
    c_intensity = bit
    intensity = min(1.0, q_intensity + theta_intensity + c_intensity)

    return depth_v, psi_acc, p_v, (bit, conf), intensity


def close_window(psi_acc: np.ndarray) -> Tuple[np.ndarray, float]:
    """Normalise the accumulated ``psi`` and compute ``EQ``."""

    EQ = float(np.vdot(psi_acc, psi_acc).real)
    if EQ > 0:
        psi = psi_acc / np.sqrt(EQ)
    else:
        psi = psi_acc.copy()
    return psi, EQ


__all__ = ["deliver_packet", "close_window"]
