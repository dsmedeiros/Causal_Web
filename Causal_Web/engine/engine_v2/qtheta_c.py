"""Q/Θ/C field update helpers.

This module provides minimal routines for the experimental v2 engine to
update quantum (``psi``), probabilistic (``p``) and classical (``bit``)
fields when a packet is delivered across an edge. The functions operate on
simple ``dict`` structures matching the loader's output and return per-layer
intensity contributions used by the density-delay model.
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
    update_p: bool = True,
) -> Tuple[
    int,
    np.ndarray,
    np.ndarray,
    Tuple[int, float],
    Tuple[float, float, float],
    float,
    float,
]:
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
        Edge parameters ``alpha`` and unitary ``U`` with a precomputed
        complex ``phase``. ``phase`` may be provided directly or derived
        from ``phi`` and ``A`` if absent.
    max_deque:
        Maximum length of ``bit_deque``.
    update_p:
        When ``True`` accumulate the probabilistic field ``p_v``.

    Returns
    -------
    tuple
        Updated ``depth_v``, ``psi_acc``, ``p_v``, ``(bit, conf)``, the per-layer
        intensity contributions ``(I_Q, I_Θ, I_C)`` each in ``[0, 1]`` and the
        local phase statistics ``mu`` and ``kappa``.
    """

    depth_v = max(depth_v, int(packet.get("depth_arr", 0)))

    U = np.asarray(edge.get("U"), dtype=np.complex64)
    psi = np.asarray(packet.get("psi"), dtype=np.complex64)
    alpha = np.float32(edge.get("alpha", 1.0))
    phase = edge.get("phase")
    if phase is None:
        phi = np.float32(edge.get("phi", 0.0))
        A = np.float32(edge.get("A", 0.0))
        phase = np.exp(1j * (phi + A))
    phase = np.complex64(phase)
    psi_out = U @ psi
    psi_rot = phase * psi_out
    psi_acc = psi_acc + alpha * psi_rot
    psi_acc = np.where(np.isfinite(psi_acc), psi_acc, np.zeros_like(psi_acc))
    z = np.vdot(psi_rot, psi)
    mu = float(np.angle(z))
    kappa = float(abs(z))

    if update_p:
        p_v = p_v + alpha * np.asarray(packet.get("p"), dtype=np.float32)
        p_v = np.clip(p_v, 0.0, 1.0)
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

    q_intensity = min(1.0, float(np.linalg.norm(psi_rot) ** 2))
    theta_intensity = min(1.0, float(np.sum(np.abs(packet.get("p", [])))))
    c_intensity = float(bit)

    intensities = (q_intensity, theta_intensity, c_intensity)

    return depth_v, psi_acc, p_v, (bit, conf), intensities, mu, kappa


def close_window(psi_acc: np.ndarray) -> Tuple[np.ndarray, float]:
    """Normalise the accumulated ``psi`` and compute ``EQ``."""

    EQ = float(np.vdot(psi_acc, psi_acc).real)
    if EQ > 0:
        psi = psi_acc / np.sqrt(EQ)
    else:
        psi = psi_acc.copy()
    psi = np.where(np.isfinite(psi), psi, np.zeros_like(psi))
    return psi, EQ


def deliver_packets_batch(
    depth_v: int,
    psi_acc: np.ndarray,
    p_v: np.ndarray,
    bit_deque: Deque[int],
    psi: Iterable[np.ndarray],
    p: Iterable[np.ndarray],
    bits: Iterable[int],
    depth_arr: Iterable[int] | None,
    alpha: Iterable[float],
    phase: Iterable[complex],
    U: Iterable[np.ndarray],
    max_deque: int = 8,
    update_p: bool = True,
) -> Tuple[int, np.ndarray, np.ndarray, Tuple[int, float]]:
    """Vectorised delivery for packets sharing destination and window.

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
    psi, p, bits, depth_arr:
        Packet fields supplied as sequences or arrays.
    alpha, phase, U:
        Edge parameters supplied as sequences or arrays.
    max_deque:
        Maximum length of ``bit_deque``.
    update_p:
        When ``True`` accumulate the probabilistic field ``p_v``.

    Returns
    -------
    tuple
        Updated ``depth_v``, ``psi_acc``, ``p_v``, ``(bit, conf)``.
    """

    if depth_arr is not None:
        depth_v = max(depth_v, int(np.max(depth_arr)))

    psi = np.asarray(list(psi), dtype=np.complex64)
    p = np.asarray(list(p), dtype=np.float32)
    bits = np.asarray(list(bits), dtype=np.int8)

    U = np.asarray(list(U), dtype=np.complex64)
    alpha = np.asarray(list(alpha), dtype=np.float32)
    phase = np.asarray(list(phase), dtype=np.complex64)[:, None]

    out = np.einsum("nij,nj->ni", U, psi)
    psi_rot = phase * out
    psi_acc = psi_acc + (alpha[:, None] * psi_rot).sum(axis=0)
    psi_acc = np.where(np.isfinite(psi_acc), psi_acc, np.zeros_like(psi_acc))
    if update_p:
        p_v = p_v + (alpha[:, None] * p).sum(axis=0)
        p_v = np.clip(p_v, 0.0, 1.0)
        total = float(np.sum(p_v))
        if total > 0:
            p_v = p_v / total

    bit_deque.extend(bits.tolist())
    while len(bit_deque) > max_deque:
        bit_deque.popleft()
    ones = sum(bit_deque)
    zeros = len(bit_deque) - ones
    bit = 1 if ones >= zeros else 0
    conf = abs(ones - zeros) / len(bit_deque)

    return depth_v, psi_acc, p_v, (bit, conf)


__all__ = ["deliver_packet", "deliver_packets_batch", "close_window"]
