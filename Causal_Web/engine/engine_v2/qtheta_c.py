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
    layer: str,
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
        Packet carrying ``depth_arr``, ``psi`", ``p`` and ``bit`` fields.
    edge:
        Edge parameters ``alpha``, ``phi`", ``A`` and unitary ``U``.
    layer:
        Layer of the destination vertex (``"Q"``, ``"Θ"`` or ``"C"``).
    max_deque:
        Maximum length of ``bit_deque``.

    Returns
    -------
    tuple
        Updated ``depth_v``, ``psi_acc``, ``p_v``, ``(bit, conf)`` and the
        intensity contribution from the supplied ``layer`` in ``[0, 1]``.
    """

    depth_v = max(depth_v, int(packet.get("depth_arr", 0)))

    U = np.asarray(edge.get("U"), dtype=np.complex64)
    psi = np.asarray(packet.get("psi"), dtype=np.complex64)
    alpha = np.float32(edge.get("alpha", 1.0))
    phi = np.float32(edge.get("phi", 0.0))
    A = np.float32(edge.get("A", 0.0))
    psi_out = U @ psi
    psi_rot = np.exp(1j * (phi + A)) * psi_out
    psi_acc = psi_acc + alpha * psi_rot

    p_v = p_v + alpha * np.asarray(packet.get("p"), dtype=np.float32)
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
    c_intensity = bit
    if layer == "Q":
        intensity = q_intensity
    elif layer == "Θ":
        intensity = theta_intensity
    elif layer == "C":
        intensity = c_intensity
    else:
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


def deliver_packets_batch(
    depth_v: int,
    psi_acc: np.ndarray,
    p_v: np.ndarray,
    bit_deque: Deque[int],
    packets: dict,
    edges: dict,
    layer: str,
    max_deque: int = 8,
) -> Tuple[int, np.ndarray, np.ndarray, Tuple[int, float], float]:
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
    packets:
        Struct-of-arrays packet fields ``{psi, p, bit, depth_arr}``.
    edges:
        Struct-of-arrays edge parameters ``{alpha, phi, A, U}``.
    layer:
        Layer of the destination vertex (``"Q"``, ``"Θ"`` or ``"C"``).
    max_deque:
        Maximum length of ``bit_deque``.

    Returns
    -------
    tuple
        Updated ``depth_v``, ``psi_acc``, ``p_v``, ``(bit, conf)`` and
        intensity contribution from the supplied ``layer`` in ``[0, 1]``.
    """

    if packets.get("depth_arr") is not None:
        depth_v = max(depth_v, int(np.max(packets.get("depth_arr"))))

    psi = np.asarray(packets.get("psi"), dtype=np.complex64)
    p = np.asarray(packets.get("p"), dtype=np.float32)
    bits = np.asarray(packets.get("bit", []), dtype=np.int8)

    U = np.asarray(edges.get("U"), dtype=np.complex64)
    alpha = np.asarray(edges.get("alpha", 1.0), dtype=np.float32)
    phi = np.asarray(edges.get("phi", 0.0), dtype=np.float32)
    A = np.asarray(edges.get("A", 0.0), dtype=np.float32)

    out = np.einsum("nij,nj->ni", U, psi)
    phase = np.exp(1j * (phi + A))[:, None]
    psi_rot = phase * out
    psi_acc = psi_acc + (alpha[:, None] * psi_rot).sum(axis=0)
    p_v = p_v + (alpha[:, None] * p).sum(axis=0)
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

    q_intensity = min(1.0, float(np.linalg.norm(psi_rot) ** 2))
    theta_intensity = min(1.0, float(np.sum(np.abs(p))))
    c_intensity = bit
    if layer == "Q":
        intensity = q_intensity
    elif layer == "Θ":
        intensity = theta_intensity
    elif layer == "C":
        intensity = c_intensity
    else:
        intensity = min(1.0, q_intensity + theta_intensity + c_intensity)

    return depth_v, psi_acc, p_v, (bit, conf), intensity


__all__ = ["deliver_packet", "deliver_packets_batch", "close_window"]
