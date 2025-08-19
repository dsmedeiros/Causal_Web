"""Fitness utilities with invariant guardrails."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence
import math


DEFAULT_BASELINES = {
    "residual": 1.0,
    "ns_delta": 1.0,
    "target_success": 1.0,
    "coherence": 1.0,
}


def scalar_fitness(
    metrics: Mapping[str, float | bool],
    invariants: Mapping[str, float | bool],
    weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
    baselines: Mapping[str, float] | None = None,
    residual_threshold: float = 1e-9,
    ns_delta_threshold: float = 1e-9,
) -> float:
    """Return a scalar fitness score with hard guardrails.

    Parameters
    ----------
    metrics:
        Metric mapping produced by the gate harness.
    invariants:
        Invariant values extracted from ``metrics``.
    weights:
        Four weights ``(w1, w2, w3, w4)`` applied to the normalised terms.
    baselines:
        Normalisation baselines for ``residual``, ``ns_delta``,
        ``target_success`` and ``coherence``. Defaults to ``1.0`` for each
        entry.
    residual_threshold:
        Reject runs with ``inv_conservation_residual`` exceeding this value.
    ns_delta_threshold:
        Reject runs with ``inv_no_signaling_delta`` exceeding this value.

    Returns
    -------
    float
        Negative cost so that higher values indicate better fitness.

    Raises
    ------
    ValueError
        If any hard constraint is violated.
    """

    baselines = {**DEFAULT_BASELINES, **(baselines or {})}
    inv_residual = float(invariants.get("inv_conservation_residual", 0.0))
    if abs(inv_residual) > residual_threshold:
        raise ValueError("conservation residual above threshold")
    if not bool(invariants.get("inv_causality_ok", True)):
        raise ValueError("causality invariant failed")
    if not bool(invariants.get("inv_ancestry_ok", True)):
        raise ValueError("ancestry invariant failed")
    inv_ns = float(invariants.get("inv_no_signaling_delta", 0.0))
    if abs(inv_ns) > ns_delta_threshold:
        raise ValueError("no-signaling delta above threshold")
    chsh = metrics.get("G6_CHSH")
    if chsh is None or (isinstance(chsh, float) and math.isnan(chsh)):
        raise ValueError("invalid Bell run")

    residual_norm = min(abs(inv_residual) / baselines["residual"], 1.0)
    ns_delta_norm = min(abs(inv_ns) / baselines["ns_delta"], 1.0)
    target_success = float(metrics.get("target_success", 0.0))
    target_success_norm = min(target_success / baselines["target_success"], 1.0)
    coherence = float(metrics.get("coherence", 0.0))
    coherence_norm = min(coherence / baselines["coherence"], 1.0)

    w1, w2, w3, w4 = weights
    cost = (
        w1 * residual_norm
        + w2 * ns_delta_norm
        + w3 * (1.0 - target_success_norm)
        + w4 * (1.0 - coherence_norm)
    )
    return -float(cost)


def vector_fitness(
    metrics: Mapping[str, float | bool],
    invariants: Mapping[str, float | bool],
    baselines: Mapping[str, float] | None = None,
    residual_threshold: float = 1e-9,
    ns_delta_threshold: float = 1e-9,
) -> Sequence[float]:
    """Return a multi-objective fitness vector with guardrails.

    The returned objectives are ``[residual_norm, ns_delta_norm,
    1 - target_success_norm]`` all normalised to ``[0, 1]``.  The same hard
    constraints as :func:`scalar_fitness` are applied prior to computing the
    objectives.

    Parameters
    ----------
    metrics:
        Metric mapping produced by the gate harness.
    invariants:
        Invariant values extracted from ``metrics``.
    baselines:
        Normalisation baselines for ``residual``, ``ns_delta`` and
        ``target_success``. Defaults to ``1.0`` for each entry.
    residual_threshold:
        Reject runs with ``inv_conservation_residual`` exceeding this value.
    ns_delta_threshold:
        Reject runs with ``inv_no_signaling_delta`` exceeding this value.

    Returns
    -------
    Sequence[float]
        Objective values suitable for multi-objective optimisation.

    Raises
    ------
    ValueError
        If any hard constraint is violated.
    """

    baselines = {**DEFAULT_BASELINES, **(baselines or {})}
    inv_residual = float(invariants.get("inv_conservation_residual", 0.0))
    if abs(inv_residual) > residual_threshold:
        raise ValueError("conservation residual above threshold")
    if not bool(invariants.get("inv_causality_ok", True)):
        raise ValueError("causality invariant failed")
    if not bool(invariants.get("inv_ancestry_ok", True)):
        raise ValueError("ancestry invariant failed")
    inv_ns = float(invariants.get("inv_no_signaling_delta", 0.0))
    if abs(inv_ns) > ns_delta_threshold:
        raise ValueError("no-signaling delta above threshold")
    chsh = metrics.get("G6_CHSH")
    if chsh is None or (isinstance(chsh, float) and math.isnan(chsh)):
        raise ValueError("invalid Bell run")

    residual_norm = min(abs(inv_residual) / baselines["residual"], 1.0)
    ns_delta_norm = min(abs(inv_ns) / baselines["ns_delta"], 1.0)
    target_success = float(metrics.get("target_success", 0.0))
    target_success_norm = min(target_success / baselines["target_success"], 1.0)

    return (
        float(residual_norm),
        float(ns_delta_norm),
        float(1.0 - target_success_norm),
    )
