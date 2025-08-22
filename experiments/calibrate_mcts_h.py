from __future__ import annotations

import itertools

"""Utility to sweep MCTS-H hyperparameters on a canonical graph.

The sweep evaluates MCTS-H on a minimal engine graph so calibration reflects
realistic search behaviour rather than a synthetic toy function. Proxy
evaluations run a short simulation while full evaluations step the graph
further, producing a noisier signal with which to correlate the proxy scores.
"""

from typing import Dict, Sequence

import argparse
import itertools

import numpy as np

from Causal_Web.engine.engine_v2.adapter import EngineAdapter
from Causal_Web.engine.engine_v2.state import Packet
from experiments.optim import MCTS_H, build_priors


def _simulate(w0: float, max_events: int) -> int:
    """Run the canonical single-node graph for ``max_events`` events."""

    graph = {
        "params": {"W0": w0},
        "nodes": [{"id": "0", "rho_mean": 0.0}],
        "edges": [{"from": "0", "to": "0", "delay": 1.0}],
    }
    adapter = EngineAdapter()
    adapter.build_graph(graph)
    adapter._scheduler.push(0, 0, 0, Packet(0, 0))
    frame = adapter.step(max_events=max_events, collect_packets=False)
    return frame.events


def _proxy_fn(cfg: Dict[str, float]) -> float:
    events = _simulate(cfg["W0"], 3)
    return float((events - 3) ** 2)


def _full_fn(cfg: Dict[str, float]) -> float:
    events = _simulate(cfg["W0"], 5)
    return float((events - 3) ** 2)


def _run_once(
    c_ucb: float,
    alpha_pw: float,
    k_pw: float,
    bins: int,
    proxy_frames: int = 300,
    full_frames: int = 3000,
    seed: int = 0,
) -> Dict[str, float]:
    space: Sequence[str] = ["W0"]
    topk = [{"W0": 2.0}]
    priors = build_priors(topk, bins=bins)
    opt = MCTS_H(
        space,
        priors,
        {"c_ucb": c_ucb, "alpha_pw": alpha_pw, "k_pw": k_pw, "rng_seed": seed},
    )
    proxy_count = 0
    full_count = 0
    best_full = float("inf")
    while proxy_count < proxy_frames or opt._pending_full:  # type: ignore[attr-defined]
        cfg = opt.suggest(1)[0]
        key = tuple(sorted(cfg.items()))
        if key in opt._suggest_full:  # type: ignore[attr-defined]
            if full_count >= full_frames:
                break
            score = _full_fn(cfg)
            opt.observe([{"config": cfg, "fitness": score}])
            full_count += 1
            best_full = min(best_full, score)
        else:
            if proxy_count >= proxy_frames:
                break
            score = _proxy_fn(cfg)
            opt.observe([{"config": cfg, "fitness_proxy": score}])
            proxy_count += 1
    metrics = opt.metrics()
    metrics.update({"best_full_fitness": best_full, "full_evals": full_count})
    return metrics


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy-frames", type=int, default=300)
    parser.add_argument("--full-frames", type=int, default=3000)
    args = parser.parse_args(list(argv) if argv is not None else None)

    grid = itertools.product(
        [0.7, 1.0, 1.3],
        [0.4, 0.5, 0.6],
        [1, 2, 3],
        [3, 5, 7],
    )
    rows = []
    for c_ucb, alpha_pw, k_pw, bins in grid:
        metrics = _run_once(
            c_ucb,
            alpha_pw,
            float(k_pw),
            bins,
            proxy_frames=args.proxy_frames,
            full_frames=args.full_frames,
        )
        rows.append(
            {
                "c_ucb": c_ucb,
                "alpha_pw": alpha_pw,
                "k_pw": k_pw,
                "bins": bins,
                **metrics,
            }
        )
    rows.sort(key=lambda r: r["best_full_fitness"])
    headers = [
        "c_ucb",
        "alpha",
        "k",
        "bins",
        "fitness",
        "full_evals",
        "expansion",
        "promotion",
        "depth",
        "spearman",
    ]
    print("\t".join(headers))
    for r in rows:
        print(
            f"{r['c_ucb']:.1f}\t{r['alpha_pw']:.1f}\t{r['k_pw']}\t{r['bins']}\t"
            f"{r['best_full_fitness']:.3f}\t{r['full_evals']}\t"
            f"{r['expansion_rate']:.2f}\t{r['promotion_rate']:.2f}\t"
            f"{r['avg_rollout_depth']:.2f}\t{r['spearman_proxy_full']:.2f}"
        )


if __name__ == "__main__":
    main()
