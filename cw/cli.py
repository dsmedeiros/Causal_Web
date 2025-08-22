"""Console entrypoint for the ``cw`` command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import yaml

from experiments import OptimizerQueueManager, MCTS_H, build_priors, scalar_fitness


def main(argv: Optional[List[str]] = None) -> None:
    """Parse ``cw`` CLI arguments and dispatch to the runner."""

    parser = argparse.ArgumentParser(prog="cw")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("run", help="Run simulation")
    optim_p = sub.add_parser("optim", help="Run hyperparameter optimisation")
    optim_p.add_argument("--optim", choices=["mcts_h"], default="mcts_h")
    optim_p.add_argument("--base", required=True, help="YAML baseline config")
    optim_p.add_argument(
        "--space", required=True, help="YAML list of dimensionless groups"
    )
    optim_p.add_argument("--priors", help="Top-K JSON file for priors")
    optim_p.add_argument("--budget", type=int, default=1, help="Evaluation budget")
    optim_p.add_argument("--gates", default="1", help="Comma-separated gate ids")
    optim_p.add_argument("--seed", type=int, default=0, help="Random seed")
    optim_p.add_argument(
        "--multi-objective",
        action="store_true",
        help="Enable Dirichlet scalarisation for multi-objective runs",
    )
    optim_p.add_argument(
        "--promote-threshold", type=float, help="Proxy fitness promotion threshold"
    )
    optim_p.add_argument(
        "--promote-quantile",
        type=float,
        help="Proxy fitness quantile for promotion",
    )
    optim_p.add_argument(
        "--promote-window",
        type=int,
        help="Window size for quantile promotion",
    )
    optim_p.add_argument(
        "--proxy-frames", type=int, default=300, help="Frame budget for proxy runs"
    )
    optim_p.add_argument(
        "--full-frames", type=int, default=3000, help="Frame budget for full runs"
    )
    optim_p.add_argument(
        "--bins", type=int, default=3, help="Quantile bins per parameter"
    )
    optim_p.add_argument("--state", help="Path to optimiser state checkpoint")

    args, rest = parser.parse_known_args(argv)
    if args.command == "run":
        from Causal_Web.main import MainService

        MainService(argv=rest).run()
    elif args.command == "optim":
        with Path(args.base).open() as fh:
            base = yaml.safe_load(fh)
        with Path(args.space).open() as fh:
            space = yaml.safe_load(fh)
        priors = {}
        if args.priors:
            data = json.loads(Path(args.priors).read_text())
            rows = [r["groups"] for r in data.get("rows", [])]
            priors = build_priors(rows, bins=args.bins)
        state_path = Path(args.state) if args.state else None
        cfg = {"rng_seed": args.seed}
        if args.multi_objective:
            cfg["multi_objective"] = True
        if args.promote_quantile is not None:
            cfg["promote_quantile"] = args.promote_quantile
        elif args.promote_threshold is not None:
            cfg["promote_threshold"] = args.promote_threshold
        if args.promote_window is not None:
            cfg["promote_window"] = args.promote_window
        if state_path and state_path.exists():
            opt = MCTS_H.load(state_path, priors)
        else:
            opt = MCTS_H(space, priors, cfg)
        gates = [int(g) for g in args.gates.split(",") if g]
        fitness_fn = lambda m, inv, groups: scalar_fitness(m, inv)
        mgr = OptimizerQueueManager(
            base,
            gates,
            fitness_fn,
            opt,
            seed=args.seed,
            proxy_frames=args.proxy_frames,
            full_frames=args.full_frames,
            state_path=state_path,
        )
        for _ in range(args.budget):
            if mgr.run_next() is None:
                break


if __name__ == "__main__":
    main()
