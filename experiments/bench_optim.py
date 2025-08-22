"""Optimizer bake-off benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import statistics
import time
from typing import Callable, Dict, List, Tuple

import numpy as np

from experiments.gates import run_gates


@dataclass
class Task:
    """Container describing a benchmark task.

    Parameters
    ----------
    space:
        Continuous parameter bounds for the task. All parameters are expressed
        in normalised units.
    evaluate:
        Function mapping a parameter dictionary to a scalar loss.
    gates:
        Gate identifiers to probe when evaluating the task.
    conditional:
        Whether the search space contains conditional branches. This flag is
        used when aggregating benchmark statistics.
    """

    space: Dict[str, Tuple[float, float]]
    evaluate: Callable[[Dict[str, float]], float]
    gates: List[int]
    conditional: bool = False


# ---------------------------------------------------------------------------
# Task definitions


def _bell_eval(params: Dict[str, float]) -> float:
    metrics = run_gates({"bell": {"kappa_a": params["kappa_a"]}}, [6])
    return -float(metrics.get("G6_CHSH", 0.0))


def _interference_eval(params: Dict[str, float]) -> float:
    metrics = run_gates({"prob": params["prob"]}, [1])
    return -float(metrics.get("G1_visibility", 0.0))


def _conservation_eval(params: Dict[str, float]) -> float:
    gamma = params["gamma"] if params["use_gamma"] > 0.5 else 0.1
    cfg = {
        "alpha_leak": params["alpha_leak"],
        "eta": 0.1,
        "d0": 1.0,
        "gamma": gamma,
        "rho0": 1.0,
    }
    metrics = run_gates(cfg, [2, 5])
    return float(metrics.get("inv_conservation_residual", 0.0))


TASKS: Dict[str, Task] = {
    "bell": Task({"kappa_a": (0.0, 5.0)}, _bell_eval, [6]),
    "interference": Task({"prob": (0.0, 1.0)}, _interference_eval, [1]),
    "conservation": Task(
        {
            "alpha_leak": (0.0, 1.0),
            "use_gamma": (0.0, 1.0),
            "gamma": (0.0, 1.0),
        },
        _conservation_eval,
        [2, 5],
        conditional=True,
    ),
}


# ---------------------------------------------------------------------------
# Optimizer runners


def _run_ga(task: Task, budget: int) -> Tuple[List[float], float]:
    """Run a minimal genetic algorithm."""

    rng = np.random.default_rng(0)
    pop_size = min(10, budget)

    def random_cfg() -> Dict[str, float]:
        return {
            k: float(rng.uniform(low, high)) for k, (low, high) in task.space.items()
        }

    start = time.time()
    population = [random_cfg() for _ in range(pop_size)]
    fits = [task.evaluate(cfg) for cfg in population]
    all_fits = fits.copy()
    evaluations = len(all_fits)

    while evaluations < budget:
        parents = rng.choice(population, size=2, replace=False)
        child: Dict[str, float] = {}
        for k, (low, high) in task.space.items():
            child[k] = parents[0][k] if rng.random() < 0.5 else parents[1][k]
            child[k] += rng.normal(0.0, 0.1 * (high - low))
            child[k] = float(np.clip(child[k], low, high))
        fit = task.evaluate(child)
        population.append(child)
        fits.append(fit)
        all_fits.append(fit)
        evaluations += 1
        order = np.argsort(fits)
        population = [population[i] for i in order[:pop_size]]
        fits = [fits[i] for i in order[:pop_size]]

    duration = time.time() - start
    return all_fits[:budget], duration


def _run_tpe(task: Task, budget: int) -> Tuple[List[float], float]:
    """Run Hyperopt's TPE algorithm."""

    try:
        from hyperopt import Trials, fmin, hp, tpe
    except Exception:  # pragma: no cover - optional dependency
        return _run_ga(task, budget)

    space = {k: hp.uniform(k, low, high) for k, (low, high) in task.space.items()}
    trials = Trials()

    def objective(params: Dict[str, float]) -> float:
        return task.evaluate(params)

    start = time.time()
    fmin(
        objective,
        space=space,
        algo=tpe.suggest,
        max_evals=budget,
        trials=trials,
        rstate=np.random.default_rng(0),
        show_progressbar=False,
    )
    duration = time.time() - start
    fits = [float(t["result"]["loss"]) for t in trials.trials]
    return fits[:budget], duration


def _run_cma(task: Task, budget: int) -> Tuple[List[float], float]:
    """Run CMA-ES using the ``cma`` package."""

    try:
        import cma
    except Exception:  # pragma: no cover - optional dependency
        return _run_ga(task, budget)

    try:
        lows = np.array([task.space[k][0] for k in task.space])
        highs = np.array([task.space[k][1] for k in task.space])
        x0 = (lows + highs) / 2
        sigma = 0.2 * (highs - lows)
        popsize = min(max(2, budget), 4)
        es = cma.CMAEvolutionStrategy(
            x0, float(np.mean(sigma)), {"bounds": [lows, highs], "popsize": popsize}
        )

        fits: List[float] = []
        start = time.time()
        while len(fits) < budget:
            solutions = es.ask()
            vals = []
            for s in solutions:
                params = {k: float(s[i]) for i, k in enumerate(task.space)}
                loss = task.evaluate(params)
                vals.append(loss)
                fits.append(loss)
            es.tell(solutions, vals)
        duration = time.time() - start
        return fits[:budget], duration
    except Exception:  # pragma: no cover - errors fallback to GA
        return _run_ga(task, budget)


def _run_mcts(task: Task, budget: int) -> Tuple[List[float], float]:
    """Run the built-in MCTS-H optimizer."""

    from experiments.optim import MCTS_H

    opt = MCTS_H(list(task.space.keys()), cfg={"rng_seed": 0})
    fits: List[float] = []
    start = time.time()
    for _ in range(budget):
        cfg_norm = opt.suggest(1)[0]
        params = {
            k: float(cfg_norm[k] * (high - low) + low)
            for k, (low, high) in task.space.items()
        }
        loss = task.evaluate(params)
        fits.append(loss)
        opt.observe([{"config": cfg_norm, "objectives": {"loss": loss}}])
    duration = time.time() - start
    return fits, duration


RUNNERS = {
    "GA": _run_ga,
    "TPE": _run_tpe,
    "CMA-ES": _run_cma,
    "MCTS-H": _run_mcts,
}


# ---------------------------------------------------------------------------
# Ordering helpers


def optimizer_order(
    conditional: bool, summary_path: Path = Path("bench/optim/summary.csv")
) -> List[str]:
    """Return optimizers ordered by past performance.

    Parameters
    ----------
    conditional:
        Whether the target search space contains conditional branches.
    summary_path:
        Location of the benchmark summary produced by :func:`analyze`.

    Returns
    -------
    list[str]
        Optimizer names sorted by historical win counts with evaluation rate
        as a tie-breaker. If ``summary_path`` does not exist the default
        ``RUNNERS`` order is returned.
    """

    if not summary_path.exists():
        return list(RUNNERS)

    with summary_path.open() as fh:
        data = list(csv.reader(fh))

    try:
        sep = data.index([])
    except ValueError:
        return list(RUNNERS)

    task_rows = [r for r in data[1:sep] if len(r) == 3]
    rate_rows = [r for r in data[sep + 2 :] if len(r) == 3]

    wins: Dict[bool, Dict[str, int]] = {True: {}, False: {}}
    for _task, cond, winner in task_rows:
        flag = cond.lower() == "true"
        wins[flag][winner] = wins[flag].get(winner, 0) + 1

    rates: Dict[bool, Dict[str, float]] = {True: {}, False: {}}
    for opt, cond_label, rate in rate_rows:
        flag = cond_label == "conditional"
        rates[flag][opt] = float(rate)

    order = sorted(
        RUNNERS,
        key=lambda o: (wins[conditional].get(o, 0), rates[conditional].get(o, 0.0)),
        reverse=True,
    )
    return order


def select_optimizer(
    task_id: str, summary_path: Path = Path("bench/optim/summary.csv")
) -> str:
    """Return the highest-ranked optimizer for ``task_id``."""

    task = TASKS[task_id]
    return optimizer_order(task.conditional, summary_path)[0]


# ---------------------------------------------------------------------------
# Public API


def bench(task_id: str, budget: int) -> None:
    """Run the optimizer bake-off for ``task_id``."""

    task = TASKS[task_id]
    out_dir = Path("bench/optim")
    out_dir.mkdir(parents=True, exist_ok=True)

    order = optimizer_order(task.conditional)

    rows = []
    for name in order:
        runner = RUNNERS[name]
        fits, duration = runner(task, budget)
        best = float(np.min(fits)) if fits else float("nan")
        median = float(statistics.median(fits)) if fits else float("nan")
        rows.append((name, best, median, len(fits), duration))

    csv_path = out_dir / f"{task_id}.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["optimizer", "best", "median", "evals", "wall_clock_s"])
        writer.writerows(rows)

    md_path = out_dir / f"{task_id}.md"
    with md_path.open("w", encoding="utf8") as fh:
        fh.write("| optimizer | best | median | evals | wall_clock_s |\n")
        fh.write("|---|---|---|---|---|\n")
        for name, best, median, evals, dur in rows:
            fh.write(f"| {name} | {best:.6g} | {median:.6g} | {evals} | {dur:.3f} |\n")


def analyze() -> None:
    """Aggregate per-task results and summarise performance.

    This writes ``summary.csv`` and ``summary.md`` to ``bench/optim`` with the
    best optimizer per task and average evaluation rates split by conditional
    and non-conditional search spaces.
    """

    out_dir = Path("bench/optim")
    rows: List[Tuple[str, bool, str]] = []
    rates: Dict[bool, Dict[str, List[float]]] = {True: {}, False: {}}
    for task_id, task in TASKS.items():
        csv_path = out_dir / f"{task_id}.csv"
        if not csv_path.exists():
            continue
        with csv_path.open() as fh:
            data = list(csv.DictReader(fh))
        if not data:
            continue
        best_row = min(data, key=lambda r: float(r["best"]))
        rows.append((task_id, task.conditional, best_row["optimizer"]))
        for r in data:
            wall = float(r["wall_clock_s"])
            rate = float(r["evals"]) / wall if wall else float("nan")
            rates[task.conditional].setdefault(r["optimizer"], []).append(rate)

    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["task", "conditional", "winner"])
        writer.writerows(rows)
        writer.writerow([])
        writer.writerow(["optimizer", "conditional", "evals_per_s"])
        for cond, opts in rates.items():
            label = "conditional" if cond else "unconditional"
            for opt, vals in opts.items():
                writer.writerow([opt, label, statistics.fmean(vals)])

    win_counts: Dict[bool, Dict[str, int]] = {True: {}, False: {}}
    for _task, cond, winner in rows:
        win_counts[cond][winner] = win_counts[cond].get(winner, 0) + 1

    avg_rates: Dict[bool, Dict[str, float]] = {
        cond: {opt: statistics.fmean(vals) for opt, vals in opts.items()}
        for cond, opts in rates.items()
    }
    orders = {
        cond: sorted(
            RUNNERS,
            key=lambda o: (
                win_counts[cond].get(o, 0),
                avg_rates[cond].get(o, 0.0),
            ),
            reverse=True,
        )
        for cond in (True, False)
    }

    md_path = out_dir / "summary.md"
    with md_path.open("w", encoding="utf8") as fh:
        fh.write("| task | conditional | winner |\n")
        fh.write("|---|---|---|\n")
        for task_id, cond, winner in rows:
            fh.write(f"| {task_id} | {cond} | {winner} |\n")
        fh.write("\n| optimizer | conditional | evals_per_s |\n")
        fh.write("|---|---|---|\n")
        for cond, opts in rates.items():
            label = "conditional" if cond else "unconditional"
            for opt, vals in opts.items():
                fh.write(f"| {opt} | {label} | {statistics.fmean(vals):.3f} |\n")
        fh.write("\n")
        fh.write("Recommended order (conditional): " + " > ".join(orders[True]) + "\n")
        fh.write(
            "Recommended order (unconditional): " + " > ".join(orders[False]) + "\n"
        )


def bench_all(budget: int) -> None:
    """Run the bake-off for all tasks and aggregate results."""

    for task_id in TASKS:
        bench(task_id, budget)
    analyze()
