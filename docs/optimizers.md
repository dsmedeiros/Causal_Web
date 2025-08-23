# Optimizers

## When to use which optimizer

Choose an optimizer based on the characteristics of your search space:

- **MCTS-H** – leverage when you have a cheap proxy metric and want to expand a
  tree incrementally, promoting only promising configurations.
- **GA** – suited to large or multi-objective spaces where populations of
  candidates explore broadly.
- **TPE** – effective for conditional or categorical spaces and when you want a
  quick suggestion-based search without maintaining a population.
- **CMA-ES** – best for low-dimensional, smooth and purely continuous
  parameters.

## MCTS-H

The Monte Carlo Tree Search optimiser explores hyperparameter spaces as a
sequential decision tree. Nodes expand according to progressive widening and
rollouts draw samples from priors built from previous top-performing runs.
The implementation supports incremental observation via ``suggest`` and
``observe`` calls and can resume deterministically when seeded with the same
random state. Partial assignments are cached in a transposition table so that
different branches of the search tree can share statistics. Candidates are
first evaluated using a proxy metric and only promoted for a full evaluation
when the proxy score passes a configurable threshold or falls within a chosen
quantile. When configured with ``multi_objective`` the optimiser draws random
Dirichlet weights to scalarise objective vectors for backpropagation. Enable
this mode by passing ``--multi-objective`` to ``cw optim`` or toggling the
option in the UI.
State, including the RNG, can be saved to and restored from JSON checkpoints
using ``save`` and ``load`` to resume searches across sessions.

Run the optimiser from the command line via ``cw optim``:

```bash
cw optim --base base.yaml --space space.yaml --optim mcts_h \
    --state mcts_state.json --budget 100 --proxy-frames 300 --full-frames 3000 \
    --bins 3 --promote-quantile 0.6 --promote-window 20 --multi-objective
```

The optimiser defaults to ``c_ucb=0.7``, ``alpha_pw=0.4`` and ``k_pw=1`` with
promotion based on the 60th percentile of recent proxy scores.

Passing ``--state`` ensures the optimiser writes its state to ``mcts_state.json``
after each evaluation and reloads it on subsequent invocations.

The Qt Quick UI also exposes an ``MCTS`` tab, allowing interactive tree search runs alongside the existing DOE and GA panels.

Regression tests compare MCTS-H against the genetic algorithm on a toy task,
demonstrating that MCTS-H reaches comparable fitness with fewer full
evaluations.

To aid interpretation, ``experiments.ablation.local_ablation`` computes
partial dependence slices around the best discovered configuration. These
1D or 2D sweeps highlight which dimensions most influence optimisation near
the optimum and can be triggered from the MCTS tab via the *Local Ablation*
button, with the resulting curves and heatmaps rendered directly below the
controls. Telemetry plots now surface bootstrapped confidence bands on rolling
metrics so uncertainty is visible directly in the UI.

Leaf evaluations can be processed in parallel via ``OptimizerQueueManager.run_parallel``.
Passing ``parallel>1`` and ``use_processes=True`` dispatches rollouts to a
process pool, while ``use_ray=True`` ships evaluations to a Ray cluster.
Both modes retain deterministic seed streams and honour ASHA rung scheduling
when configured.

The ``experiments/calibrate_mcts_h.py`` helper sweeps ``c_ucb``, ``alpha_pw``,
``k_pw`` and initial prior bins over a canonical single-node graph. The sweep
now explores bins ``{3,5,7}`` to provide deeper insight into prior
discretisation.
