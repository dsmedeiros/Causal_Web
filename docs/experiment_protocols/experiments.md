# Experiments

The experiment helpers now expose a simple optimiser interface. ``MCTS_H`` can
be initialised with priors derived from Top-K results using
``experiments.optim.build_priors`` and queried for new configurations via
``suggest``. Completed runs are reported with ``observe`` so search statistics
are updated. When only a proxy metric is available the optimiser can decide to
promote the configuration for a follow-up full evaluation when the proxy score
is sufficiently good. Multi-objective searches may be enabled by setting
``multi_objective`` in the optimiser configuration which causes each evaluation
to scalarise its objective vector using random Dirichlet weights. Pass
``--multi-objective`` to ``cw optim`` to enable this from the command line.
Optimiser state may be checkpointed with ``save`` and later restored with
``load`` to resume an interrupted search.

For ad-hoc optimisation runs the ``cw optim`` command wires the optimiser
into the existing gate runner and persists results to the usual Top-K and
hall-of-fame artifacts. Supplying ``--state`` checkpoints the optimiser after
each evaluation so searches can be resumed later. ``--proxy-frames`` and
``--full-frames`` control the frame budgets for the initial proxy and promoted
full evaluations. ``--bins`` sets the number of quantile bins used when
building priors and ``--promote-quantile`` applies a percentile-based
 promotion policy. ``--promote-window`` restricts the quantile to recent
  proxy scores while ``--multi-objective`` toggles Dirichlet scalarisation of
  multiple objectives.
Full evaluation manifests include an ``mcts_run_id`` so runs can be linked
back to the originating search session.
