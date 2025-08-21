# Causal Web

Causal Web is a simulation engine and GUI for experimenting with causal graphs. Nodes emit frames that propagate through edges with delay and attenuation while observers infer hidden state from the resulting activity. Delays now retain sub-frame precision and are quantised only when scheduled, enabling finer-grained simulations. The project is written in Python and uses [PySide6](https://doc.qt.io/qtforpython/) for the graphical interface. The legacy PySide6 widget interface has been removed in favour of a Qt Quick / QML frontend.

A GPU-accelerated Qt Quick / QML interface lives in `ui_new` and serves as the default frontend. It renders graph nodes, edges and pulses via instanced `QSGGeometry`, exchanges `GraphStatic` and `SnapshotDelta` messages over a MessagePack WebSocket client, and tracks state via an immutable store with coalesced deltas. GPU buffers are built once and snapshot deltas modify only the affected instance offsets for positions, colors and visibility flags, while the view repaints just the touched region for minimal viewport updates. Per-instance attributes supply offsets, colors and visibility flags directly to the GPU so nodes, edges and pulses render in a single batch. Regression tests exercise graphs with up to 5,000 nodes to verify large-scene stability. GraphStatic can supply initial node labels, colors and visibility flags, and edges reuse a single line mesh with per-edge endpoints while parallel edges are collapsed before rendering. The interface includes basic panels for Telemetry, Meters, Experiment, Replay and Log Explorer, renders node labels, and applies level-of-detail rules that toggle label visibility, antialiasing and edge visibility based on zoom, exposing `labelsVisible` and `edgesVisible` properties so panels can react. Label text nodes are cached and updated in place to minimize scene-graph churn and avoid unnecessary painter state changes. Panels reside in a docked TabView on the right so the graph remains unobscured, and threshold values for label, edge and antialiasing visibility can be tuned via properties on `GraphView`. Users can zoom with the mouse wheel to scale the view and trigger these LOD changes. The Telemetry panel reports live node and edge counts alongside current LOD visibility, the Meters panel shows an estimated frames-per-second rate, the Experiment panel displays live status and residuals, the Replay panel tracks progress, and the Log Explorer streams log entries. Experiment, Replay and Log Explorer panels now offer interactive controls for starting, pausing or resetting experiments, controlling replay position, and filtering or clearing log entries. The Replay panel can load run directories for deterministic playback, streaming logged GraphStatic data and snapshot deltas, provides a timeline scrubber and supports bookmarks and frame annotations. Replays reproduce recorded HUD metrics such as frame count and residual trace for faithful analysis. DOE and GA result lists include a "Replay" button for one-click loading. Both panels also offer a "Promote to baseline" action that writes the selected configuration to `experiments/best_config.yaml` for reuse, with the path echoed in the Experiment panel's status bar. The DOE panel exposes range editors, LHS or grid sweeps, start/stop/resume controls, progress/ETA feedback and an interactive parallel-coordinate brush, while the GA panel surfaces population parameters, a live population table with objective and constraint flags, pause/resume and promote/export actions. A new Compare panel loads two runs side-by-side with synchronized playback controls, shows diff overlays and reports metric deltas. The backend runs experiments in a background thread, records deltas for replay and broadcasts updated experiment status and replay progress to keep panels in sync.

## Architecture & IPC

The engine and QML frontend communicate over a MessagePack WebSocket. The initial
`GraphStatic` message seeds the scene while subsequent `SnapshotDelta` messages
stream geometry and metric changes.

```json
GraphStatic = {
  "node_positions": [[x, y], ...],
  "edges": [[src, dst], ...],
  "node_labels": [...],
  "node_colors": [...],
  "node_flags": [...]
}

SnapshotDelta = {
  "frame": int,
  "node_positions": {id: [x, y], ...},
  "edges": [[src, dst], ...],
  "closed_windows": [[id, window], ...],
  "counters": {...},
  "invariants": {...}
}
```

On startup the engine prints a random session token. Clients begin with a
`Hello` message carrying this token and the server closes connections that omit
or mismatch it. A single client is accepted by default; set
`CW_ALLOW_MULTI=1` to allow additional read-only spectator clients. Only the
first connection retains control. Float32 fields keep payloads lean.

## Compare panel

The Compare panel accepts two run directories and plays their frames side-by-side. A scrubber and playback buttons keep the runs synchronized while an optional diff overlay and metric list highlight per-category deltas.

The interface exposes an Edit/Run toggle. Edit mode keeps the canvas interactive while Run mode locks edits, compiles the current graph into a deterministic `GraphDTO` and loads it into the engine.

Edit mode now includes basic editor tools for adding, selecting, connecting, dragging and deleting nodes alongside a property inspector panel for nodes, edges, observers and bridges plus a validation console that checks for duplicate identifiers, self-loops and missing properties.
Observers and bridges are rendered on the canvas and can be selected for editing.

Launch the GUI with:

```bash
python -m Causal_Web.main
```

Frames carry both phase and amplitude. Their influence on interference and coherence is weighted by amplitude and each frame records the local `generation_tick` at which it was emitted.

The engine now includes a lightweight quantum upgrade. Each node maintains a
two-component complex state vector `psi` instead of a single phase, edges can
optionally apply a Hadamard transform (`u_id=1`), and fan-in thresholds
`Config.N_DECOH` and `Config.N_CLASS` switch nodes between quantum,
thermodynamic, and classical behaviour. Hitting the classical threshold now
collapses a node to an eigenstate using the Born rule, while the decoherence
threshold preserves ``psi`` but freezes unitary evolution and records only the
resulting probability distribution.

Engine v2 stores graph data in a struct-of-arrays format using `float32` and
`complex64` types and batches deliveries by destination to vectorise quantum
accumulation. A bucketed scheduler keyed by integer depth reduces heap
operations to amortised *O*(1) and delivery logs may be sampled via
`Config.log_delivery_sample_rate` to reduce I/O overhead. Per-edge
ρ/delay updates are sampled via `Config.logging.sample_rho_rate`
(defaults to 0.01) which records only a fraction of per-hop `edge_delivery`
events and is further capped at roughly one log per 100 edges processed.
Seed and bridge events are sampled the same way using
`Config.logging.sample_seed_rate` and `Config.logging.sample_bridge_rate`
(both default to 0.01).

To cap memory growth for long coherent lines, the engine detects tensor clusters
and represents them as Matrix Product States. Local edge unitaries contract with
these tensors and singular values beyond ``Config.chi_max`` are truncated. A
chain of one hundred Hadamards now consumes less than 50 MB with under one
percent numerical error.

Each node also accumulates a proper-time `tau` that accounts for local velocity
and density effects. Run `python -m Causal_Web.analysis.twin` for a simple
twin-paradox demonstration showcasing this time dilation. Run
`python -m Causal_Web.analysis.lensing` to approximate lensing wedge amplitudes
via a Monte-Carlo path sampler over the graph's causal structure.

- The UI disables its controls within a few seconds if the engine connection is
  lost.
- Fixed runaway zoom in the frames graph that occurred on startup.
- Closing the GUI no longer hangs; the engine worker thread now shuts down
  cleanly.
- GUI now includes a status bar with frame metrics, an engine profile panel
  and lightweight real-time plots using ``pyqtgraph``.
- Canvas performance improved with minimal viewport updates, item caching and
  label level-of-detail. HUD wording clarified and telemetry buffers capped.
- Telemetry panel now exposes rolling counter and invariant histories for live plots.
- HUD overlay reports frame, depth, active windows, active bridges, FPS,
  events per second and residual metrics.
- Experiment panel adds single-step controls, a rate slider and label/edge
  visibility toggles.
- Canvas renders the latest snapshot diffs at up to 60 FPS via a pull-based loop.
- Client coalesces snapshot notifications and reuses scratch buffers (including
  pooled unitary, phase and alpha scaling arrays) while edge and event logging respect budgets
  unless diagnostics are enabled.
- Scratch buffers for ψ and p are bucketed by group size to curb allocations and
  snapshot deltas now encode float32 metrics and positions for leaner payloads.
- Telemetry histories cap at roughly 3k samples per series and DeltaReady
  messages drop older snapshots so only the newest frame renders between paints.
- Window closures trigger brief red pulses on affected nodes for visual feedback.
- GraphView exposes `save_snapshot(path, duration=0.0, fps=30)` to capture the current canvas as a PNG image or MP4 clip. For
  MP4 exports both `duration` and `fps` must be positive.
- A headless `video_export` utility under `tools/` composites logged snapshots
  into an MP4 without launching the GUI.
- Fixed a startup crash in read-only mode where a stale HUD item was
  re-added after clearing the scene.
- Visible "Tick" terminology has been replaced with "Frame" throughout the
  interface and tooltips.
- Added local Forman curvature diagnostics with per-region statistics and
  histogram logging.
- Introduced a switchable delay mapping system with the existing logarithmic
  rule (`log_scalar`) and a new Φ-linear variant (`phi_linear`).
- Seed carrying now iterates through sorted depth arrivals, preserving
  per-depth causality within a batch.
- Baseline edge delays are floored to integers and the initial effective delay
  `d_eff` uses this coerced value.
- Non-incoming injection modes average per-packet intensities to avoid
  saturation with high fan-in.
- Added DOE runner with invariant checks and metrics logging.
- Runner CLI now accepts separate experiment (`--exp`) and base (`--base`)
  configs, persists per-sample seeds and gate metrics, supports
  parallel execution via `--parallel` (use `--processes` for a process pool),
  and can re-evaluate cached configurations with `--force`. The genetic
  algorithm runner exposes the same `--force` option to rerun cached genomes.
- DOE summaries now record selected gates and aggregate gate metrics
  (mean and standard deviation).
- DOE queue manager can generate Latin Hypercube or grid sweeps, tracks live per-run invariant and fitness status, and dispatches runs to the engine via IPC.
- New DOE panel integrates the queue manager into the Qt Quick UI with a Top-K table, scatter plot, parallel-coordinate view linking groups to metric axes with brush filtering, fitness heatmap and live status updates for sampled configurations.
- Added a lightweight Genetic Algorithm framework with tournament selection, uniform crossover, Gaussian mutation and elitism along with a GA panel showing a live population table with objectives and per-constraint flags, fitness-history and Pareto-front charts, and promote/export actions.
- Introduced scalar fitness helpers with hard invariant guardrails and normalised terms, providing a clear objective for optimisation and a path toward multi-objective Pareto support.
- Added NSGA-II-lite multi-objective capabilities with non-dominated sorting, crowding-distance selection, a persistent Pareto archive and UI promotion of chosen trade-offs.
- GA panel now offers a multi-objective toggle, Pareto scatter with selectable objectives, descriptive axis labels, and a table showing rank and crowding distance with a promotion dialog.
- DOE and GA batches now persist summaries under ``experiments/``:
  - ``top_k.json`` records the best runs and is consumed by the Top-K UI table.
  - ``hall_of_fame.json`` archives per-generation GA champions.
  - ``best_config.yaml`` captures the promoted configuration for quick reuse.
  - run directories under ``experiments/runs/<date>/<id>/`` hold per-run ``config.json``, ``result.json`` and ``delta_log.jsonl`` for replay.
- GA evaluation can dispatch genomes to the engine via IPC, with engine-side handling of ``ExperimentControl`` messages for ``run`` requests.
- GA panel evaluations now use the shared IPC loop so genomes are executed on the engine during interactive runs.
- GA runs can be checkpointed and later resumed from disk—including any in-flight evaluations—to support reproducible interrupted searches.
- Sweeps and GA populations reuse existing results via a persistent run index
  at ``experiments/runs/index.json`` keyed by a deterministic run hash,
  skipping duplicate configurations and allowing interrupted batches to
  resume safely. Command-line sweeps accept ``--force`` to re-evaluate
  configurations even when present in the index.
- Gate harness now executes Gates 1–6 via engine primitives rather than
  returning proxy metrics.
- Gate metrics now capture interference visibility, delay slopes and
  relaxation times, LCCM hysteresis transition depths, ε-pair locality
  statistics and CHSH scores.
- Gate harness parameters are sourced from run configuration, removing
  hard-coded defaults.
- CI workflow automatically bumps the project version using semantic
  versioning rules derived from Conventional Commits.
  statistics, conservation residuals and CHSH outcomes with marginal
  bias for Gates 1–6.
- Introduced split-step quantum walk helpers with dispersion and lightcone
  experiment utilities and configuration knobs.
- Metrics logger now emits `summary_invariants.json` with pass rates for gate
  invariants.
- Replaced legacy `NodeManager` with v2 `EngineAdapter` and removed the former
  from the public API.
- `EngineAdapter` now exposes a `get_engine()` factory; the module no longer
  instantiates a global engine at import time.
- `EngineAdapter.step` accepts a `collect_packets` flag to capture processed packets
  for debugging; collection is disabled by default to reduce memory overhead.

## Table of Contents
- [Architecture & IPC](#architecture--ipc)
- [Quick Start](#quick-start)
- [Troubleshooting](#troubleshooting)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Output Logs](#output-logs)
- [Contributing](#contributing)

## Quick Start
```bash
pip install -r requirements.txt
cw run             # GUI
cw run --no-gui    # headless
cw sweep --lhs ... # DOE
cw ga --config ... # GA
```

## Troubleshooting
- **Disconnected** → token mismatch or single-client limit.
- **Low FPS** → zoom out, labels auto-hide; disable AA at far zoom.
- **Invariant failures** → inspect `result.json` for violations.

## Installation
Clone the repository and install the packages listed in `requirements.txt`. The GUI requires an X11 compatible display.

## Usage
Graphs are stored as JSON files under `Causal_Web/input/`. Each file defines
`nodes`, `edges`, optional `bridges`, `tick_sources`, `observers` and
`meta_nodes`. See [docs/graph_format.md](docs/graph_format.md) for the complete
schema and an example.

The GUI allows interactive editing of graphs. Drag nodes to reposition them and use the toolbar to add connections or observers. After editing, click **Apply Changes** in the Graph View to update the simulation and save the file. Details on all GUI actions are provided in [docs/gui_usage.md](docs/gui_usage.md).
During simulation a small HUD overlay reports the current frame, depth and window when using the v2 engine, offering immediate feedback on scheduler progress.
Nodes can optionally enable self-connections via a checkbox in the node panel. When enabled, dragging from a node back onto itself creates a curved edge.
Bridges now support an `Entanglement Enabled` option. When selected, the bridge
is tagged with an `entangled_id` used by observers to generate deterministic
measurement outcomes for Bell-type experiments.
Observers can enable a *Detector Mode* that records a binary outcome whenever an
arrival-depth from an entangled bridge is detected. Bridge propagation now
occurs before observers handle an arrival-depth so detector events reflect
entangled activity in the same cycle.
These detector events are additionally written to `entangled_log.jsonl` for
Bell inequality analysis.
The underlying Bell helpers now track explicit 256-bit ancestry hash lanes
(`h0`–`h3`) and a three-component moment vector (`m0`–`m2`) with an
associated normalisation (`m_norm`). Detector settings can be drawn either
strictly independently or from a von Mises–Fisher distribution conditioned on
shared ancestry.  This toggle enables controlled measurement dependence
studies.
Graphs may also define ``epsilon`` edges linking two nodes in a singlet state.
When one node collapses, its ``epsilon`` partner is projected onto the opposite
eigenvector, enabling Bell-test correlations without a bridge. Collapse now
propagates regardless of edge direction so even a single directed ``epsilon``
edge enforces the pairing.
Nodes flagged with ``cnot_source`` dynamically convert their first two outgoing
edges into such an ``epsilon`` pair whenever they fire. Each edge additionally
supports an ``A_phase`` parameter representing a U(1) gauge potential; ticks
traversing the edge accumulate this phase shift. Serialized graphs now
preserve ``cnot_source`` flags and their resulting ``epsilon`` pairings on
reload.
The GUI now includes an **Analysis** menu with a *Bell Inequality Analysis...*
action that opens a window showing CHSH statistics and a histogram of
expectation values. Any metadata fields found in `entangled_log.jsonl`, such as
`mi_mode`, `kappa_a`, `kappa_xi`, `batch_id` or `h_prefix_len`, are displayed
alongside the CHSH score.

Runs produce a set of JSON logs in `output/`. Full descriptions of each log file and their fields are available in [docs/log_schemas.md](docs/log_schemas.md).

## Configuration
Runtime parameters are loaded from `Causal_Web/input/config.json`. Any value can
be overridden with CLI flags using dot notation for nested keys, for example:
```bash
python -m Causal_Web.main --no-gui --max_ticks 20
```
sets a depth limit of 20 for a headless run.
CLI flags for nested `log_files` entries now include the full path prefix to avoid duplicate
argument names. For example, use `--log_files.tick.coherence_log false` to disable a single tick log.
Use `--init-db` to create PostgreSQL tables defined in the configuration and exit.
Additional flags allow enabling or disabling specific logs without editing
`config.json`:

```bash
python -m Causal_Web.main --disable-tick=coherence_log,interference_log \
    --enable-events=bridge_rupture_log
```

The `chi_max` option caps the bond dimension used when compressing linear
chains into Matrix Product States. Raising it reduces truncation error at the
cost of memory.

The `backend` option selects the compute backend. It defaults to `cpu` but
may be set to `cupy` for CUDA acceleration.

The `engine_mode` flag selects the simulation core via the `EngineMode` enum.
The default and only supported value `v2` (`EngineMode.V2`) enables the
strict-local engine; the legacy `tick` engine has been removed. Parameter
groups `windowing`, `rho_delay`, `epsilon_pairs`, `ancestry`, and `bell` provide
advanced controls for the v2 engine. Each group is a nested mapping:

```json
{
  "engine_mode": "v2",
  "windowing": {"W0": 4, "zeta1": 0.3, "zeta2": 0.3, "a": 0.7, "b": 0.4,
                 "T_hold": 2, "C_min": 0.1, "k_rho": 1.0},
  "rho_delay": {"alpha_d": 0.1, "alpha_leak": 0.01, "eta": 0.2,
                "gamma": 0.8, "rho0": 1.0, "inject_mode": "incoming",
                "vectorized": true},
  "rho": {"update_mode": "heuristic", "variational": {"lambda_s": 0.2, "lambda_l": 0.01, "lambda_I": 1.0}},
  "lccm": {"mode": "thresholds", "free_energy": {"k_theta": 1.0, "k_c": 1.0, "k_q": 0.2, "F_min": 0.3}},
  "epsilon_pairs": {"delta_ttl": 8, "ancestry_prefix_L": 16,
                     "theta_max": 0.261799, "sigma0": 0.3,
                     "lambda_decay": 0.05, "sigma_reinforce": 0.1,
                     "sigma_min": 0.001, "decay_interval": 32,
                     "decay_on_window_close": true,
                     "max_seeds_per_site": 64,
                     "emit_per_delivery": false},
  "ancestry": {"beta_m0": 0.1, "delta_m": 0.02},
  "bell": {"enabled": false, "mi_mode": "MI_strict", "kappa_a": 0.0,
            "kappa_xi": 0.0, "beta_m": 0.0, "beta_h": 0.0},
  "theta_reset": "renorm",
  "max_deque": 8
}
```

The `windowing` values control vertex window advancement. `rho_delay` affects how edge density relaxes toward a baseline. The `rho` group selects the update rule for ρ while `lccm` chooses the Θ→C transition criterion.
The `inject_mode` option selects
whether ρ input applies to `"incoming"` (default), `"incident"` or `"outgoing"` edges.
Non-incoming modes average per-packet Θ intensities over the window to set the injection intensity. The optional
`vectorized` flag toggles batch updates: set it to `false` to apply per-delivery Gauss–Seidel ordering when studies
require exact event sequencing. With vectorized updates enabled (the default) neighbour sums are maintained incrementally so large fan-in bursts avoid full recomputation while approximating Gauss–Seidel behaviour.
`epsilon_pairs` governs dynamic
ε-pair behaviour – seeds with a limited TTL can bind to form temporary bridge
edges whose `sigma` values decay unless reinforced. The default `delta_ttl`
scales with `W0` (`2*W0`) to simplify experiments, while the remaining
parameters set decay and reinforcement dynamics. `decay_interval` controls how
often bridges decay and `decay_on_window_close` toggles a decay step when a
window closes. `max_seeds_per_site` bounds how many unmatched seeds a vertex
retains, evicting the oldest when full. An `overflow_drops` counter records how
many seeds are removed due to this limit for post-run diagnostics.
`emit_per_delivery` enables a high-fidelity mode where seeds emit on each
Q-delivery instead of the default moment-angle emission once per
vertex-window. The `ancestry` group tunes
phase-moment updates and decay. `bell` sets mutual information gates for Bell
pair matching. The hidden variable ζ is drawn from the destination's ancestry
hash with optional noise blended according to `beta_h`. Bridge creation and removal now emit `bridge_created` and
`bridge_removed` events (carrying a stable synthetic `bridge_id` and final `σ`),
providing additional telemetry for analysis.
`seed_emitted` and `seed_dropped` events capture ε-pair propagation and
expiry reasons (`expired`, `angle`, `prefix`), aiding locality tests. The
Bell block is disabled by default;
set `"enabled": true` to activate measurement-interaction modes.

The top-level `max_deque` knob sets the length of the classical majority buffer.
Within the Bell block, setting `"kappa_xi": 0` yields maximal measurement noise.
`zeta_mode` selects how the hidden scalar is computed: the default
`"float"` maps a 64‑bit hash to `[0,1)` while `"int_mod_k"` preserves the
previous discrete modulo behaviour via `k_mod`.  Detector rotations are scaled
by `alpha_R`.

`run_seed` provides a deterministic seed used by sampling, Bell helpers and
ε-pair routines, allowing reproducible runs.

An adapter in ``engine_v2`` mirrors a subset of the legacy tick engine API and
generates *synthetic telemetry frames* so the GUI can tick while the new
physics is under development.  A telemetry frame is a simple structure:

```json
{"depth": 3, "events": 5, "packets": [{"src": 1, "dst": 2, "payload": null}]}
```

Each frame is also logged to `ticks_log.jsonl` with its sequential `frame` number,
coarse ``depth_bucket`` and the highest ``window_idx`` encountered. Additional
edge and vertex telemetry fields are appended without renaming existing keys so
existing ingestion remains compatible.

The adapter exposes methods like `build_graph`, `step`, `pause` and
`snapshot_for_ui` (returning a `ViewSnapshot` dataclass) to remain drop-in
compatible. Snapshots include the identifiers of nodes and edges touched since
the previous call along with any closed window events, allowing the GUI to
incrementally update its model. An EWMA of the energy conservation residual
is tracked per window and surfaced via the snapshot counters. Additional
micro-counters such as ``windows_closed``, ``bridges_active``,
``events_processed`` and ``edges_traversed`` are exposed alongside global
``residual_ewma`` and per-window ``residual_max`` to aid diagnostics. For Bell experiments the
adapter also tracks a no-signaling delta under ``inv_no_signaling_delta``. Internally a depth-based
scheduler orders packets by their arrival depth and advances vertex windows
using the Local Causal Consistency Model (LCCM).  The LCCM computes a window
size ``W(v)`` from the vertex's incident degree (fan-in plus fan-out) and local
quantum ``Q``, decohered ``Θ`` and classical ``C`` layers with simple hysteresis
timers.  A lightweight loader converts graph JSON into struct-of-arrays via
``engine_v2.loader.load_graph_arrays`` to prime this core.
The LCCM recomputes the mean incident edge density ``ρ`` at every window
boundary so that subsequent window sizes adapt to current traffic.  Classical
dominance (Θ→C) additionally requires the majority-vote confidence to exceed
``conf_min`` alongside the existing bit fraction and entropy thresholds.

An event-driven helper ``on_window_close`` provides adaptive window sizing
using an exponentially weighted moving average of neighbour densities.  The
function maintains ``M_v`` and ``W_v`` per vertex and rate-limits adjustments to
avoid oscillation while keeping memory usage ``O(1)``.

Amplitude energy now feeds a stress–energy field that scales edge delay by
``1 + κρ``. This density diffuses each scheduler step with weight
``Config.density_diffusion_weight`` (``α``).

The helper ``engine.engine_v2.rho_delay.update_rho_delay`` applies this rule
per edge, adding leakage and layer-scoped external intensity and mapping the
resulting density to a logarithmically scaled effective delay. The engine v2 adapter
recomputes this ``d_eff`` on every packet delivery, storing it with the edge
and using the updated value to schedule the next hop. When a vertex window
closes the adapter normalises accumulated amplitudes and records ``EQ`` via
``engine.engine_v2.qtheta_c.close_window``. The Θ and C meters ``E_theta`` and
``E_C`` along with the incident-density meter ``E_rho`` are persisted to the vertex arrays for diagnostics. If a vertex is in the C layer its bit and confidence persist across windows and are cleared only when leaving C. The post-window Θ
distribution reset policy is governed by ``Config.theta_reset`` which accepts
``"uniform"`` for an even reset, ``"renorm"`` to normalise existing values or
``"hold"`` to leave the distribution unchanged (default ``"renorm"``).

Scheduler steps also integrate a toy horizon thermodynamics model. Interior
nodes may emit Hawking pairs with probability ``exp(-ΔE/T_H)``, and the
resulting radiation entropy follows a simple Page-curve: growing then
declining as the horizon evaporates. The energy quantum ``ΔE`` can be tuned at
runtime via ``Config.hawking_delta_e``.

The scheduler also supports a quantum micro layer via
``scheduler.run_multi_layer``. This helper executes ``micro_ticks`` quantum
steps before each classical update and calls a user-provided ``flush`` callback
to synchronise state between layers.

The ``qwalk`` configuration group toggles a lightweight split-step quantum
walk mode. When enabled, the ``thetas`` mapping provides the two rotation
angles used by the coin operators. Dispersion analysis samples wave numbers
from ``dispersion.k_values``:

```json
"qwalk": {"enabled": true,
          "thetas": {"theta1": 0.35, "theta2": 0.2}},
"dispersion": {"k_values": [0.0, 0.1, 0.2]}
```

These settings drive the helper modules in ``experiments.dispersion`` and
``experiments.lightcone``.

## GPU and Distributed Acceleration
The engine optionally accelerates per-edge calculations on the GPU when [Cupy](https://cupy.dev) is installed and can shard classical zones across a [Ray](https://www.ray.io) cluster. Classical nodes are partitioned into coherent zones and dispatched in parallel to Ray workers; if Ray is unavailable an info-level message is logged and processing continues locally. Select the backend with `Config.backend` or `--backend` (`cpu` by default, `cupy` for CUDA). Set `CW_USE_CUPY=1` to enable optional vectorised kernels for heavy sweeps when using the `cupy` backend.

Density accumulation and matrix-product-state propagation also leverage CuPy
when the backend is set to `cupy`, keeping computation on the GPU.

Ray cluster initialisation can be customised via the `--ray-init` flag. Pass a JSON string of keyword arguments forwarded to `ray.init`, for example:

```bash
python -m Causal_Web.main --ray-init '{"num_cpus":4}'
```

This enables tailoring the local or remote Ray cluster before sharding zones.

Edge propagation now batches phase factors and attenuations before offloading the
complex multiply to CuPy, falling back to vectorised NumPy when CUDA is
unavailable. A micro-benchmark under `tests/perf_edge_kernel.py` asserts the
kernel processes one million edges in under 100 ms on compatible hardware.
Edge phases ``exp(1j * (phi + A))`` are cached during graph loading so packet
delivery avoids redundant exponentials.

## Benchmarks

`bench/engine_bench.py` measures engine step throughput while
`bench/gui_bench.py` records GUI frame rate using an offscreen Qt
renderer. Both benchmarks run nightly in CI, comparing results against
checked-in baselines and uploading JSON artifacts. Regressions greater than
5% generate workflow warnings while slowdowns beyond 10% fail the job. Any
regression over 5% also opens a GitHub issue so maintainers receive external
notifications. Both benchmarks record basic hardware metadata for reproducibility. Manual
GUI frame-rate measurement guidelines remain in `bench/gui_fps.md` for
scenarios not covered by the automated benchmark.

The GUI benchmark accepts ``--nodes`` to control graph size and
``--aa/--no-aa`` and ``--labels/--no-labels`` flags to toggle
anti-aliasing and label rendering. Optional ``--target-fps`` records
runtime expectations and ``--machine`` adds free-form notes; detailed
hardware metadata is collected automatically in the output JSON.

## Output Logs
Each run creates a timestamped directory under `output/runs` containing the graph, configuration and logs. Logging can be enabled or disabled via the GUI **Log Files** window or the `log_files` section of `config.json`. In `config.json` the keys are the categories (`tick`, `phenomena`, `event`) containing individual label flags. Logging cadence is event-driven; metrics and graph snapshots are written when windows advance or other triggers occur. The `logging_mode` option selects which categories are written: `diagnostic` (all logs), `tick`, `phenomena` and `events`.
Logs are consolidated by category into `ticks_log.jsonl`, `phenomena_log.jsonl` and `events_log.jsonl`.
An accompanying `metrics.csv` summarises event counts per frame in a tall format
with columns ``frame``, ``category`` and ``count``. Records labelled
``adapter_frame`` are omitted from these totals.
Individual files can still be toggled via `log_files` for advanced filtering.
Entangled tick metadata and detector outcomes are stored separately in
`entangled_log.jsonl`.
Law-wave propagation events now appear as `law_wave_event` records in the `events` log.
Per-tick law-wave frequencies are still written under the `law_wave_log` label in `ticks_log.jsonl`.
The interpreter provides `records_for_tick()` and `assemble_timeline()` helpers to query these consolidated logs.
The ingestion service also consumes these unified files, routing records to database tables based on their `label` or `event_type`.

## Diagnostics Sweep
Parameter sweeps can be scripted in YAML and executed via `tools/sweep.py`. Each
experiment logs metrics to a CSV file and produces a Matplotlib heat-map.

Example configuration:

```yaml
bell:
  epsilon: [true, false]
interference:
  fan_in: [0, 3]
twin:
  velocity: [0.2, 0.4, 0.6]
```

Run the sweep with:

```bash
cw-sweep sweep.yml
```

or directly via the module:

```bash
python tools/sweep.py sweep.yml
```

The resulting `*_sweep.csv` and `*_heatmap.png` files summarise Bell scores,
interference visibility and proper-time ratios. Sweeps seed module-level random
generators, so parallel sweeps should run in separate processes to avoid
contention.

## Contributing

Run style checks with `pre-commit`:

```bash
pre-commit run --files <path>
```

Unit tests live under `tests/` and can be run with `pytest`. Golden replay logs in `tests/goldens/` span a few hundred frames from production runs and are exercised via `pytest tests/test_replay_golden.py` with unique residual profiles per log. The helper functions `Causal_Web.engine.replay.build_engine` and `replay_from_log` load either run directories or trimmed delta logs for these regressions. New golden logs can be recorded from engine runs using `tools/record_golden.py`. Coding guidelines and packaging instructions are documented in [AGENTS.md](AGENTS.md) and [docs/developer_guide.md](docs/developer_guide.md).
