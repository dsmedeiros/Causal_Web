# Causal Web

Causal Web is a simulation engine and GUI for experimenting with causal graphs. Nodes emit ticks that propagate through edges with delay and attenuation while observers infer hidden state from the resulting activity. Delays now retain sub-tick precision and are quantised only when scheduled, enabling finer-grained simulations. The project is written in Python and uses [PySide6](https://doc.qt.io/qtforpython/) for the graphical interface.

Ticks carry both phase and amplitude. Their influence on interference and coherence is weighted by amplitude and each tick records the local `generation_tick` at which it was emitted.

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
`Config.log_delivery_sample_rate` to reduce I/O overhead. Edge hops can
also be throttled by setting `Config.logging.sample_edge_rate`
(0.0–1.0) which records only a fraction of per-hop `edge_delivery`
events; a value of `0.0` disables per-edge logs while still emitting a
per-window summary of edge activity.

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

## Table of Contents
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Output Logs](#output-logs)
- [Contributing](#contributing)

## Quick Start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pyside6
   ```
2. Run the GUI:
   ```bash
   python -m Causal_Web.main
   ```
   Or run headless:
   ```bash
   python -m Causal_Web.main --no-gui
   ```
3. Optional flags:
   - `--config <path>` to use a custom configuration file.
   - `--graph <path>` to load a different graph.
   - `--profile <file>` to write `cProfile` stats to the given path.
   - `--backend cupy` to enable GPU acceleration when available.

## Installation
Clone the repository and install the packages listed in `requirements.txt`. The GUI requires PySide6 and an X11 compatible display.

## Usage
Graphs are stored as JSON files under `Causal_Web/input/`. Each file defines
`nodes`, `edges`, optional `bridges`, `tick_sources`, `observers` and
`meta_nodes`. See [docs/graph_format.md](docs/graph_format.md) for the complete
schema and an example.

The GUI allows interactive editing of graphs. Drag nodes to reposition them and use the toolbar to add connections or observers. After editing, click **Apply Changes** in the Graph View to update the simulation and save the file. Details on all GUI actions are provided in [docs/gui_usage.md](docs/gui_usage.md).
During simulation a small HUD overlay reports the current tick, depth and
window when using the v2 engine, offering immediate feedback on scheduler
progress.
Nodes can optionally enable self-connections via a checkbox in the node panel. When enabled, dragging from a node back onto itself creates a curved edge.
Bridges now support an `Entanglement Enabled` option. When selected, the bridge
is tagged with an `entangled_id` used by observers to generate deterministic
measurement outcomes for Bell-type experiments.
Observers can enable a *Detector Mode* that records a binary outcome whenever a
tick from an entangled bridge is detected.
Bridge propagation now occurs before observers handle a tick so detector events
reflect entangled activity in the same cycle.
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

Runs produce a set of JSON logs in `output/`. The script `bundle_run.py` can be used after a simulation to archive the results. Full descriptions of each log file and their fields are available in [docs/log_schemas.md](docs/log_schemas.md).

## Configuration
Runtime parameters are loaded from `Causal_Web/input/config.json`. Any value can
be overridden with CLI flags using dot notation for nested keys, for example:
```bash
python -m Causal_Web.main --no-gui --max_ticks 20
```
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

The `engine_mode` flag selects the simulation core. The default `v2` value
enables the strict-local engine while `tick` selects the legacy implementation.
Parameter groups `windowing`, `rho_delay`, `epsilon_pairs`, and `bell` provide
advanced controls for the v2 engine.  Each group is a nested mapping:

```json
{
  "engine_mode": "v2",
  "windowing": {"W0": 4, "zeta1": 0.3, "zeta2": 0.3, "a": 0.7, "b": 0.4,
                 "T_hold": 2, "C_min": 0.1},
  "rho_delay": {"alpha_d": 0.1, "alpha_leak": 0.01, "eta": 0.2,
                "gamma": 0.8, "rho0": 1.0},
  "epsilon_pairs": {"delta_ttl": 8, "ancestry_prefix_L": 16,
                     "theta_max": 0.261799, "sigma0": 0.3,
                     "lambda_decay": 0.05, "sigma_reinforce": 0.1,
                     "sigma_min": 0.001},
  "bell": {"enabled": false, "mi_mode": "MI_strict", "kappa_a": 0.0,
            "kappa_xi": 0.0, "beta_m": 0.0, "beta_h": 0.0},
  "theta_reset": "renorm"
}
```

The `windowing` values control vertex window advancement. `rho_delay` affects
how edge density relaxes toward a baseline. `epsilon_pairs` governs dynamic
ε-pair behaviour – seeds with a limited TTL can bind to form temporary bridge
edges whose `sigma` values decay unless reinforced. The default `delta_ttl`
scales with `W0` (`2*W0`) to simplify experiments, while the remaining
parameters set decay and reinforcement dynamics. `bell` sets mutual information
gates for Bell pair matching. Bridge creation and removal now emit
`bridge_created` and `bridge_removed` events (carrying a stable synthetic
`bridge_id` and final `σ`), providing additional telemetry for analysis.
`seed_emitted` and `seed_dropped` events capture ε-pair propagation and
expiry reasons (`expired`, `angle`, `prefix`), aiding locality tests. The
Bell block is disabled by default;
set `"enabled": true` to activate measurement-interaction modes.

`run_seed` provides a deterministic seed used by sampling, Bell helpers and
ε-pair routines, allowing reproducible runs.

An adapter in ``engine_v2`` mirrors a subset of the legacy tick engine API and
generates *synthetic telemetry frames* so the GUI can tick while the new
physics is under development.  A telemetry frame is a simple structure:

```json
{"depth": 3, "events": 5, "packets": [{"src": 1, "dst": 2, "payload": null}]}
```

Each frame is also logged to `ticks_log.jsonl` with its sequential frame number,
coarse ``depth_bucket`` and the highest ``window_idx`` encountered. Additional
edge and vertex telemetry fields are appended without renaming existing keys so
existing ingestion remains compatible.

The adapter exposes methods like `build_graph`, `step`, `pause` and
`snapshot_for_ui` to remain drop-in compatible.  Internally a depth-based
scheduler orders packets by their arrival depth and advances vertex windows
using the Local Causal Consistency Model (LCCM).  The LCCM computes a window
size ``W(v)`` from the vertex's incident degree (fan-in plus fan-out) and local
density and transitions between
quantum ``Q``, decohered ``Θ`` and classical ``C`` layers with simple hysteresis
timers.  A lightweight loader converts graph JSON into struct-of-arrays via
``engine_v2.loader.load_graph_arrays`` to prime this core.
The LCCM recomputes the mean incident edge density ``ρ`` at every window
boundary so that subsequent window sizes adapt to current traffic.  Classical
dominance (Θ→C) additionally requires the majority-vote confidence to exceed
``conf_min`` alongside the existing bit fraction and entropy thresholds.

The `density_calc` option controls how edge density is computed. Set one of:

- `local_tick_saturation` (default) – density increases with recent traffic
- `modular-<mode>` – use a registered modular function (`tick_history`,
  `node_coherence`, `spatial_field`, `bridge_saturation`)
- `manual_overlay` – sample values from an external overlay file defined by
  `density_overlay_file`.

`density_calc` can also be specified via `--density-calc` on the command line.

Amplitude energy now feeds a stress–energy field that scales edge delay by
``1 + κρ``. This density diffuses each scheduler step with weight
``Config.density_diffusion_weight`` (``α``).

The helper ``engine.engine_v2.rho_delay.update_rho_delay`` applies this rule
per edge, adding leakage and layer-scoped external intensity and mapping the
resulting density to a logarithmically scaled effective delay. The engine v2 adapter
recomputes this ``d_eff`` on every packet delivery, storing it with the edge
and using the updated value to schedule the next hop. When a vertex window
closes the adapter normalises accumulated amplitudes and records ``EQ`` via
``engine.engine_v2.qtheta_c.close_window``. The post-window Θ distribution
reset policy is governed by ``Config.theta_reset`` which accepts ``"uniform"``
for an even reset, ``"renorm"`` to normalise existing values or ``"hold"`` to
leave the distribution unchanged (default ``"renorm"``).

Scheduler steps also integrate a toy horizon thermodynamics model. Interior
nodes may emit Hawking pairs with probability ``exp(-ΔE/T_H)``, and the
resulting radiation entropy follows a simple Page-curve: growing then
declining as the horizon evaporates. The energy quantum ``ΔE`` can be tuned at
runtime via ``Config.hawking_delta_e``.

The scheduler also supports a quantum micro layer via
``scheduler.run_multi_layer``. This helper executes ``micro_ticks`` quantum
steps before each classical update and calls a user-provided ``flush`` callback
to synchronise state between layers.

## GPU and Distributed Acceleration
The engine optionally accelerates per-edge calculations on the GPU when [Cupy](https://cupy.dev) is installed and can shard classical zones across a [Ray](https://www.ray.io) cluster. Classical nodes are partitioned into coherent zones and dispatched in parallel to Ray workers; if Ray is unavailable an info-level message is logged and processing continues locally. Select the backend with `Config.backend` or `--backend` (`cpu` by default, `cupy` for CUDA).

Ray cluster initialisation can be customised via the `--ray-init` flag. Pass a JSON string of keyword arguments forwarded to `ray.init`, for example:

```bash
python -m Causal_Web.main --ray-init '{"num_cpus":4}'
```

This enables tailoring the local or remote Ray cluster before sharding zones.

Edge propagation now batches phase factors and attenuations before offloading the
complex multiply to CuPy, falling back to vectorised NumPy when CUDA is
unavailable. A micro-benchmark under `tests/perf_edge_kernel.py` asserts the
kernel processes one million edges in under 100 ms on compatible hardware.

## Output Logs
Each run creates a timestamped directory under `output/runs` containing the graph, configuration and logs. Logging can be enabled or disabled via the GUI **Log Files** window or the `log_files` section of `config.json`. In `config.json` the keys are the categories (`tick`, `phenomena`, `event`) containing individual label flags. The `log_interval` option controls how often metrics and graph snapshots are written, while `logging_mode` selects which categories are written: `diagnostic` (all logs), `tick`, `phenomena` and `events`.
Logs are consolidated by category into `ticks_log.jsonl`, `phenomena_log.jsonl` and `events_log.jsonl`.
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

### Phase smoothing

The `smooth_phase` option applies exponential decay to each node's internal
oscillator phase. Enable it from the GUI control panel or set `"smooth_phase":
true` in `Causal_Web/input/config.json`.

### Propagation control

Check boxes on the control panel allow SIP budding, SIP recombination and
collapse seeded propagation to be disabled independently. The
`propagation_control` section of `Causal_Web/input/config.json` contains
`enable_sip_child`, `enable_sip_recomb` and `enable_csp` flags. These can also
be toggled via the CLI using `--disable-sip-child`, `--disable-sip-recomb` and
`--disable-csp`.

## Contributing
Unit tests live under `tests/` and can be run with `pytest`. Coding guidelines and packaging instructions are documented in [AGENTS.md](AGENTS.md) and [docs/developer_guide.md](docs/developer_guide.md).
