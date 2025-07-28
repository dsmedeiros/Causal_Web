# Causal Web

This project contains a small simulation engine and GUI for experimenting with causal graphs. It is written in Python and uses [PySide6](https://doc.qt.io/qtforpython/) for the graphical interface.

## Overview

The engine models a directed network of nodes. Each node maintains its own oscillator phase and can emit "ticks" that travel along edges with delay and attenuation. Nodes accumulate incoming phases and fire when they pass a threshold, scheduling more ticks. Bridges create additional links whose strength can change over time. Observers watch the network and attempt to infer hidden state.

The graph editor supports undo/redo operations via ``Ctrl+Z``/``Ctrl+Y``,
allows connecting nodes by dragging between them, applies an automatic spring
layout based on ``networkx`` and validates connections to prevent duplicates or
self-loops. Observers appear as draggable squares connected to each of their
target nodes with dotted lines and open the observer panel when clicked.
Right-clicking the canvas opens a menu to insert new nodes, observers or meta
nodes at the clicked location. Right-clicking existing items exposes a delete
option that removes the object and any associated links.

Key modules include:

- **`engine/graph.py`** – container for nodes, edges and bridges. Graphs can be loaded from or written to JSON files.
- **`engine/node.py`** – implementation of `Node`, `Edge` and related logic.
- **`engine/bridge.py`** – manages dynamic bridges between nodes.
- **`engine/tick_engine/`** – modular package driving the simulation and logging metrics under `output/`.
- **`engine/tick_engine/orchestrators.py`** – separates evaluation, mutation and I/O duties for the simulation loop.
- **`engine/tick_router.py`** – moves ticks through LCCM layers and logs transitions.
- **`engine/tick_seeder.py`** – seeds periodic ticks based on the configuration file.
- **`engine/log_interpreter.py`** – parses the generated logs and aggregates statistics.
- **`engine/causal_analyst.py`** – infers causal chains and produces explanation files.
- **`engine/meta_node.py`** – groups clusters of nodes into collapsed meta nodes.
- **`engine/node_manager.py`** – NumPy-backed container for bulk node updates
  using dynamically resized pre-allocated arrays.
- **`engine/observer.py`** – observers that infer hidden state from tick history.
- **`engine/logger.py`** – centralized buffer that batches log writes to disk.
- **`engine/tick.py`** – defines :class:`Tick` and the reusable object pool.
- **`gui_pyside/main_window.py`** – PySide6 dashboard with a dockable canvas and
  toolbar for interactive runs.
- **`gui_pyside/canvas_widget.py`** – reusable ``QGraphicsView`` for graph rendering.
- **`main.py`** – simple entry point that launches the dashboard.

Graphs are stored in `input/graph.json` inside the package. Paths are resolved relative to the package so the module can be run from any working directory. All output is written next to the code in the `output` directory.

## Configuration

Runtime parameters such as tick rate or seeding strategy can be overridden by
providing a JSON configuration file. Invoke the module with the `--config`
argument and optionally override individual keys via CLI flags:

```bash
python -m Causal_Web.main --config Causal_Web/input/config.json
```

Flags take the form `--<key>` where `<key>` matches an entry in the
configuration file (nested keys use dot notation). For example to run the
simulation headless for 20 ticks you can use:

```bash
python -m Causal_Web.main --no-gui --max_ticks 20
```

The `max_ticks` value takes effect when `allow_tick_override` is enabled in
the configuration (the default behaviour).

Only keys matching attributes on `Causal_Web.config.Config` are applied. Nested
dictionaries merge with the existing values.

To set up a PostgreSQL database using the credentials in the configuration file,
invoke the module with the `--init-db` flag. The application will create the
required tables and then exit without starting the simulation.

The configuration now includes a `tick_threshold` option controlling how many
ticks a node must receive in a single timestep before it can fire. This value
defaults to `1` and can be overridden via CLI or the Parameters window in the
GUI.

Tick energy can dissipate between scheduling and evaluation. The
`tick_decay_factor` parameter specifies how much stored tick energy decays per
tick. A value of `1.0` disables decay while values below `1.0` gradually reduce
the influence of older ticks. This setting is available via CLI and the GUI.

Nodes also observe a **refractory period** after firing.  The global
`refractory_period` setting determines how many ticks a node must wait before it
may emit again, preventing rapid oscillation.  This value is applied when nodes
are created unless a specific period is provided in the graph file and can be
adjusted through the CLI or GUI.

Edges can optionally vary their propagation strength using the
`edge_weight_range` setting. Each edge is assigned a random weight within this
range when the graph loads. The weight scales the delay returned by
`Edge.adjusted_delay` and inversely affects attenuation, allowing the network to
model non-uniform distances or resistance.

To model limited causal bandwidth you can restrict how many nodes may fire on a
single tick. Set `total_max_concurrent_firings` for a global limit or
`max_concurrent_firings_per_cluster` to cap activity within each detected
cluster. A value of `0` disables these limits. Both parameters are configurable
via CLI flags or the Parameters window in the GUI.

Cluster detection and bridge management are computationally heavy. The
`cluster_interval` setting controls how often these operations run (default
every 10 ticks). Node evaluation, meta-node updates and metric logging also
run on this interval to reduce overhead.
Spatial queries are cached for the duration of each tick to avoid redundant
lookups when clustering and managing bridges.

The `propagation_control` section toggles node growth mechanisms. Set
`enable_sip` or `enable_csp` to `false` to disable Stability-Induced or
Collapse-Seeded Propagation. These options are also exposed as CLI flags and can
be modified in the Parameters window.
## Graph format


Graphs are defined by a JSON file with `nodes`, `edges`, optional `bridges`, `tick_sources`, `observers` and `meta_nodes`. Each node defines its position, frequency and thresholds. Edges specify delays and attenuation. Tick sources seed periodic activity and observers describe which metrics to record. Meta nodes group related nodes under additional constraints.

### Meta nodes

Configured meta nodes are declared under the `meta_nodes` key and can enforce constraints such as `phase_lock`, `shared_tick_input` or `coherence_tie`. Emergent meta nodes are discovered at runtime when the engine observes naturally synchronized clusters. Only configured meta nodes modify behaviour; emergent ones are logged for analysis.

Observers include optional `x` and `y` fields storing their location on the canvas.

Example:
```json

{
  "nodes": {
    "A": {
      "x": 100,
      "y": 100,
      "frequency": 1.0,
      "refractory_period": 0.5,
      "base_threshold": 0.1,
      "phase": 0.0,
      "origin_type": "seed",
      "generation_tick": 0,
      "parent_ids": [],
      "goals": {}
    },
    "B": {
      "x": 100,
      "y": 200,
      "frequency": 1.0,
      "refractory_period": 0.5,
      "base_threshold": 0.1,
      "phase": 0.0,
      "origin_type": "derived",
      "generation_tick": 0,
      "parent_ids": [ "A" ]
    }
  },
  "edges": [
    {
      "from": "A",
      "to": "B",
      "attenuation": 1.0,
      "density": 0.0,
      "delay": 1,
      "phase_shift": 0.0
    }
  ],
  "bridges": [],
  "tick_sources": [
    {
      "node_id": "A",
      "tick_interval": 2,
      "phase": 0.0
    }
  ],
  "observers": [
    {
      "id": "OBS",
      "monitors": [ "collapse", "law_wave", "region" ],
      "frequency": 1
    }
  ],
  "meta_nodes": {
    "MN1": {
      "members": ["A", "B"],
      "constraints": {"phase_lock": {"tolerance": 0.1}},
      "type": "Configured",
      "collapsed": false,
      "x": 0.0,
      "y": 0.0
    }
  }
  }
```


## Running the simulation

1. Install the dependencies (`pyside6` is required for the GUI).
   An X11-compatible display is needed to create the window. If running on a
   headless server consider using a virtual frame buffer such as Xvfb.
   The dashboard automatically designates the *Causal Graph* window as the
   primary viewport so all windows remain interactive.
2. Launch the dashboard or run headless:

```bash
python -m Causal_Web.main            # with GUI
python -m Causal_Web.main --no-gui   # headless run
```

For automated runs the `headless` configuration flag can be enabled to suppress
observer output and intermediate logs.

Use the on-screen controls to start, pause/resume or stop the simulation and adjust the tick rate. A text box next to the tick rate slider displays the current value and allows direct entry&mdash;when focus leaves the field the slider synchronises to match. A tick counter shows the current tick and a **Tick Limit** field determines how many ticks to run. These inputs now reside in the **Control Panel** window which is docked to the top left. Windows can be freely resized and the graph view will scroll if its contents exceed the available space. Window resizing is now handled more robustly to avoid occasional freezes. As the simulation runs, a number of JSON log files are produced inside `Causal_Web/output`.
The main window now refreshes automatically as nodes and edges change during the run.
Use the **File** menu to load, save or start a new graph. Editing actions
include **Edit Graph...**, **Undo** and **Redo** in the **Edit** menu.
The **Graph View** dock now embeds a small toolbar offering **Add Node**,
**Add Connection**, **Add Observer**, **Auto Layout** and a **Load Graph** button for quick access.
The **Auto Layout** action still arranges nodes using a spring layout.
When you press **Start Simulation** the current graph is saved and also copied
to `input/graph.json`. A new run directory is created via `Config.new_run()` and
the graph file is copied into the run's `input/` folder. This preserves the
exact input used for each run. Any unsaved edits in the **Graph View** are
applied automatically so the main window reflects the latest changes when the
simulation begins.
These actions operate on the `graph.json` format and update the shared in-memory model.
The dashboard also includes a **Graph View** tab which renders the loaded graph and displays
basic information for the currently selected node. The **Graph View** window
now includes **Add Node**, **Add Connection**, **Add Observer**, **Auto Layout** and **Load Graph**
tools for building and applying the graph.
Selecting a node shows a docked panel where its attributes can be edited. The panel now includes
the node's initial **phase** and an optional **Tick Source** section for emitting periodic ticks.
If the panel remains visible while dragging a node its ``x`` and ``y`` values update live.
When two nodes are chosen for a new connection a connection panel allows its type and
parameters to be configured.  Fields in this window now change depending on whether the
connection is an **Edge** or **Bridge**.
Both panels include an **Apply** button to commit any changes.
Edges in the Graph View are selectable and will open the connection panel for
editing when clicked.
Nodes can be repositioned directly in the **Graph View** by dragging them with
the mouse. Interaction is handled by :class:`CanvasWidget`, a reusable
``QGraphicsView`` subclass that supports selection, dragging, zooming and panning.
You can pan the scene by dragging with the left mouse button over empty space or
using the middle mouse button.
Dragging begins on mouse press for smooth movement and remains responsive after
resizing the window. Debug messages are printed to the console whenever nodes
are clicked or dragged.
Edges connected to a moving node are redrawn in real time so the relationships stay visible while dragging.
Lines from meta nodes and observers are now drawn using a shared helper for consistent dashed styling. The context menu uses a dispatch map for clearer action handling.
Once the drag ends the node's new ``(x, y)`` coordinates are saved back to the
graph model automatically. Meta nodes and observers store their updated
positions in the same way.
Connections from observers and meta nodes now appear immediately after applying
changes, without needing to drag the items first.
Observer and meta node panels correctly remember their targeted nodes when reopened.
The Graph View window now resizes correctly, keeping graph elements interactive.
A resize handler bug that halted GUI updates after resizing has been fixed.
Dragging connections between nodes now works again across PySide6 versions.

Rendering is now event driven so the canvas only updates when the graph changes, greatly reducing idle CPU usage.
Graph editing panels are now docked within the Graph View window. They close automatically when the view is hidden and prompt about unapplied changes before closing. Panels minimize when they lose focus and restore when clicked again, warning about unsaved edits if you switch to a different object. The Graph View also warns when in-memory edits have not been saved to ``graph.json``.

### Analysing the output

Once a simulation has finished you can interpret the logs via:

```bash
python -m Causal_Web.engine.log_interpreter
```

This command loads the logs and generates several summary files:

- **`interpretation_log.json`** – aggregated metrics such as coherence ranges and collapse events.
- **`interpretation_summary.txt`** – human readable summary of the above data.
- **`causal_explanations.json`** and **`causal_summary.txt`** – explanation events and a narrative produced by `CausalAnalyst`.
- **`explanation_graph.json`** – causal chains expressed as a graph for visualisation.
- **`causal_timeline.json`** – ordered timeline of notable events.

## Output logs
Simulation results are organised under `output/` which now contains separate
directories for each run. A new run directory is created via
`Config.new_run()` and has the form `runs/<timestamp>__<slug>`. The current
`graph.json` and `config.json` files are copied into each run's `input/`
subdirectory so every run has a frozen copy of its inputs. Basic metadata
about the run is inserted into the PostgreSQL `runs` table automatically. The
Timestamps are generated in UTC to avoid timezone discrepancies.
location of `runs/` and other output folders can be customised using the
`paths` section in `input/config.json`.
Logging for each file can be enabled or disabled individually using
**Settings > Log Files...** in the GUI or via the `log_files` section of the
configuration file.
All log entries are buffered in memory and flushed periodically to minimise
disk writes. The frequency of metric logging is controlled by the
`log_interval` setting which defaults to `1` tick. This value can be adjusted
in the **Log Files** window.
Each record now conforms to Pydantic models defined in
`engine/logging_models.py`, ensuring consistent structure across files and
simplifying downstream analysis.

- `boundary_interaction_log.json` – interactions with void or boundary nodes.
- `bridge_decay_log.json` – gradual weakening of inactive bridges.
- `bridge_dynamics_log.json` – state changes for each bridge.
- `bridge_reformation_log.json` – bridges reforming after rupture.
- `bridge_rupture_log.json` – details of bridge failures.
- `bridge_state_log.json` – snapshot of all bridge states per tick.
- `classicalization_map.json` – which nodes are collapsed each tick.
- `cluster_influence_matrix.json` – edge counts between node regions.
- `cluster_log.json` – hierarchical clustering results.
- `coherence_log.json` – node coherence values.
- `coherence_velocity_log.json` – change in coherence between ticks.
- `collapse_chain_log.json` – propagation chains triggered by collapse.
- `collapse_front_log.json` – first collapse tick for each node.
- `connectivity_log.json` – number of links per node at load time.
- `curvature_log.json` – delay adjustments from law-wave curvature.
- `curvature_map.json` – aggregated curvature grid for visualisation.
- `decoherence_log.json` – node decoherence levels.
- `event_log.json` – high level events such as bridge ruptures.
- `global_diagnostics.json` – overall stability metrics.
- `inspection_log.json` – superposition inspection summary.
- `interference_log.json` – interference classification per node.
- `law_drift_log.json` – refractory period adjustments.
- `law_wave_log.json` – node law-wave frequencies.
- `stable_frequency_log.json` – nodes with converged law-wave frequency values.
- `layer_transition_log.json` – tick transitions between LCCM layers.
- `layer_transition_events.json` – counts of layer transitions per node.
- `magnitude_failure_log.json` – ticks rejected for low magnitude.
- `meta_node_tick_log.json` – ticks emitted by meta nodes.
- `node_emergence_log.json` – new nodes created via propagation.
- `node_state_log.json` – node type, credit and debt metrics.
- `node_state_map.json` – transitions between node types.
- `observer_disagreement_log.json` – difference between observers and reality.
- `observer_perceived_field.json` – tick counts inferred by observers.
- `propagation_failure_log.json` – reasons tick propagation failed.
- `refraction_log.json` – rerouted ticks through alternative paths.
- `regional_pressure_map.json` – decoherence pressure per region.
- `should_tick_log.json` – results of tick emission checks.
- `simulation_state.json` – current tick and pause state.
- `structural_growth_log.json` – summary of SIP/CSP growth each tick.
- `tick_delivery_log.json` – incoming tick phases for each node.
- `tick_density_map.json` – interference density map.
- `tick_drop_log.json` – ticks discarded before firing.
- `tick_emission_log.json` – ticks emitted by nodes.
- `tick_evaluation_log.json` – evaluation results for each potential tick.
- `tick_propagation_log.json` – ticks travelling across edges.
- `tick_seed_log.json` – activity injected by the seeder.
- `tick_trace.json` – final graph including tick history.
- `void_node_map.json` – isolated nodes with no connections.
- `manifest.json` – run manifest written by `bundle_run.py`.
- `interpretation_log.json` and `interpretation_summary.txt` – aggregated metrics.
- `causal_explanations.json` and `causal_summary.txt` – textual explanation of causes.
- `explanation_graph.json` – causal chains in graph form.
- `causal_chains.json` and `causal_timeline.json` – ordered causal events.
- `cwt_console.txt` – copy of console output.

### Log file fields
The following lists describe the JSON keys recorded in each output file.

#### `boundary_interaction_log.json`
- `tick` – simulation tick of the event.
- `origin` – node that emitted the triggering tick.
- `node`/`void` – identifier of the boundary or void node affected.

#### `bridge_decay_log.json`
- `tick` – tick when decay occurred.
- `bridge` – bridge identifier.
- `strength` – remaining strength after decay.
- `duration` – ticks since the bridge was last active.

#### `bridge_dynamics_log.json`
- `bridge_id` – unique bridge id.
- `source` – source node id.
- `target` – target node id.
- `event` – state change name.
- `tick` – tick of the event.
- `seeded` – whether created by the initial graph.
- `conditions` – optional context such as phase difference.

#### `bridge_reformation_log.json`
- `tick` – tick when the bridge reformed.
- `bridge` – id of the bridge.
- `coherence` – average coherence at reformation.

#### `bridge_rupture_log.json`
- `tick` – tick of the rupture.
- `bridge` – bridge id.
- `source` – originating node id.
- `target` – target node id.
- `reason` – rupture cause.
- `coherence` – coherence value at failure.
- `fatigue` – accumulated fatigue level.

#### `bridge_state_log.json`
- keyed by tick with nested bridge entries:
  - `active` – whether the bridge is enabled.
  - `last_activation` – most recent activation tick.
  - `last_rupture_tick` – tick of the last rupture or `null`.
  - `last_reform_tick` – tick of the last recovery or `null`.
  - `coherence_at_reform` – coherence when last reformed.
  - `trust_score` – current trust value.
  - `reinforcement` – reinforcement streak count.

#### `classicalization_map.json`
- keyed by tick with `{node: bool}` indicating if a node is classicalised.

#### `cluster_influence_matrix.json`
- `{ "regionA->regionB": count }` mapping between regions based on edges.

#### `cluster_log.json`
- keyed by tick with hierarchical cluster assignments per level.

#### `coherence_log.json`
- keyed by tick with `{node: coherence}` values.

#### `coherence_velocity_log.json`
- keyed by tick with `{node: delta}` change since the previous tick.

#### `collapse_chain_log.json`
- `tick` – tick when propagation occurred.
- `source` – collapsing node.
- `collapsed` – list of nodes with depth information.
- `collapsed_entity` – id of the initiating node.
- `children_spawned` – ids of any spawned nodes.

#### `collapse_front_log.json`
- either `{tick, node, event}` for collapse start events or
  `{tick, source, chain}` describing propagation depth.

#### `connectivity_log.json`
- `{node: {edges_out, edges_in, bridges, total}}` summary at load time.

#### `curvature_log.json`
- keyed by tick with per-edge delay adjustments:
  - `delta_f` – frequency difference.
  - `curved_delay` – resulting delay value.

#### `curvature_map.json`
- list of `{tick, edges:[{source, target, delay}]}` for visualisation.

#### `decoherence_log.json`
- keyed by tick with `{node: decoherence}` values.

#### `event_log.json`
- records bridge events with fields:
  - `tick`, `event_type`, `bridge_id`, `source`, `target`,
    `coherence_at_event`.

#### `global_diagnostics.json`
- run-level metrics:
  - `coherence_stability_score`, `entropy_delta`,
    `collapse_resilience_index`, `network_adaptivity_index`.

#### `inspection_log.json`
- list of superposition inspections with keys:
  - `tick`, `node`, `contributors`, `interference_result`,
    `collapsed`, `bridge_status`.

#### `interference_log.json`
- keyed by tick with `{node: superposition_count}`.

#### `law_drift_log.json`
- `tick`, `node` and the updated `new_refractory_period`.

#### `law_wave_log.json`
- entries either keyed by tick with `{node: frequency}` or
  `{tick, origin, affected}` when a law wave is emitted.

#### `stable_frequency_log.json`
- keyed by tick mapping nodes to stabilised law-wave frequencies.

#### `layer_transition_log.json`
- `tick`, `node`, source `from` layer, destination `to` layer and `trace_id`.

#### `layer_transition_events.json`
- counts of layer transitions summarised as
  `{node: {layer: count}}`.

#### `magnitude_failure_log.json`
- `tick`, `node`, `magnitude`, `threshold` and number of `phases` when a tick fails.

#### `meta_node_tick_log.json`
- keyed by tick with `{meta_id: [member_nodes]}` entries.

#### `node_emergence_log.json`
- new node details including `id`, `tick`, `parents`,
  `origin_type`, `generation_tick`, `sigma_phi` and `phase_confidence_index`.

#### `node_state_log.json`
- keyed by tick containing:
  - `type` – node type per id.
  - `credit` – coherence credit values.
  - `debt` – decoherence debt values.

#### `node_state_map.json`
- records transitions with `node`, `from` and `to` state identifiers.

#### `observer_disagreement_log.json`
- `tick`, `observer` and `diff` between observation and reality.

#### `observer_perceived_field.json`
- `tick`, `observer` and the inferred `state` per node.

#### `propagation_failure_log.json`
- heterogeneous records describing why propagation failed. Common
  fields include `tick`, `node` or `parent`, failure `type` and `reason`.

#### `refraction_log.json`
- rerouting information containing `tick` and either
  `recursion_from` or `from`/`via`/`to` paths.

#### `regional_pressure_map.json`
- mapping of regions to averaged decoherence pressure.

#### `should_tick_log.json`
- decisions from `should_tick` with `tick`, `node` and `reason`.

#### `simulation_state.json`
- `paused`, `stopped`, `current_tick` and optional `graph_snapshot` path.

#### `structural_growth_log.json`
- per tick record of node counts and SIP/CSP success/failure totals.

#### `tick_delivery_log.json`
- `source`, `node_id`, `tick_time` and `stored_phase` for incoming ticks.

#### `tick_density_map.json`
- keyed by tick mirroring `interference_log` densities.

#### `tick_drop_log.json`
- dropped tick info: `tick`, `node`, `reason`, `coherence`, `node_type`.

#### `tick_emission_log.json`
- emitted ticks with `node_id`, `tick_time` and `phase`.

#### `tick_evaluation_log.json`
- evaluation outcome fields:
  `tick`, `node`, `coherence`, `threshold`, `refractory`, `fired`, `reason`.

#### `tick_propagation_log.json`
- `source`, `target`, `tick_time`, `arrival_time` and propagated `phase`.

#### `tick_seed_log.json`
- seeder actions recording `tick`, `node`, `phase`, `strategy`,
  `coherence`, `threshold`, `success` and optional `failure_reason`.

#### `tick_trace.json`
- complete graph snapshot including nodes, edges, bridges and tick history.

#### `void_node_map.json`
- list of node ids with no connections.

#### `manifest.json`
- summary produced by `bundle_run.py` containing run metadata such as
  `run_id`, `timestamp`, tick counts, collapse statistics and diagnostics.

#### `interpretation_log.json`
- aggregated metrics produced by the interpreter. Keys may include
  `curvature`, `collapse`, `coherence`, `law_wave`, `decoherence`,
  `layer_transitions`, `rerouting`, `node_state_transitions`, `clusters`,
  `bridges`, `law_drift`, `meta_nodes`, `tick_counts`, `layer_summary`,
  `inspection_events` and `console`.

#### `causal_explanations.json`
- explanation events with `tick_range`, `affected_nodes`,
  `origin` rule and textual `explanation`.

#### `causal_summary.txt`
- human readable narrative of the explanation events.

#### `explanation_graph.json`
- DAG describing causal chains with nodes (`id`, `tick`, `type`, `node`, `description`)
  and edges (`source`, `target`, `label`).

#### `causal_chains.json`
- list of causal chains each containing `root_event`, `chain` steps
  and overall `confidence`.

#### `causal_timeline.json`
- ordered list of `{tick, events}` where each event notes a `type` and involved nodes.

#### `cwt_console.txt`
- captured console output from the run.

The raw logs (`tick_trace.json`, `coherence_log.json`, `event_log.json`, etc.) remain in `output/` for detailed inspection. For convenience, running `bundle_run.py` packages the important files with a manifest describing the run.

## Testing

Basic unit tests cover some of the utility methods inside the engine. They verify
phase interference detection, cluster formation logic, edge delay adjustment and
several node behaviours. Run them with:

```bash
pytest
```

## Service objects

Large functions have been decomposed into reusable services. The `NodeTickService`
encapsulates the tick emission lifecycle while `GraphLoadService` handles JSON
graph loading. Metric collection is delegated to `NodeMetricsService` which
replaced the bulky `log_metrics_per_tick` function. The `NodeTickDecisionService`
isolates the tick decision logic from `Node.should_tick`. Serialization and
`EdgePropagationService` manages edge traversal. Serialization and
narrative generation are now handled by `GraphSerializationService` and
`NarrativeGeneratorService`. GUI setup moved to `ToolbarBuildService` and
`NodePanelSetupService`. All
services now live in `Causal_Web/engine/services/` or the GUI package.
Recent refactors introduced `ConnectionDisplayService` for showing existing
links, `GlobalDiagnosticsService` for exporting run metrics and
`SIPRecombinationService` for recombination-based spawning. A lightweight
`LoggingMixin` now centralises JSON logging for classes like `Node` and `Bridge`.
`OutputDirMixin` adds a common `_path` helper used by the log interpreter and
causal analyst. `PathLoggingMixin` offers direct logging to arbitrary file paths
for utilities like the tick seeder.

### Identified long functions

- `main.py:main` – 66 lines
- `graph/model.py:add_connection` – 56 lines
- `engine/tick_engine/evaluator.py:_process_csp_seeds` – 83 lines
- `engine/services/sim_services.py:NodeMetricsService.log_metrics` – 45 lines
- `engine/tick_engine/core.py:simulation_loop` – 98 lines
- `engine/tick_engine/core.py:SimulationRunner.run` – 93 lines
- `engine/causal_analyst.py:infer_causal_chains` – 53 lines
- `engine/bridge.py:apply` – 125 lines
- `engine/node.py:__init__` – 97 lines
- `engine/node.py:should_tick` – 111 lines
- `engine/node.py:apply_tick` – 143 lines
- `engine/graph.py:detect_clusters` – 55 lines
- `engine/graph.py:load_from_file` – 138 lines
- `engine/explanation_rules.py:_match_emergence_events` – 51 lines
- `gui_pyside/toolbar_builder.py:__init__` – 57 lines
- `gui_pyside/toolbar_builder.py:__init__` – 101 lines
- `gui_pyside/toolbar_builder.py:commit` – 99 lines
- `gui_pyside/toolbar_builder.py:__init__` – 68 lines

