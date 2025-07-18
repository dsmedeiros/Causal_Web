# Causal Web

This project contains a small simulation engine and GUI for experimenting with causal graphs. It is written in Python and uses [Dear PyGui](https://github.com/hoffstadt/DearPyGui) for the graphical interface.

## Overview

The engine models a directed network of nodes. Each node maintains its own oscillator phase and can emit "ticks" that travel along edges with delay and attenuation. Nodes accumulate incoming phases and fire when they pass a threshold, scheduling more ticks. Bridges create additional links whose strength can change over time. Observers watch the network and attempt to infer hidden state.

Key modules include:

- **`engine/graph.py`** – container for nodes, edges and bridges. Graphs can be loaded from or written to JSON files.
- **`engine/node.py`** – implementation of `Node`, `Edge` and related logic.
- **`engine/bridge.py`** – manages dynamic bridges between nodes.
- **`engine/tick_engine.py`** – drives the discrete simulation and records metrics under `output/`.
- **`engine/tick_router.py`** – moves ticks through LCCM layers and logs transitions.
- **`engine/tick_seeder.py`** – seeds periodic ticks based on the configuration file.
- **`engine/log_interpreter.py`** – parses the generated logs and aggregates statistics.
- **`engine/causal_analyst.py`** – infers causal chains and produces explanation files.
- **`engine/meta_node.py`** – groups clusters of nodes into collapsed meta nodes.
- **`engine/node_manager.py`** – NumPy-backed container for bulk node updates.
- **`engine/observer.py`** – observers that infer hidden state from tick history.
- **`engine/logger.py`** – asynchronous writer used by many modules.
- **`engine/tick.py`** – defines :class:`Tick` and the reusable object pool.
- **`gui/dashboard.py`** – Dear PyGui dashboard for interactive runs.
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

Only keys matching attributes on `Causal_Web.config.Config` are applied. Nested
dictionaries merge with the existing values.

The configuration now includes a `tick_threshold` option controlling how many
ticks a node must receive in a single timestep before it can fire. This value
defaults to `1` and can be overridden via CLI or the Parameters window in the
GUI.

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
## Graph format

Graphs are defined by a JSON file with `nodes`, `edges`, optional `bridges`, `tick_sources` and `observers`. Each node defines its position, frequency and thresholds. Edges specify delays and attenuation. Tick sources seed periodic activity and observers describe which metrics to record.

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
  ]
}
```


## Running the simulation

1. Install the dependencies (`dearpygui` is required for the GUI).
   An X11-compatible display is needed to create the window. If running on a
   headless server consider using a virtual frame buffer such as Xvfb.
2. Launch the dashboard or run headless:

```bash
python -m Causal_Web.main            # with GUI
python -m Causal_Web.main --no-gui   # headless run
```

Use the on-screen controls to start or pause the simulation and adjust the tick rate. Windows can be freely resized and the graph view will scroll if its contents exceed the available space. As the simulation runs, a number of JSON log files are produced inside `Causal_Web/output`.

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
The simulation writes many JSON files to `output/`. Logging for each file can be
enabled or disabled individually using the **Logging** window in the GUI or via
the `log_files` section of `input/config.json`.

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
