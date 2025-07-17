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
- **`gui/dashboard.py`** – Dear PyGui dashboard for interactive runs.
- **`main.py`** – simple entry point that launches the dashboard.

Graphs are stored in `input/graph.json` inside the package. Paths are resolved relative to the package so the module can be run from any working directory. All output is written next to the code in the `output` directory.
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
2. Launch the dashboard:

```bash
python -m Causal_Web.main
```

Use the on-screen controls to start or pause the simulation and adjust the tick rate. As it runs, a number of JSON log files are produced inside `Causal_Web/output`.

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
The simulation writes many JSON files to `output/`.

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

The raw logs (`tick_trace.json`, `coherence_log.json`, `event_log.json`, etc.) remain in `output/` for detailed inspection. For convenience, running `bundle_run.py` packages the important files with a manifest describing the run.

## Testing

Basic unit tests cover some of the utility methods inside the engine. They verify
phase interference detection, cluster formation logic, edge delay adjustment and
several node behaviours. Run them with:

```bash
pytest
```
