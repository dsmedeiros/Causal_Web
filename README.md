# Causal Web

This project contains a small simulation engine and GUI for experimenting with causal graphs. It is written in Python and uses [Dear PyGui](https://github.com/hoffstadt/DearPyGui) for the graphical interface.

## Overview

The engine models a directed network of nodes. Each node maintains its own oscillator phase and can emit "ticks" that travel along edges with delay and attenuation. Nodes accumulate incoming phases and fire when they pass a threshold, scheduling more ticks. Bridges create additional links whose strength can change over time. Observers watch the network and attempt to infer hidden state.

Key modules include:

- **`engine/graph.py`** – container for nodes, edges and bridges. Graphs can be loaded from or written to JSON files.
- **`engine/node.py`** – implementation of `Node`, `Edge` and related logic.
- **`engine/bridge.py`** – manages dynamic bridges between nodes.
- **`engine/tick_engine.py`** – drives the discrete simulation and records metrics under `output/`.
- **`engine/log_interpreter.py`** – parses the generated logs and aggregates statistics.
- **`engine/causal_analyst.py`** – infers causal chains and produces explanation files.
- **`gui/dashboard.py`** – Dear PyGui dashboard for interactive runs.
- **`main.py`** – simple entry point that launches the dashboard.

Graphs are stored in `input/graph.json` inside the package. Paths are resolved relative to the package so the module can be run from any working directory. All output is written next to the code in the `output` directory.

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

The raw logs (`tick_trace.json`, `coherence_log.json`, `event_log.json`, etc.) remain in `output/` for detailed inspection. For convenience, running `bundle_run.py` packages the important files with a manifest describing the run.
