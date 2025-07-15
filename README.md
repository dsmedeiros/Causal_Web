# Causal Web

This project contains a small simulation engine and GUI for experimenting with causal
graphs. It is written in Python and uses [Dear PyGui](https://github.com/hoffstadt/DearPyGui)
for the graphical interface.

## Overview

The code models nodes connected by directed edges. Each node can emit "ticks" with a
phase value that propagates through the graph. Edges attenuate and delay phases before
they arrive at downstream nodes. Nodes decide whether to fire based on the combined
incoming phases and their current threshold.

Phase 4 introduces memory, intention and rhythmic forcing. Nodes now keep sliding
histories of recent coherence and learn trust scores for their neighbours. Bridges
track ruptures and reinforcement streaks while observers attempt to infer unseen
events. Optional global modulation fields can jitter phases or thresholds for all
nodes.

The main components are:

- **`engine/graph.py`** – Defines `CausalGraph` which stores nodes and edges. Graphs can be
  loaded from or saved to JSON files.
- **`engine/node.py`** – Implements `Node` and `Edge` classes. Nodes keep a history of
ticks, queue incoming phases and apply refractory periods before firing.
- **`engine/tick_engine.py`** – Runs the discrete tick simulation. It emits ticks from
nodes, propagates phases along edges and records the results to
`output/tick_trace.json`.
- **`gui/dashboard.py`** – Launches a simple dashboard built with Dear PyGui. Controls allow
starting, pausing and adjusting the tick rate while visualising the graph in real time.
- **`main.py`** – Entry point that simply launches the dashboard.

Graphs are expected to be defined in `input/graph.json` inside the package. The
code now resolves this path relative to the package location so the module can
be executed from any working directory. Simulation output is written to the
`output` folder located next to the code.

## Running

1. Install dependencies (e.g. `dearpygui`).
2. Run the application:

```bash
python -m Causal_Web.main
```

A window will appear showing the causal graph and controls for the simulation.

