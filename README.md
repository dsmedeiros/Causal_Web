# Causal Web

This project contains a small simulation engine and GUI for experimenting with causal
graphs. It is written in Python and uses [Dear PyGui](https://github.com/hoffstadt/DearPyGui)
for the graphical interface.

## Overview

The code models nodes connected by directed edges. Each node can emit "ticks" with a
phase value that propagates through the graph. Edges attenuate and delay phases before
they arrive at downstream nodes. Nodes decide whether to fire based on the combined
incoming phases and their current threshold.

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

Graphs are expected to be defined in `input/graph.json` which is loaded when the GUI
starts. Simulation output is written to the `output` directory when the run finishes.

## Running

1. Install dependencies (e.g. `dearpygui`).
2. Run the application:

```bash
python -m Causal_Web.main
```

A window will appear showing the causal graph and controls for the simulation.

