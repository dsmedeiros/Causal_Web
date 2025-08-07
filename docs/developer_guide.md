# Developer Guide

This project follows the Single Responsibility Principle and decomposes complex tasks into small service classes. Key services include:

- **NodeInitializationService** – sets up node instances with runtime state and metadata.
- **NodeTickService** – manages the tick emission lifecycle.
- **GraphLoadService** – loads graphs from JSON files.
- **NodeMetricsService** – collects metrics per tick.
- **NodeTickDecisionService** – determines whether a node should emit.
- **EdgePropagationService** – handles propagation across edges.
- **GraphSerializationService** – writes graph snapshots to disk.
- **NarrativeGeneratorService** – produces textual explanations from logs.
- **ConnectionDisplayService** – visualises existing links in the GUI.
- **GlobalDiagnosticsService** – exports run metrics.
- **SIPRecombinationService** – manages recombination-based spawning.
- **EntanglementService** – collapses ε-linked partners and manages entangled pairs.

Logging helpers such as `LoggingMixin`, `OutputDirMixin` and `PathLoggingMixin` provide common functionality across the engine. Services reside under `Causal_Web/engine/services/` or within the GUI package.

For coding conventions and testing instructions see [AGENTS.md](../AGENTS.md). Run `pytest` to execute the unit tests and format Python files with `black` before submitting changes.

## Ray Cluster

Distributed execution uses a lightweight wrapper around [Ray](https://www.ray.io). A cluster can be configured programmatically via `ray_cluster.init_cluster()` or from the CLI using `--ray-init`:

```bash
python -m Causal_Web.main --ray-init '{"num_cpus":4}'
```

The JSON payload is forwarded directly to `ray.init` allowing resource limits or addresses to be specified.
