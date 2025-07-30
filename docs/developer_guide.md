# Developer Guide

This project follows the Single Responsibility Principle and decomposes complex tasks into small service classes. Key services include:

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

Logging helpers such as `LoggingMixin`, `OutputDirMixin` and `PathLoggingMixin` provide common functionality across the engine. Services reside under `Causal_Web/engine/services/` or within the GUI package.

For coding conventions and testing instructions see [AGENTS.md](../AGENTS.md). Run `pytest` to execute the unit tests and format Python files with `black` before submitting changes.
