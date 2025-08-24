# Developer Guide

This project follows the Single Responsibility Principle and decomposes complex tasks into small modules. The event/PQ kernel described in [theory.md](../theory.md) §11 is the canonical execution model.

Notable modules:

- `engine/engine_v2/adapter.py`
- `engine/engine_v2/scheduler.py`
- `engine/engine_v2/lccm/`
- `engine/engine_v2/rho_delay.py`
- `engine/engine_v2/epairs.py`
- `engine/stream/server.py`
- `ui_new/ipc/Client.py`
- `ui_new/graph/GraphView.py`
- `experiments/runner.py`
- `experiments/ga.py`
- `experiments/artifacts.py`
- etc.

## Dev loop

```bash
pre-commit install
pre-commit run --files <file1> <file2>
ruff . && black . && mypy .
pytest -q
cw run --no-gui
cw run
```

## Ray Cluster

Distributed execution uses a lightweight wrapper around [Ray](https://www.ray.io). Configure a cluster programmatically via `ray_cluster.init_cluster()` before starting the simulation:

```bash
python -c 'from Causal_Web.engine.backend import ray_cluster; ray_cluster.init_cluster(num_cpus=4)'
```

The JSON payload is forwarded directly to `ray.init` allowing resource limits or addresses to be specified.

## Troubleshooting

- **Token mismatch** – restart the engine and GUI to refresh the token.
- **Single-client limit** – only one GUI may connect; close any extra instances.
- **Low FPS** – zoom out; labels auto-hide; disable AA at far zoom.
- **Invariant failures** – inspect `result.json` in the run directory.
