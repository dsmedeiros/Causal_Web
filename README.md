# Causal Web

Causal Web is a simulation engine and GUI for experimenting with causal graphs. Nodes emit ticks that propagate through edges with delay and attenuation while observers infer hidden state from the resulting activity. The project is written in Python and uses [PySide6](https://doc.qt.io/qtforpython/) for the graphical interface.

Ticks carry both phase and amplitude. Their influence on interference and coherence is weighted by amplitude and each tick records the local `generation_tick` at which it was emitted.

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

## Installation
Clone the repository and install the packages listed in `requirements.txt`. The GUI requires PySide6 and an X11 compatible display.

## Usage
Graphs are stored as JSON files under `input/`. Each file defines `nodes`, `edges`, optional `bridges`, `tick_sources`, `observers` and `meta_nodes`. See [docs/graph_format.md](docs/graph_format.md) for the complete schema and an example.

The GUI allows interactive editing of graphs. Drag nodes to reposition them and use the toolbar to add connections or observers. After editing, click **Apply Changes** in the Graph View to update the simulation and save the file. Details on all GUI actions are provided in [docs/gui_usage.md](docs/gui_usage.md).
Nodes can optionally enable self-connections via a checkbox in the node panel. When enabled, dragging from a node back onto itself creates a curved edge.
Bridges now support an `Entanglement Enabled` option. When selected, the bridge
is tagged with an `entangled_id` used by observers to generate deterministic
measurement outcomes for Bell-type experiments.
Observers can enable a *Detector Mode* that records a binary outcome whenever a
tick from an entangled bridge is detected.
These detector events are additionally written to `entangled_log.jsonl` for
Bell inequality analysis.

Runs produce a set of JSON logs in `output/`. The script `bundle_run.py` can be used after a simulation to archive the results. Full descriptions of each log file and their fields are available in [docs/log_schemas.md](docs/log_schemas.md).

## Configuration
Runtime parameters are loaded from `input/config.json`. Any value can be overridden with CLI flags using dot notation for nested keys, for example:
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

The `density_calc` option controls how edge density is computed. Set one of:

- `local_tick_saturation` (default) – density increases with recent traffic
- `modular-<mode>` – use a registered modular function (`tick_history`,
  `node_coherence`, `spatial_field`, `bridge_saturation`)
- `manual_overlay` – sample values from an external overlay file defined by
  `density_overlay_file`.

`density_calc` can also be specified via `--density-calc` on the command line.

## Output Logs
Each run creates a timestamped directory under `output/runs` containing the graph, configuration and logs. Logging can be enabled or disabled via the GUI **Log Files** window or the `log_files` section of `config.json`. In `config.json` the keys are the categories (`tick`, `phenomena`, `event`) containing individual label flags. The `logging_mode` option selects which categories are written: `diagnostic` (all logs), `tick`, `phenomena` and `events`.
Logs are consolidated by category into `ticks_log.jsonl`, `phenomena_log.jsonl` and `events_log.jsonl`.
Individual files can still be toggled via `log_files` for advanced filtering.
Entangled tick metadata and detector outcomes are stored separately in
`entangled_log.jsonl`.
Law-wave propagation events now appear as `law_wave_event` records in the `events` log.
Per-tick law-wave frequencies are still written under the `law_wave_log` label in `ticks_log.jsonl`.
The interpreter provides `records_for_tick()` and `assemble_timeline()` helpers to query these consolidated logs.
The ingestion service also consumes these unified files, routing records to database tables based on their `label` or `event_type`.

### Phase smoothing

The `smooth_phase` option applies exponential decay to each node's internal oscillator phase. Enable it from the GUI control panel or set `"smooth_phase": true` in `input/config.json`.

### Propagation control

Check boxes on the control panel allow SIP budding, SIP recombination and collapse seeded propagation to be disabled independently. The `propagation_control` section of `input/config.json` contains `enable_sip_child`, `enable_sip_recomb` and `enable_csp` flags. These can also be toggled via the CLI using `--disable-sip-child`, `--disable-sip-recomb` and `--disable-csp`.

## Contributing
Unit tests live under `tests/` and can be run with `pytest`. Coding guidelines and packaging instructions are documented in [AGENTS.md](AGENTS.md) and [docs/developer_guide.md](docs/developer_guide.md).
