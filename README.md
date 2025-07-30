# Causal Web

Causal Web is a simulation engine and GUI for experimenting with causal graphs. Nodes emit ticks that propagate through edges with delay and attenuation while observers infer hidden state from the resulting activity. The project is written in Python and uses [PySide6](https://doc.qt.io/qtforpython/) for the graphical interface.

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

Runs produce a set of JSON logs in `output/`. The script `bundle_run.py` can be used after a simulation to archive the results. Full descriptions of each log file and their fields are available in [docs/log_schemas.md](docs/log_schemas.md).

## Configuration
Runtime parameters are loaded from `input/config.json`. Any value can be overridden with CLI flags using dot notation for nested keys, for example:
```bash
python -m Causal_Web.main --no-gui --max_ticks 20
```
Use `--init-db` to create PostgreSQL tables defined in the configuration and exit.

## Output Logs
Each run creates a timestamped directory under `output/runs` containing the graph, configuration and logs. Logging can be enabled or disabled via the GUI **Log Files** window or the `log_files` section of `config.json`. The `logging_mode` option selects which categories are written: `diagnostic` (all logs), `tick`, `phenomena` and `events`.
Logs are consolidated by category into `ticks_log.jsonl`, `phenomena_log.jsonl` and `events_log.jsonl`.
Individual files can still be toggled via `log_files` for advanced filtering.
Law-wave propagation events now appear as `law_wave_event` records in the `events` log.
Per-tick law-wave frequencies are still written under the `law_wave_log` label in `ticks_log.jsonl`.
The interpreter provides `records_for_tick()` and `assemble_timeline()` helpers to query these consolidated logs.
The ingestion service also consumes these unified files, routing records to database tables based on their `label` or `event_type`.

## Contributing
Unit tests live under `tests/` and can be run with `pytest`. Coding guidelines and packaging instructions are documented in [AGENTS.md](AGENTS.md) and [docs/developer_guide.md](docs/developer_guide.md).
