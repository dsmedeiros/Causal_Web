# main.py

"""Entry point for launching the Causal Web simulation or GUI."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Any

from .config import Config, load_config


def _add_config_args(
    parser: argparse.ArgumentParser, data: dict[str, Any], prefix: str = ""
) -> None:
    """Recursively add CLI flags based on ``data`` keys."""
    for key, value in data.items():
        arg_name = f"--{prefix}{key}"
        dest = f"{prefix}{key}".replace(".", "_")
        if isinstance(value, dict):
            _add_config_args(parser, value, prefix=f"{key}.")
            continue
        arg_type = type(value)
        if isinstance(value, bool):
            parser.add_argument(arg_name, type=lambda x: x.lower() == "true", dest=dest)
        else:
            parser.add_argument(arg_name, type=arg_type, dest=dest)


def _apply_overrides(
    args: argparse.Namespace, data: dict[str, Any], prefix: str = ""
) -> None:
    """Apply CLI overrides back onto :class:`Config`."""
    for key, value in data.items():
        full = f"{prefix}{key}"
        dest = full.replace(".", "_")
        override = getattr(args, dest, None)
        if override is not None:
            parts = full.split(".")
            target = Config
            for part in parts[:-1]:
                target = getattr(target, part)
            if isinstance(target, dict):
                target[parts[-1]] = override
            else:
                setattr(target, parts[-1], override)
        elif isinstance(value, dict):
            _apply_overrides(args, value, prefix=f"{key}.")


@dataclass
class MainService:
    """Handle CLI parsing and runtime selection."""

    argv: list[str] | None = None

    def run(self) -> None:
        args, cfg = self._parse_args()
        _apply_overrides(args, cfg)
        if args.init_db:
            self._init_db(args.config)
            return
        if args.no_gui:
            self._run_headless()
        else:
            self._launch_gui()

    # ------------------------------------------------------------------
    def _parse_args(self) -> tuple[argparse.Namespace, dict[str, Any]]:
        initial = argparse.ArgumentParser(add_help=False)
        initial.add_argument(
            "--config",
            default=Config.input_path("config.json"),
            help="Path to JSON configuration file",
        )
        initial.add_argument(
            "--graph",
            default=Config.graph_file,
            help="Path to graph JSON file",
        )
        initial.add_argument(
            "--no-gui",
            action="store_true",
            help="Run simulation without launching the GUI",
        )
        initial.add_argument(
            "--init-db",
            action="store_true",
            help="Initialize PostgreSQL schema and exit.",
        )
        known, _ = initial.parse_known_args(self.argv)
        Config.graph_file = known.graph

        config_data: dict[str, Any] = {}
        if known.config and os.path.exists(known.config):
            with open(known.config) as f:
                config_data = json.load(f)
            Config.load_from_file(known.config)

        parser = argparse.ArgumentParser(
            parents=[initial], description="Run Causal Web simulation"
        )
        _add_config_args(parser, config_data)
        args = parser.parse_args(self.argv)
        Config.graph_file = args.graph
        return args, config_data

    # ------------------------------------------------------------------
    def _init_db(self, cfg_path: str) -> None:
        from .database import initialize_database

        load_config(cfg_path)
        initialize_database(Config.database)

    # ------------------------------------------------------------------
    def _run_headless(self) -> None:
        """Run the simulation without launching the GUI."""
        from .engine import tick_engine

        tick_engine.build_graph()
        with Config.state_lock:
            Config.is_running = True
        tick_engine.simulation_loop()
        limit = Config.max_ticks if Config.allow_tick_override else Config.tick_limit
        try:
            while True:
                with Config.state_lock:
                    running = Config.is_running
                    tick = Config.current_tick
                if limit and limit != -1 and tick >= limit:
                    tick_engine.stop_simulation()
                if not running:
                    break
                time.sleep(0.1)
        except KeyboardInterrupt:
            tick_engine.stop_simulation()

    # ------------------------------------------------------------------
    @staticmethod
    def _launch_gui() -> None:
        from .gui_pyside import launch

        launch()


def main() -> None:
    """Entry point for external callers."""
    MainService().run()


if __name__ == "__main__":
    main()
