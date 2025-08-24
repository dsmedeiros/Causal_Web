# main.py

"""Entry point for launching the Causal Web simulation or GUI."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

# Internal Config attributes that should not be exposed as CLI flags
from Causal_Web.config import Config, load_config


_PRIVATE_KEYS = {
    "base_dir",
    "input_dir",
    "config_file",
    "graph_file",
    "output_root",
    "runs_dir",
    "archive_dir",
    "analysis_dir",
    "ingest_dir",
    "output_dir",
    "profile_output",
    "state_lock",
    "current_tick",
    "is_running",
    "TICK_POOL_SIZE",
}


def _configure_logging() -> None:
    """Configure application logging and capture uncaught exceptions."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        filename="cw_gui.log",
        filemode="a",
    )

    def _log_excepthook(exc_type, exc, tb) -> None:
        logging.getLogger(__name__).exception(
            "Uncaught exception", exc_info=(exc_type, exc, tb)
        )

    sys.excepthook = _log_excepthook


def _add_config_args(
    parser: argparse.ArgumentParser, data: dict[str, Any], prefix: str = ""
) -> None:
    """Recursively add CLI flags based on ``data`` keys."""
    for key, value in data.items():
        if key in _PRIVATE_KEYS:
            continue
        arg_name = f"--{prefix}{key}"
        dest = f"{prefix}{key}".replace(".", "_")
        if isinstance(value, dict):
            _add_config_args(parser, value, prefix=f"{prefix}{key}.")
            continue
        arg_type = type(value)
        if dest == "backend":
            parser.add_argument(arg_name, choices=["cpu", "cupy"], dest=dest)
            continue
        if isinstance(value, bool):
            parser.add_argument(arg_name, type=lambda x: x.lower() == "true", dest=dest)
        else:
            parser.add_argument(arg_name, type=arg_type, dest=dest)


def _config_defaults() -> dict[str, Any]:
    """Return a dictionary of all attributes defined on :class:`Config`."""
    defaults: dict[str, Any] = {}
    for key, value in Config.__dict__.items():
        if key.startswith("_") or callable(value) or key in _PRIVATE_KEYS:
            continue
        defaults[key] = value
    return defaults


def _merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` returning a new dict."""
    result: dict[str, Any] = {}
    keys = set(base) | set(override)
    for key in keys:
        if isinstance(base.get(key), dict) and isinstance(override.get(key), dict):
            result[key] = _merge_configs(base[key], override[key])
        elif key in override:
            result[key] = override[key]
        else:
            result[key] = base[key]
    return result


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
        _configure_logging()
        args, cfg = self._parse_args()
        _apply_overrides(args, cfg)
        self._apply_log_overrides(args)
        Config.profile_output = getattr(args, "profile_output", None)
        if args.init_db:
            self._init_db(args.config)
            return
        if args.no_gui:
            self._run_headless()
        else:
            self._launch_gui(args)

    # ------------------------------------------------------------------
    @staticmethod
    def _apply_log_overrides(args: argparse.Namespace) -> None:
        """Update :class:`Config.log_files` based on CLI flags."""

        mappings = {
            "tick": (args.enable_tick, args.disable_tick),
            "phenomena": (args.enable_phenomena, args.disable_phenomena),
            "event": (args.enable_events, args.disable_events),
        }
        for cat, (en, dis) in mappings.items():
            cfg = Config.log_files.setdefault(cat, {})
            if en:
                for label in en.split(","):
                    label = label.strip()
                    if label:
                        cfg[label] = True
            if dis:
                for label in dis.split(","):
                    label = label.strip()
                    if label:
                        cfg[label] = False

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
            initial.set_defaults(graph=Config.graph_file)

        parser = argparse.ArgumentParser(
            parents=[initial], description="Run Causal Web simulation"
        )
        defaults = _merge_configs(_config_defaults(), config_data)
        _add_config_args(parser, defaults)
        parser.add_argument(
            "--profile",
            dest="profile_output",
            default=None,
            help="Write cProfile statistics to the given file",
        )
        parser.add_argument(
            "--enable-tick", default="", help="Comma-separated tick labels to enable"
        )
        parser.add_argument(
            "--disable-tick", default="", help="Comma-separated tick labels to disable"
        )
        parser.add_argument(
            "--enable-phenomena",
            default="",
            help="Comma-separated phenomena labels to enable",
        )
        parser.add_argument(
            "--disable-phenomena",
            default="",
            help="Comma-separated phenomena labels to disable",
        )
        parser.add_argument(
            "--enable-events", default="", help="Comma-separated event types to enable"
        )
        parser.add_argument(
            "--disable-events",
            default="",
            help="Comma-separated event types to disable",
        )
        parser.add_argument("--ws-url", default=None, help="WebSocket URL of engine")
        parser.add_argument("--ws-host", default=None, help="Engine host override")
        parser.add_argument(
            "--ws-port", type=int, default=None, help="Engine port override"
        )
        parser.add_argument("--token", default=None, help="Session token override")
        parser.add_argument(
            "--token-file", default=None, help="Path to session bundle JSON"
        )
        args = parser.parse_args(self.argv)
        Config.graph_file = args.graph
        return args, defaults

    # ------------------------------------------------------------------
    def _init_db(self, cfg_path: str) -> None:
        from Causal_Web.database import initialize_database

        load_config(cfg_path)
        initialize_database(Config.database)

    # ------------------------------------------------------------------
    def _run_headless(self) -> None:
        """Run the simulation without launching the GUI."""
        from Causal_Web.config import Config
        from Causal_Web.engine.engine_v2 import adapter as eng

        eng.build_graph()
        with Config.state_lock:
            Config.is_running = True
        eng.simulation_loop()
        limit = Config.max_ticks if Config.allow_tick_override else Config.tick_limit
        try:
            while True:
                with Config.state_lock:
                    running = Config.is_running
                    tick = Config.current_tick
                if limit and limit != -1 and tick >= limit:
                    eng.stop_simulation()
                if not running:
                    break
                time.sleep(0.1)
        except KeyboardInterrupt:
            eng.stop_simulation()

    # ------------------------------------------------------------------
    @staticmethod
    def _launch_gui(args) -> None:
        """Launch the Qt Quick interface."""

        import asyncio
        from PySide6.QtWidgets import QApplication, QMessageBox
        from PySide6.QtQml import QQmlApplicationEngine
        from PySide6.QtQuick import QQuickItem
        from qasync import QEventLoop
        from ui_new import core
        from ui_new.auth import resolve_connection_info
        from ui_new.ipc import ConnectError
        from Causal_Web.engine.stream.auth import default_session_file
        from ui_new.state import (
            Store,
            TelemetryModel,
            MetersModel,
            ExperimentModel,
            ReplayModel,
            LogsModel,
            DOEModel,
            GAModel,
            MCTSModel,
            PolicyModel,
            CompareModel,
            ResultsModel,
        )
        from ui_new.graph import GraphView  # noqa: F401  # register QML module

        app = QApplication([])
        engine = QQmlApplicationEngine()
        telemetry = TelemetryModel()
        meters = MetersModel()
        experiment = ExperimentModel()
        replay = ReplayModel()
        logs = LogsModel()
        store = Store()
        doe = DOEModel()
        ga_model = GAModel()
        mcts_model = MCTSModel()
        policy_model = PolicyModel()
        compare = CompareModel()
        results = ResultsModel()
        ctx = engine.rootContext()
        ctx.setContextProperty("telemetryModel", telemetry)
        ctx.setContextProperty("metersModel", meters)
        ctx.setContextProperty("experimentModel", experiment)
        ctx.setContextProperty("replayModel", replay)
        ctx.setContextProperty("logsModel", logs)
        ctx.setContextProperty("store", store)
        ctx.setContextProperty("doeModel", doe)
        ctx.setContextProperty("gaModel", ga_model)
        ctx.setContextProperty("mctsModel", mcts_model)
        ctx.setContextProperty("policyModel", policy_model)
        ctx.setContextProperty("compareModel", compare)
        ctx.setContextProperty("resultsModel", results)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        qml_path = os.path.join(base_dir, "..", "ui_new", "main.qml")
        if not os.path.exists(qml_path):
            qml_path = os.path.join(base_dir, "ui_new", "main.qml")
        qml_path = os.path.abspath(qml_path)
        qml_warnings: list = []
        engine.warnings.connect(lambda w: qml_warnings.extend(w))
        engine.load(qml_path)
        if not engine.rootObjects():
            errors = "\n".join(str(e) for e in qml_warnings)
            QMessageBox.critical(
                None,
                "UI load error",
                f"Failed to load QML at {qml_path}:\n{errors}",
            )
            return
        root = engine.rootObjects()[0]
        view = root.findChild(QQuickItem, "graphView")
        view.frameRendered.connect(meters.frame_drawn)

        loop = QEventLoop(app)
        asyncio.set_event_loop(loop)

        async def runner() -> None:
            for _ in range(10):
                try:
                    url, token = resolve_connection_info(
                        ws_url=args.ws_url,
                        token=args.token,
                        token_file=args.token_file,
                        ws_host=args.ws_host,
                        ws_port=args.ws_port,
                    )
                    await core.run(
                        url,
                        view,
                        telemetry,
                        experiment,
                        replay,
                        logs,
                        store,
                        doe,
                        ga_model,
                        mcts_model,
                        policy_model,
                        root,
                        token=token,
                    )
                    return
                except (FileNotFoundError, ConnectError) as e:
                    if isinstance(e, ConnectError) and "session expired" in str(e):
                        QMessageBox.critical(
                            None,
                            "Session expired",
                            "Engine session expired. Restart engine.",
                        )
                        return
                    await asyncio.sleep(0.3)
            path = args.token_file or default_session_file()
            host = args.ws_host or "127.0.0.1"
            port = args.ws_port or 8765
            QMessageBox.critical(
                None,
                "Engine not found",
                (
                    "Couldn't find local Causal Web engine. "
                    f"Looked for session file at {path} and tried ws://{host}:{port}."
                ),
            )

        loop.create_task(runner())
        with loop:
            loop.run_forever()


def main() -> None:
    """Entry point for external callers."""
    MainService().run()


if __name__ == "__main__":
    main()
