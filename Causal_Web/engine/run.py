from __future__ import annotations

"""CLI entrypoint to run the engine WebSocket server."""

import argparse
import asyncio

from ..config import Config, load_config
from .engine_v2 import adapter as eng
from .stream.server import DeltaBus, serve


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the engine server."""

    parser = argparse.ArgumentParser(description="Run Causal Web engine")
    parser.add_argument(
        "--config", default=Config.input_path("config.json"), help="Config JSON file"
    )
    parser.add_argument("--graph", default=Config.graph_file, help="Graph JSON file")
    parser.add_argument("--host", default="127.0.0.1", help="WebSocket host")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port")
    parser.add_argument(
        "--session-file", default=None, help="Path to session bundle JSON"
    )
    parser.add_argument("--session-token", default=None, help="Explicit session token")
    parser.add_argument(
        "--session-ttl", type=int, default=3600, help="Token lifetime in seconds"
    )
    return parser.parse_args(argv)


async def _serve(args: argparse.Namespace) -> None:
    """Launch the WebSocket server with the configured adapter."""

    adapter = eng.get_engine()
    bus = DeltaBus()
    await serve(
        bus,
        adapter,
        host=args.host,
        port=args.port,
        session_token=args.session_token,
        session_file=args.session_file,
        session_ttl=args.session_ttl,
    )


def main(argv: list[str] | None = None) -> None:
    """Run the engine server with the provided arguments."""

    args = _parse_args(argv)
    load_config(args.config)
    Config.graph_file = args.graph
    eng.build_graph(args.graph)
    with Config.state_lock:
        Config.is_running = True
    eng.simulation_loop()
    try:
        asyncio.run(_serve(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
