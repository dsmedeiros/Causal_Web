"""Console entrypoint for the ``cw`` command."""

from __future__ import annotations

import argparse
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> None:
    """Parse ``cw`` CLI arguments and dispatch to the runner."""

    parser = argparse.ArgumentParser(prog="cw")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("run", help="Run simulation")

    args, rest = parser.parse_known_args(argv)
    if args.command == "run":
        from Causal_Web.main import MainService

        MainService(argv=rest).run()


if __name__ == "__main__":
    main()
