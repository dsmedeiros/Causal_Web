"""Console entrypoint for the ``cw`` command."""

from __future__ import annotations

import argparse
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="cw")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run simulation")
    run_parser.add_argument("--gui", choices=["legacy", "new"], default="legacy")

    args, rest = parser.parse_known_args(argv)
    if args.command == "run":
        if args.gui == "legacy":
            from Causal_Web.main import MainService

            MainService(argv=rest).run()
        else:
            raise RuntimeError("New GUI is not yet available")


if __name__ == "__main__":
    main()
