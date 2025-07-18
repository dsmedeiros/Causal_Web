# main.py

"""Entry point for launching the Causal Web simulation GUI."""

import argparse
from .gui.dashboard import launch
from .config import Config


def main() -> None:
    """Parse CLI arguments and start the GUI."""
    parser = argparse.ArgumentParser(description="Run Causal Web simulation")
    parser.add_argument("--config", help="Path to JSON configuration file")
    args = parser.parse_args()

    if args.config:
        Config.load_from_file(args.config)

    launch()


if __name__ == "__main__":
    main()
