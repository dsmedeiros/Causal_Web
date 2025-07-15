# config.py

import os


class Config:
    # Base directories for package resources
    base_dir = os.path.abspath(os.path.dirname(__file__))
    input_dir = os.path.join(base_dir, "input")
    output_dir = os.path.join(base_dir, "output")

    @staticmethod
    def input_path(*parts: str) -> str:
        """Return absolute path under the ``input`` directory."""
        return os.path.join(Config.input_dir, *parts)

    @staticmethod
    def output_path(*parts: str) -> str:
        """Return absolute path under the ``output`` directory."""
        return os.path.join(Config.output_dir, *parts)

    tick_rate = 1.0  # Seconds between ticks (adjustable via GUI)
    is_running = False  # Whether the simulation loop is active
    max_ticks = 100  # Stop after this many ticks unless set to 0 for infinite
    tick_limit = 10000
    allow_tick_override = True
    current_tick = 0  # Counter to display progress

    # Global rhythmic forcing parameters
    phase_jitter = {"amplitude": 0.0, "period": 20}  # radians  # ticks
    coherence_wave = {"amplitude": 0.0, "period": 30}  # threshold modulation
    # interval for saving runtime snapshots of the graph
    snapshot_interval = 10

    # tick seeding configuration
    seeding = {
        "strategy": "static",  # static or probabilistic
        "probability": 0.1,  # used when strategy == "probabilistic"
    }
