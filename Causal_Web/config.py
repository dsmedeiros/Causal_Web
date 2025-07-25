# config.py

import os
import shutil
import threading


class Config:
    """Global configuration loaded from ``input/config.json``.

    Attributes
    ----------
    propagation_control:
        Dictionary containing ``enable_sip`` and ``enable_csp`` flags used to
        toggle the two propagation mechanisms.
    log_interval:
        Number of ticks between metric log writes.
    headless:
        When ``True`` disables observers and intermediate logging.
    """

    # Base directories for package resources
    base_dir = os.path.abspath(os.path.dirname(__file__))
    input_dir = os.path.join(base_dir, "input")
    config_file = os.path.join(input_dir, "config.json")
    output_root = os.path.join(base_dir, "output")
    runs_dir = os.path.join(output_root, "runs")
    archive_dir = os.path.join(output_root, "archive")
    analysis_dir = os.path.join(output_root, "analysis")
    ingest_dir = os.path.join(output_root, "ingest")
    output_dir = output_root

    @staticmethod
    def input_path(*parts: str) -> str:
        """Return absolute path under the ``input`` directory."""
        return os.path.join(Config.input_dir, *parts)

    @staticmethod
    def output_path(*parts: str) -> str:
        """Return absolute path under the current run directory."""
        return os.path.join(Config.output_dir, *parts)

    # Synchronization lock for cross-thread state access
    state_lock = threading.Lock()

    tick_rate = 1.0  # Seconds between ticks (adjustable via GUI)
    is_running = False  # Whether the simulation loop is active
    max_ticks = 100  # Stop after this many ticks unless set to 0 for infinite
    tick_limit = 10000
    allow_tick_override = True
    current_tick = 0  # Counter to display progress
    # Preallocated ticks for object pool
    TICK_POOL_SIZE = 10000

    # Per-log enable flags. Keys correspond to log file names under
    # ``output`` and values determine whether the file should be written.
    log_files = {
        "boundary_interaction_log.json": True,
        "bridge_decay_log.json": True,
        "bridge_dynamics_log.json": True,
        "bridge_reformation_log.json": True,
        "bridge_rupture_log.json": True,
        "bridge_state_log.json": True,
        "cluster_log.json": True,
        "coherence_log.json": True,
        "coherence_velocity_log.json": True,
        "collapse_chain_log.json": True,
        "collapse_front_log.json": True,
        "connectivity_log.json": True,
        "curvature_log.json": True,
        "decoherence_log.json": True,
        "event_log.json": True,
        "inspection_log.json": True,
        "interference_log.json": True,
        "interpretation_log.json": True,
        "law_drift_log.json": True,
        "law_wave_log.json": True,
        "layer_transition_log.json": True,
        "magnitude_failure_log.json": True,
        "meta_node_tick_log.json": True,
        "node_emergence_log.json": True,
        "node_state_log.json": True,
        "observer_disagreement_log.json": True,
        "propagation_failure_log.json": True,
        "refraction_log.json": True,
        "should_tick_log.json": True,
        "stable_frequency_log.json": True,
        "structural_growth_log.json": True,
        "tick_delivery_log.json": True,
        "tick_drop_log.json": True,
        "tick_emission_log.json": True,
        "tick_evaluation_log.json": True,
        "tick_propagation_log.json": True,
        "tick_seed_log.json": True,
    }

    @classmethod
    def is_log_enabled(cls, name: str) -> bool:
        """Return ``True`` if the given log file should be written."""
        return cls.log_files.get(name, True)

    @classmethod
    def save_log_files(cls, path: str | None = None) -> None:
        """Persist ``log_files`` settings back to ``config.json``.

        Parameters
        ----------
        path:
            Optional path to write. Defaults to :attr:`config_file`.
        """
        import json
        import os

        if path is None:
            path = cls.config_file
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
        else:
            data = {}
        data.setdefault("log_files", {})
        data["log_files"].update(cls.log_files)
        data["log_interval"] = getattr(cls, "log_interval", 1)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # Global rhythmic forcing parameters
    phase_jitter = {"amplitude": 0.0, "period": 20}  # radians  # ticks
    coherence_wave = {"amplitude": 0.0, "period": 30}  # threshold modulation
    # ticks over which to ramp up global forcing effects
    forcing_ramp_ticks = 20
    # interval for saving runtime snapshots of the graph
    snapshot_interval = 10
    # interval for expensive clustering and bridge management
    cluster_interval = 10
    random_seed: int | None = None
    thread_count = 1
    log_verbosity = "info"
    # interval between metric logs
    log_interval = 1
    # disable observers and intermediate logging when True
    headless = False

    # Node defaults
    memory_window = 20
    initial_coherence_threshold = 0.6
    steady_coherence_threshold = 0.85
    coherence_ramp_ticks = 10
    # minimum number of incoming ticks required for activation
    tick_threshold = 1
    # ticks a node must wait after firing before it can fire again
    refractory_period = 2.0

    # tick seeding configuration
    seeding = {
        "strategy": "static",  # static or probabilistic
        "probability": 0.1,  # used when strategy == "probabilistic"
        "phase_offsets": {},  # per-node phase offsets when strategy == "static"
    }

    # SIP recombination
    SIP_RECOMB_MIN_TRUST = 0.75
    SIP_MUTATION_SCALE = 0.005

    # SIP failure
    SIP_STABILIZATION_WINDOW = 5
    SIP_FAILURE_ENTROPY_INJECTION = 0.1

    # CSP
    CSP_RADIUS = 80
    CSP_MAX_NODES = 3
    CSP_TICK_BURST = 25
    CSP_STABILIZATION_WINDOW = 6
    CSP_COLLAPSE_INTENSITY_THRESHOLD = 0.4
    CSP_DECOHERENCE_THRESHOLD = 0.3
    CSP_ENTROPY_INJECTION = 0.2

    # Propagation limits
    max_children_per_node = 0  # 0 disables limit

    # Tick fan-out
    max_tick_fanout = 0  # limit edges a tick propagates across (0 = unlimited)

    # Decay factor for stored tick energy per tick
    tick_decay_factor = 1.0

    # Concurrency limits
    total_max_concurrent_firings = 0  # 0 disables global limit
    max_concurrent_firings_per_cluster = 0  # 0 disables per-cluster limit

    # Early formation tuning
    DRIFT_TOLERANCE_RAMP = 10
    FORMATION_REFRACTORY_RAMP = 20

    # Bridge stabilization
    BRIDGE_STABILIZATION_TICKS = 50

    # Spatial partitioning
    SPATIAL_GRID_SIZE = 50

    # Range for per-edge weights influencing delay/attenuation
    edge_weight_range = [1.0, 1.0]

    # Toggle propagation mechanisms
    propagation_control = {"enable_sip": True, "enable_csp": True}

    # Database connection details
    database = {
        "host": "localhost",
        "port": 5432,
        "user": "sim_user",
        "password": "secret_password",
        "dbname": "cwt_simulation",
    }

    @classmethod
    def new_run(cls, slug: str = "run") -> str:
        """Create and activate a new run directory.

        The current ``graph.json`` and ``config.json`` are copied into the
        run's ``input`` folder and summary metadata is stored in PostgreSQL.

        Parameters
        ----------
        slug:
            Human friendly slug appended to the timestamp.

        Returns
        -------
        str
            Absolute path to the newly created directory.
        """
        from datetime import datetime

        ts = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = os.path.join(cls.runs_dir, f"{ts}__{slug}")
        os.makedirs(run_dir, exist_ok=True)

        input_dest = os.path.join(run_dir, "input")
        os.makedirs(input_dest, exist_ok=True)
        graph_src = cls.input_path("graph.json")
        if os.path.exists(graph_src):
            shutil.copy2(graph_src, os.path.join(input_dest, "graph.json"))
        if os.path.exists(cls.config_file):
            shutil.copy2(cls.config_file, os.path.join(input_dest, "config.json"))

        cls.output_dir = run_dir

        try:
            from .database import record_run

            record_run(
                os.path.basename(run_dir),
                os.path.join(input_dest, "config.json"),
                os.path.join(input_dest, "graph.json"),
                run_dir,
            )
        except Exception:
            pass

        return run_dir

    @classmethod
    def load_from_file(cls, path: str) -> None:
        """Load configuration values from a JSON file.

        Only keys that already exist as attributes on ``Config`` will be
        assigned. Nested dictionaries are merged recursively when the existing
        attribute is also a ``dict``.

        Parameters
        ----------
        path:
            Path to the JSON configuration file.
        """
        import json

        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path) as f:
            data = json.load(f)
        cls.config_file = os.path.abspath(path)

        paths = data.get("paths")
        if isinstance(paths, dict):
            for key, value in paths.items():
                if hasattr(cls, key):
                    setattr(cls, key, os.path.abspath(value))

        for key, value in data.items():
            if not hasattr(cls, key):
                continue
            current = getattr(cls, key)
            if isinstance(current, dict) and isinstance(value, dict):
                current.update(value)
            else:
                setattr(cls, key, value)


def load_config(path: str | None = None) -> dict:
    """Load configuration from ``path`` and return the data."""
    if path is None:
        path = Config.input_path("config.json")
    Config.load_from_file(path)
    Config.config_file = os.path.abspath(path)
    import json

    with open(path) as f:
        return json.load(f)
