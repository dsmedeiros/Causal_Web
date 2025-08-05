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
        Number of ticks between metric log writes and graph snapshots.
    headless:
        When ``True`` disables observers and intermediate logging.
    graph_file:
        Path to the current graph JSON file used by the engine.
    """

    # Base directories for package resources
    base_dir = os.path.abspath(os.path.dirname(__file__))
    input_dir = os.path.join(base_dir, "input")
    config_file = os.path.join(input_dir, "config.json")
    graph_file = os.path.join(input_dir, "graph.json")
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

    # Mapping of ``category`` -> {``label``: bool} controlling which logs are
    # written. Categories correspond to consolidated output files and the labels
    # are used as ``label`` or ``event_type`` within those files.

    DEFAULT_LOG_FILES = {
        "tick": {
            "coherence_log": True,
            "decoherence_log": True,
            "coherence_velocity_log": True,
            "law_wave_log": True,
            "stable_frequency_log": True,
            "proper_time_log": True,
            "interference_log": True,
            "curvature_log": True,
            "node_state_log": True,
            "tick_emission_log": True,
            "tick_propagation_log": True,
            "tick_delivery_log": True,
            "tick_seed_log": True,
            "tick_drop_log": True,
            "should_tick_log": True,
            "magnitude_failure_log": True,
            "cluster_log": True,
            "classicalization_map": True,
            "structural_growth_log": True,
            "bridge_state": True,
        },
        "phenomena": {
            "bridge_state": True,
            "connectivity_log": True,
            "void_node_map": True,
            "curvature_map": True,
            "regional_pressure_map": True,
            "meta_node_ticks": True,
            "global_diagnostics": True,
            "cluster_influence_matrix": True,
        },
        "event": {
            "bridge_rupture_log": True,
            "bridge_reformation_log": True,
            "bridge_decay_log": True,
            "bridge_dynamics_log": True,
            "bridge_ruptured": True,
            "bridge_reformed": True,
            "collapse_front_log": True,
            "collapse_chain_log": True,
            "propagation_failure_log": True,
            "tick_drop_log": True,
            "layer_transition_log": True,
            "refraction_log": True,
            "node_emergence_log": True,
            "law_drift_log": True,
            "law_wave_event": True,
            "observer_perceived_field": True,
            "observer_disagreement_log": True,
            "boundary_interaction_log": True,
            "simulation_state": True,
            "tick_emission_log": True,
            "tick_propagation_log": True,
            "tick_delivery_log": True,
            "tick_seed_log": True,
            "event_log": True,
        },
        "entangled": {
            "entangled_tick": True,
            "measurement": True,
        },
    }

    # Default runtime copy
    log_files = {k: dict(v) for k, v in DEFAULT_LOG_FILES.items()}

    #: Allowed logging modes. ``diagnostic`` enables all logs. ``tick`` enables
    #: per-tick metrics, ``phenomena`` enables aggregated summaries and
    #: ``events`` enables event driven logs.
    logging_mode = ["diagnostic"]

    #: Files written on a per-tick basis. Derived from :mod:`logger`.
    PERIODIC_FILES = {
        "cluster_influence_matrix",
        "curvature_map",
        "global_diagnostics",
        "regional_pressure_map",
        "void_node_map",
        "explanation_graph",
        "causal_chains",
        "causal_timeline",
        "boundary_interaction_log",
        "bridge_decay_log",
        "bridge_dynamics_log",
        "bridge_reformation_log",
        "bridge_rupture_log",
        "bridge_state",
        "classicalization_map",
        "cluster_log",
        "coherence_log",
        "coherence_velocity_log",
        "collapse_chain_log",
        "collapse_front_log",
        "connectivity_log",
        "curvature_log",
        "decoherence_log",
        "event_log",
        "inspection_log",
        "interference_log",
        "law_drift_log",
        "law_wave_log",
        "stable_frequency_log",
        "layer_transition_log",
        "layer_transition_events",
        "meta_node_ticks",
        "node_emergence_log",
        "node_state_log",
        "node_state_map",
        "observer_disagreement_log",
        "observer_perceived_field",
        "proper_time_log",
        "refraction_log",
        "structural_growth_log",
    }

    #: Files summarising emergent behaviour rather than raw events.
    PHENOMENA_FILES = {
        "cluster_influence_matrix",
        "curvature_map",
        "global_diagnostics",
        "regional_pressure_map",
        "void_node_map",
        "explanation_graph",
        "causal_chains",
        "causal_timeline",
        "connectivity_log",
        "classicalization_map",
        "interpretation_log",
        "inspection_log",
    }

    #: Files generated each tick or at regular intervals.
    TICK_FILES = {
        "bridge_state",
        "cluster_log",
        "coherence_log",
        "coherence_velocity_log",
        "curvature_log",
        "decoherence_log",
        "interference_log",
        "law_wave_log",
        "meta_node_ticks",
        "node_state_log",
        "proper_time_log",
        "structural_growth_log",
        "tick_delivery_log",
        "tick_evaluation_log",
        "tick_seed_log",
        "stable_frequency_log",
    }

    @classmethod
    def category_for_file(cls, name: str) -> str:
        """Return the logging category for *name*.

        The provided ``name`` may include a ``.json`` extension which will be
        stripped before lookup."""

        base = name.removesuffix(".json")
        if base in cls.TICK_FILES:
            return "tick"
        if base in cls.PHENOMENA_FILES:
            return "phenomena"
        return "event"

    @classmethod
    def is_category_enabled(cls, category: str) -> bool:
        """Return ``True`` if ``category`` should be written based on mode."""
        mode = set(getattr(cls, "logging_mode", ["diagnostic"]))
        return "diagnostic" in mode or category in mode

    @classmethod
    def is_log_enabled(cls, category: str, label: str | None = None) -> bool:
        """Return ``True`` if a log entry should be written."""

        cfg = cls.log_files.get(category, {})
        if label is not None and not cfg.get(label.removesuffix(".json"), True):
            return False
        return cls.is_category_enabled(category)

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
        for cat, mapping in cls.log_files.items():
            data["log_files"].setdefault(cat, {})
            data["log_files"][cat].update(mapping)
        data["log_interval"] = getattr(cls, "log_interval", 1)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # Global rhythmic forcing parameters
    phase_jitter = {"amplitude": 0.0, "period": 20}  # radians  # ticks
    coherence_wave = {"amplitude": 0.0, "period": 30}  # threshold modulation
    # ticks over which to ramp up global forcing effects
    forcing_ramp_ticks = 20
    # interval for expensive clustering operations
    cluster_interval = 10
    # interval for dynamic bridge management
    bridge_interval = 10
    random_seed: int | None = None
    thread_count = 1
    log_verbosity = "info"
    use_dynamic_density = False
    density_radius = 1
    delay_density_scaling = 1.0
    density_calc = "local_tick_saturation"
    traffic_decay = 0.9
    traffic_weight = 0.1
    density_overlay_file: str | None = None
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

    # Phase smoothing
    smooth_phase = False
    phase_smoothing_alpha = 0.1

    # Natural propagation limits
    max_cumulative_delay = 25
    min_coherence_threshold = 0.2
    log_tick_drops = True

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
    propagation_control = {
        "enable_sip_child": True,
        "enable_sip_recomb": True,
        "enable_csp": True,
    }

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
        from datetime import datetime, timezone

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = os.path.join(cls.runs_dir, f"{ts}__{slug}")
        os.makedirs(run_dir, exist_ok=True)

        input_dest = os.path.join(run_dir, "input")
        os.makedirs(input_dest, exist_ok=True)
        graph_src = cls.graph_file
        if os.path.exists(graph_src):
            shutil.copy2(
                graph_src, os.path.join(input_dest, os.path.basename(graph_src))
            )
        if os.path.exists(cls.config_file):
            shutil.copy2(cls.config_file, os.path.join(input_dest, "config.json"))

        cls.output_dir = run_dir

        try:
            from .database import record_run

            record_run(
                os.path.basename(run_dir),
                os.path.join(input_dest, "config.json"),
                os.path.join(input_dest, os.path.basename(graph_src)),
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
        attribute is also a ``dict``. Relative paths under the ``paths`` section
        are resolved relative to the directory containing ``path``.

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
        base_dir = os.path.dirname(cls.config_file)

        paths = data.get("paths")
        if isinstance(paths, dict):
            for key, value in paths.items():
                if hasattr(cls, key):
                    if not os.path.isabs(value):
                        value = os.path.join(base_dir, value)
                    setattr(cls, key, os.path.abspath(value))

        for key, value in data.items():
            if not hasattr(cls, key):
                continue
            if key == "graph_file" and not os.path.isabs(value):
                value = os.path.join(base_dir, value)
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
