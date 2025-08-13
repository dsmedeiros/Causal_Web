# config.py

import math
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
    profile_output:
        Optional path to write ``cProfile`` statistics when profiling is
        enabled.
    chi_max:
        Max MPS bond dimension used for tensor chain compression.
    backend:
        Compute backend to use: ``"cpu"`` (default) or ``"cupy"``.
    engine_mode:
        Selects the simulation engine: ``"tick"`` for the legacy engine or
        ``"v2"`` for the strict-local core.
    windowing:
        Mapping of windowing coefficients used by the v2 engine. Expected keys
        include ``W0``, ``zeta1``, ``zeta2``, ``a``, ``b``, ``T_hold`` and
        ``C_min`` which together determine how vertex windows advance.
    rho_delay:
        Parameters controlling delayed density feedback in the v2 engine.
        The group accepts ``alpha_d``, ``alpha_leak``, ``eta``, ``gamma``,
        ``rho0`` and ``inject_mode`` coefficients.
    qwalk:
        Parameters controlling the split-step quantum walk. Includes an
        ``enabled`` flag and ``thetas`` mapping with ``theta1`` and
        ``theta2`` rotation angles.
    dispersion:
        Configuration for dispersion analysis. The ``k_values`` list
        specifies the wave numbers sampled when estimating
        ``ω(k)`` and group velocity.
    epsilon_pairs:
        Controls reinforcement and decay for ε-linked partners. Keys such as
        ``delta_ttl``, ``ancestry_prefix_L``, ``theta_max``, ``sigma0``,
        ``lambda_decay``, ``sigma_reinforce`` and ``sigma_min`` shape the
        dynamics.
    ancestry:
        Parameters for local ancestry fields. ``beta_m0`` sets the base
        down-weight applied to the phase moment while ``delta_m`` controls the
        decay applied when a window closes without any quantum arrivals.
    bell:
        Mutual information gate parameters for Bell pair matching. Supported
        keys include ``mi_mode``, ``kappa_a``, ``kappa_xi``, ``beta_m`` and
        ``beta_h``.
    theta_reset:
        Policy controlling how the Θ distribution ``p_v`` is reset when a
        vertex window closes. Supported values are ``"uniform"`` to reset to
        an even distribution, ``"renorm"`` to normalise the existing values
        and ``"hold"`` to leave the distribution untouched. Defaults to
        ``"renorm"``.
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
    profile_output: str | None = None

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
    run_seed = 0  # Seed for reproducible runs
    # Preallocated ticks for object pool
    TICK_POOL_SIZE = 10000
    N_DECOH = 3  # Fan-in threshold for thermodynamic behaviour
    N_CLASS = 6  # Fan-in threshold for classical fallback
    chi_max = 16  # Max MPS bond dimension
    hawking_delta_e = 1.0  # Energy quantum for horizon emissions
    #: Compute backend; ``"cpu"`` or ``"cupy"``
    backend = "cpu"

    # Split-step quantum walk configuration
    qwalk = {"enabled": False, "thetas": {"theta1": 0.35, "theta2": 0.2}}

    # Parameters for dispersion experiments
    dispersion = {"k_values": [0.0, 0.1]}

    #: Selected engine implementation: ``"v2"`` (strict-local) or ``"tick"``
    engine_mode = "v2"

    # Parameters for the experimental strict-local engine (``engine_mode = "v2"``)
    windowing = {
        "W0": 4.0,
        "zeta1": 0.3,
        "zeta2": 0.3,
        "a": 0.7,
        "b": 0.4,
        "k_theta": 0.7,
        "k_c": 0.4,
        "k_rho": 1.0,
        "T_hold": 2.0,
        "C_min": 0.1,
    }
    rho_delay = {
        "alpha_d": 0.1,
        "alpha_leak": 0.01,
        "eta": 0.2,
        "gamma": 0.8,
        "rho0": 1.0,
        "inject_mode": "incoming",
        "vectorized": True,
    }
    rho = {
        "update_mode": "heuristic",
        "variational": {"lambda_s": 0.2, "lambda_l": 0.01, "lambda_I": 1.0},
    }
    lccm = {
        "mode": "thresholds",
        "free_energy": {"k_theta": 1.0, "k_c": 1.0, "k_q": 0.2, "F_min": 0.3},
    }
    epsilon_pairs = {
        "delta_ttl": 2 * windowing["W0"],
        "ancestry_prefix_L": 16,
        "theta_max": math.pi / 12,
        "sigma0": 0.3,
        "lambda_decay": 0.05,
        "sigma_reinforce": 0.1,
        "sigma_min": 1e-3,
        "decay_interval": 32,
        "decay_on_window_close": True,
        "max_seeds_per_site": 64,
        "emit_per_delivery": False,
    }
    ancestry = {
        "beta_m0": 0.1,
        "delta_m": 0.02,
    }
    bell = {
        "enabled": False,
        "mi_mode": "MI_strict",
        "kappa_a": 0.0,
        # ``kappa_xi`` controls measurement noise; ``0`` means maximal noise.
        "kappa_xi": 0.0,
        "beta_m": 0.0,
        "beta_h": 0.0,
        "zeta_mode": "float",
        "alpha_R": 1.0,
        "k_mod": 3,
    }

    # Maximum length of the classical bit deque used for majority voting.
    max_deque: int = 8

    #: Reset policy for Θ distribution after window closure
    theta_reset = "renorm"

    #: Logging related settings used by the experimental engine.
    logging = {
        # Probability that a per-edge ρ/delay update is recorded.
        # A value of 0.0 disables per-edge logs while 1.0 logs all updates.
        "sample_rho_rate": 0.0,
        "sample_seed_rate": 1.0,
        "sample_bridge_rate": 1.0,
    }

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

    #: Sampling probability for throttled delivery logs
    log_delivery_sample_rate: float = 0.0

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

        if "delta_ttl" not in data.get("epsilon_pairs", {}):
            cls.epsilon_pairs["delta_ttl"] = 2 * cls.windowing.get(
                "W0", cls.windowing["W0"]
            )


def load_config(path: str | None = None) -> dict:
    """Load configuration from ``path`` and return the data."""
    if path is None:
        path = Config.input_path("config.json")
    Config.load_from_file(path)
    Config.config_file = os.path.abspath(path)
    import json

    with open(path) as f:
        return json.load(f)
