# config.py

class Config:
    tick_rate = 1.0  # Seconds between ticks (adjustable via GUI)
    is_running = False  # Whether the simulation loop is active
    max_ticks = 100  # Stop after this many ticks unless set to 0 for infinite
    current_tick = 0  # Counter to display progress

    # Global rhythmic forcing parameters
    phase_jitter = {
        "amplitude": 0.0,  # radians
        "period": 20      # ticks
    }
    coherence_wave = {
        "amplitude": 0.0,  # threshold modulation
        "period": 30
    }
    # interval for saving runtime snapshots of the graph
    snapshot_interval = 10

    # tick seeding configuration
    seeding = {
        "strategy": "static",  # static or probabilistic
        "probability": 0.1,   # used when strategy == "probabilistic"
    }

