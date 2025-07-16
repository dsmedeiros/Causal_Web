import time
import threading
from ..config import Config
from .graph import CausalGraph
from .observer import Observer
from .log_interpreter import run_interpreter
from .tick_seeder import TickSeeder
from .logger import log_json, logger
import json
import numpy as np
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
import math

# Global graph instance
graph = CausalGraph()
observers = []
kappa = 0.5  # curvature strength for refraction fields
_law_wave_stability = {}
seeder = TickSeeder(graph)

# Phase 6 metrics
void_absorption_events = 0
boundary_interactions_count = 0
bridges_reformed_count = 0
_decay_durations = []

# --- Propagation tracking ---
_sip_pending = []  # list of (child_id, parents, spawn_tick, origin_type)
_csp_seeds = []  # list of (seed_id, parent_id, x, y, spawn_tick)

# GUI hook for failed CSP seed ripples
recent_csp_failures = []  # list of dict(x,y,tick,intensity)

_sip_success_count = 0
_sip_failure_count = 0
_csp_success_count = 0
_csp_failure_count = 0

# --- Adaptive constraint tracking ---
_recent_node_counts: list[int] = []
_dynamic_coherence_offset: float = 0.0

  _spawn_counts = {}
_spawn_tick = -1


# --- Phase 8 parameters ---
SIP_COHERENCE_DURATION = 3
SIP_DECOHERENCE_THRESHOLD = 0.5


def apply_global_forcing(tick: int) -> None:
    """Apply rhythmic modulation to all nodes with a startup ramp."""
    jitter = Config.phase_jitter
    wave = Config.coherence_wave
    ramp = min(tick / max(1, Config.forcing_ramp_ticks), 1.0)
    for node in graph.nodes.values():
        if jitter["amplitude"] and jitter.get("period", 0):
            node.phase += (
                ramp
                * jitter["amplitude"]
                * math.sin(2 * math.pi * tick / jitter["period"])
            )
        if wave["amplitude"] and wave.get("period", 0):
            mod = (
                ramp * wave["amplitude"] * math.sin(2 * math.pi * tick / wave["period"])
            )
            node.current_threshold = max(0.1, node.current_threshold - mod)


def update_coherence_constraints() -> None:
    """Adjust node thresholds based on network size and growth."""
    global _recent_node_counts, _dynamic_coherence_offset
    count = len(graph.nodes)
    _recent_node_counts.append(count)
    if len(_recent_node_counts) > 5:
        _recent_node_counts.pop(0)
    growth = 0
    if len(_recent_node_counts) >= 2:
        growth = _recent_node_counts[-1] - _recent_node_counts[0]
    offset = min(0.3, 0.01 * count + 0.02 * growth)
    _dynamic_coherence_offset = offset
    for node in graph.nodes.values():
        node.dynamic_offset = offset
        node.refractory_period = max(1.0, 2 + offset * 5)


def clear_output_directory():
    """Remove or truncate JSON output files from previous runs."""
    out_dir = Config.output_dir
    if not os.path.isdir(out_dir):
        return
    for name in os.listdir(out_dir):
        if name == "__init__.py":
            continue
        path = os.path.join(out_dir, name)
        if os.path.isfile(path):
            open(path, "w").close()
        elif os.path.isdir(path):
            shutil.rmtree(path)


def build_graph():
    clear_output_directory()
    graph.load_from_file(Config.input_path("graph.json"))
    global seeder
    seeder = TickSeeder(graph)


def add_observer(observer: Observer):
    observers.append(observer)


def emit_ticks(global_tick):
    """Seed ticks into the graph via the configured seeder."""
    seeder.seed(global_tick)


def propagate_phases(global_tick):
    """Propagate phases scheduled during node ticks.

    Previous versions walked through each node's ``tick_history`` and
    rescheduled downstream ticks on every global step. This caused phases to
    arrive after twice the intended edge delay. Node.apply_tick now schedules
    outgoing phases directly, so this function no longer performs any work but
    is kept for API compatibility.
    """
    pass


def evaluate_nodes(global_tick):
    for node in graph.nodes.values():
        node.maybe_tick(global_tick, graph)


def log_curvature_per_tick(global_tick):
    log = {}
    for edge in graph.edges:
        src = graph.get_node(edge.source)
        tgt = graph.get_node(edge.target)
        if not src or not tgt:
            continue
        df = abs(src.law_wave_frequency - tgt.law_wave_frequency)
        curved = edge.adjusted_delay(
            src.law_wave_frequency, tgt.law_wave_frequency, kappa
        )
        log[f"{edge.source}->{edge.target}"] = {
            "delta_f": round(df, 4),
            "curved_delay": round(curved, 4),
        }
    log_json(Config.output_path("curvature_log.json"), {str(global_tick): log})


def log_bridge_states(global_tick):
    snapshot = {
        b.bridge_id: {
            "active": b.active,
            "last_activation": b.last_activation,
            "last_rupture_tick": b.last_rupture_tick,
            "last_reform_tick": b.last_reform_tick,
            "coherence_at_reform": b.coherence_at_reform,
            "trust_score": b.trust_score,
            "reinforcement": b.reinforcement_streak,
        }
        for b in graph.bridges
    }
    log_json(Config.output_path("bridge_state_log.json"), {str(global_tick): snapshot})


def log_meta_node_ticks(global_tick):
    events = {}
    for meta_id, meta in graph.meta_nodes.items():
        member_ticks = [
            nid
            for nid in meta.member_ids
            if any(t.time == global_tick for t in graph.get_node(nid).tick_history)
        ]
        if member_ticks:
            events[meta_id] = member_ticks
        if events:
            log_json(
                Config.output_path("meta_node_tick_log.json"),
                {str(global_tick): events},
            )


def snapshot_graph(global_tick):
    interval = getattr(Config, "snapshot_interval", 0)
    if interval and global_tick % interval == 0:
        path_dir = os.path.join(Config.output_dir, "runtime_graph_snapshots")
        os.makedirs(path_dir, exist_ok=True)
        path = os.path.join(path_dir, f"graph_{global_tick}.json")
        with open(path, "w") as f:
            json.dump(graph.to_dict(), f, indent=2)
        return path
    return None


def _bridge_thresholds(global_tick: int) -> tuple[float, float]:
    """Return coherence and drift thresholds for bridge formation."""
    if global_tick < 20:
        return 0.6, 0.5
    if global_tick < 50:
        progress = (global_tick - 20) / 30
        coh = 0.6 + progress * (0.9 - 0.6)
        drift = 0.5 - progress * (0.5 - 0.1)
        return coh, drift
    return 0.9, 0.1


def dynamic_bridge_management(global_tick):
    """Form new bridges between coherent nodes on the fly."""
    ids = list(graph.nodes.keys())
    existing = {(b.node_a_id, b.node_b_id) for b in graph.bridges}
    existing |= {(b.node_b_id, b.node_a_id) for b in graph.bridges}
    coherence_thresh, drift_thresh = _bridge_thresholds(global_tick)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a = graph.get_node(ids[i])
            b = graph.get_node(ids[j])
            if (a.id, b.id) in existing:
                continue
            drift = abs((a.phase - b.phase + math.pi) % (2 * math.pi) - math.pi)
            if (
                drift < drift_thresh
                and a.coherence > coherence_thresh
                and b.coherence > coherence_thresh
            ):
                graph.add_bridge(
                    a.id,
                    b.id,
                    formed_at_tick=global_tick,
                    seeded=False,
                )
                bridge = graph.bridges[-1]
                bridge._log_dynamics(
                    global_tick,
                    "formed",
                    {
                        "phase_delta": drift,
                        "coherence": (a.coherence + b.coherence) / 2,
                    },
                )
                bridge.update_state(global_tick)


def _update_growth_log(tick: int) -> None:
    """Record structural growth per tick."""
    global _sip_success_count, _sip_failure_count, _csp_success_count, _csp_failure_count
    record = {
        "tick": tick,
        "node_count": len(graph.nodes),
        "sip_success": _sip_success_count,
        "sip_failures": _sip_failure_count,
        "csp_success": _csp_success_count,
        "csp_failures": _csp_failure_count,
    }
    log_json(Config.output_path("structural_growth_log.json"), record)
    _sip_success_count = 0
    _sip_failure_count = 0
    _csp_success_count = 0
    _csp_failure_count = 0


def _spawn_sip_child(parent, tick: int):
    """Generate a new node via Stability-Induced Propagation."""
    global _spawn_counts
    if _spawn_counts.get(parent.id, 0) >= Config.max_children_per_node > 0:
        with open(Config.output_path("propagation_failure_log.json"), "a") as f:
            f.write(
                json.dumps(
                    {
                        "tick": tick,
                        "type": "SPAWN_LIMIT",
                        "parent": parent.id,
                        "origin_type": "SIP_BUD",
                    }
                )
                + "\n"
            )
        return
    child_id = f"{parent.id}_S{tick}"
    if child_id in graph.nodes:
        return
    graph.add_node(
        child_id,
        x=parent.x + 10,
        y=parent.y + 10,
        frequency=parent.frequency,
        origin_type="SIP_BUD",
        generation_tick=tick,
        parent_ids=[parent.id],
    )
    graph.add_edge(parent.id, child_id)
    log_json(
        Config.output_path("node_emergence_log.json"),
        {
            "id": child_id,
            "tick": tick,
            "parents": [parent.id],
            "origin_type": "SIP_BUD",
            "generation_tick": tick,
            "sigma_phi": parent.law_wave_frequency,
            "phase_confidence_index": graph.get_node(child_id).phase_confidence_index,
        },
    )
    _update_growth_log(tick)
    global _sip_pending, _sip_success_count
    _sip_pending.append((child_id, [parent.id], tick, "SIP_BUD"))
    _sip_success_count += 1
    _spawn_counts[parent.id] = _spawn_counts.get(parent.id, 0) + 1


def _spawn_sip_recomb_child(parent_a, parent_b, tick: int):
    """Generate a new node via dual-parent recombination."""
    global _spawn_counts
    if any(
        _spawn_counts.get(pid, 0) >= Config.max_children_per_node > 0
        for pid in (parent_a.id, parent_b.id)
    ):
        with open(Config.output_path("propagation_failure_log.json"), "a") as f:
            f.write(
                json.dumps(
                    {
                        "tick": tick,
                        "type": "SPAWN_LIMIT",
                        "parent": f"{parent_a.id},{parent_b.id}",
                        "origin_type": "SIP_RECOMB",
                    }
                )
                + "\n"
            )
        return
    child_id = f"{parent_a.id}_{parent_b.id}_R{tick}"
    if child_id in graph.nodes:
        return
    freq = (parent_a.frequency + parent_b.frequency) / 2 + np.random.normal(
        0, Config.SIP_MUTATION_SCALE
    )
    x = (parent_a.x + parent_b.x) / 2 + np.random.uniform(-5, 5)
    y = (parent_a.y + parent_b.y) / 2 + np.random.uniform(-5, 5)
    graph.add_node(
        child_id,
        x=x,
        y=y,
        frequency=freq,
        origin_type="SIP_RECOMB",
        generation_tick=tick,
        parent_ids=[parent_a.id, parent_b.id],
    )
    graph.add_edge(parent_a.id, child_id)
    graph.add_edge(parent_b.id, child_id)
    log_json(
        Config.output_path("node_emergence_log.json"),
        {
            "id": child_id,
            "tick": tick,
            "parents": [parent_a.id, parent_b.id],
            "origin_type": "SIP_RECOMB",
            "generation_tick": tick,
            "sigma_phi": freq,
            "phase_confidence_index": graph.get_node(child_id).phase_confidence_index,
        },
    )
    global _sip_pending, _sip_success_count
    _sip_pending.append((child_id, [parent_a.id, parent_b.id], tick, "SIP_RECOMB"))
    _sip_success_count += 1
    for pid in (parent_a.id, parent_b.id):
        _spawn_counts[pid] = _spawn_counts.get(pid, 0) + 1


def _check_sip_failures(tick: int) -> None:
    """Assess pending SIP offspring for stabilization failures."""
    global _sip_pending, _sip_failure_count
    for child_id, parents, start, otype in list(_sip_pending):
        if tick - start < Config.SIP_STABILIZATION_WINDOW:
            continue
        node = graph.get_node(child_id)
        success = node and len(node.tick_history) > 0 and node.coherence > 0.5
        if not success:
            if node:
                graph.edges = [
                    e
                    for e in graph.edges
                    if e.source != child_id and e.target != child_id
                ]
                del graph.nodes[child_id]
            for pid in parents:
                p = graph.get_node(pid)
                if p:
                    p.decoherence_debt += Config.SIP_FAILURE_ENTROPY_INJECTION
            log_json(
                Config.output_path("propagation_failure_log.json"),
                {
                    "tick": tick,
                    "type": "SIP_FAILURE",
                    "parent": parents[0],
                    "child": child_id,
                    "reason": "Insufficient coherence after "
                    f"{Config.SIP_STABILIZATION_WINDOW} ticks",
                    "entropy_injected": Config.SIP_FAILURE_ENTROPY_INJECTION,
                    "origin_type": otype,
                },
            )
            _sip_failure_count += 1
        _sip_pending.remove((child_id, parents, start, otype))


def trigger_csp(parent_id: str, x: float, y: float, tick: int) -> None:
    """Initiate collapse-seeded propagation from the given location."""
    for i in range(Config.CSP_MAX_NODES):
        seed_id = f"{parent_id}_CSPseed{i}_{tick}"
        dx = np.random.uniform(-Config.CSP_RADIUS, Config.CSP_RADIUS)
        dy = np.random.uniform(-Config.CSP_RADIUS, Config.CSP_RADIUS)
        _csp_seeds.append(
            {
                "id": seed_id,
                "parent": parent_id,
                "x": x + dx,
                "y": y + dy,
                "tick": tick,
            }
        )


def _process_csp_seeds(tick: int) -> None:
    global _csp_seeds, _csp_success_count, _csp_failure_count
    for seed in list(_csp_seeds):
        if tick - seed["tick"] < Config.CSP_STABILIZATION_WINDOW:
            continue
        coherence = np.random.rand()
        if coherence > 0.5:
            if _spawn_counts.get(seed["parent"], 0) >= Config.max_children_per_node > 0:
                with open(Config.output_path("propagation_failure_log.json"), "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "tick": tick,
                                "type": "SPAWN_LIMIT",
                                "parent": seed["parent"],
                                "origin_type": "CSP",
                            }
                        )
                        + "\n"
                    )
                _csp_seeds.remove(seed)
                continue
            node_id = seed["id"].replace("CSPseed", "CSP")
            graph.add_node(
                node_id,
                x=seed["x"],
                y=seed["y"],
                origin_type="CSP",
                generation_tick=tick,
                parent_ids=[seed["parent"]],
            )
            graph.add_edge(seed["parent"], node_id)
            log_json(
                Config.output_path("node_emergence_log.json"),
                {
                    "id": node_id,
                    "tick": tick,
                    "parents": [seed["parent"]],
                    "origin_type": "CSP",
                    "generation_tick": tick,
                    "sigma_phi": graph.get_node(node_id).law_wave_frequency,
                    "phase_confidence_index": graph.get_node(
                        node_id
                    ).phase_confidence_index,
                },
            )
            log_json(
                Config.output_path("collapse_chain_log.json"),
                {
                    "tick": tick,
                    "source": seed["parent"],
                    "children_spawned": [node_id],
                },
            )
            _csp_success_count += 1
            _spawn_counts[seed["parent"]] = _spawn_counts.get(seed["parent"], 0) + 1
        else:
            parent = graph.get_node(seed["parent"])
            if parent:
                parent.decoherence_debt += Config.CSP_ENTROPY_INJECTION
            log_json(
                Config.output_path("propagation_failure_log.json"),
                {
                    "tick": tick,
                    "type": "CSP_FAILURE",
                    "parent": seed["parent"],
                    "reason": "Seed failed to cohere",
                    "entropy_injected": Config.CSP_ENTROPY_INJECTION,
                    "origin_type": "CSP",
                    "location": [seed["x"], seed["y"]],
                },
            )
            recent_csp_failures.append(
                {
                    "x": seed["x"],
                    "y": seed["y"],
                    "tick": tick,
                    "intensity": Config.CSP_ENTROPY_INJECTION,
                }
            )
            _csp_failure_count += 1
        _csp_seeds.remove(seed)


def _update_simulation_state(
    paused: bool, stopped: bool, tick: int, snapshot: str | None
) -> None:
    """Persist runtime simulation state."""
    state = {
        "paused": paused,
        "stopped": stopped,
        "current_tick": tick,
        "graph_snapshot": snapshot,
    }
    with open(Config.output_path("simulation_state.json"), "w") as f:
        json.dump(state, f, indent=2)


def check_propagation(tick: int) -> None:
    """Evaluate nodes for propagation triggers."""
    global _spawn_counts, _spawn_tick
    if _spawn_tick != tick:
        _spawn_counts = {}
        _spawn_tick = tick
    _check_sip_failures(tick)
    _process_csp_seeds(tick)
    for node in list(graph.nodes.values()):
        if (
            node.sip_streak >= SIP_COHERENCE_DURATION
            and node.decoherence_debt < SIP_DECOHERENCE_THRESHOLD
        ):
            _spawn_sip_child(node, tick)
            node.sip_streak = 0

    # recombination across stable, trusted bridges
    for bridge in graph.bridges:
        if not bridge.active or bridge.state.name != "STABLE":
            continue
        if bridge.trust_score < Config.SIP_RECOMB_MIN_TRUST:
            continue
        a = graph.get_node(bridge.node_a_id)
        b = graph.get_node(bridge.node_b_id)
        if not a or not b:
            continue
        if (
            a.sip_streak >= SIP_COHERENCE_DURATION
            and b.sip_streak >= SIP_COHERENCE_DURATION
            and a.decoherence_debt < SIP_DECOHERENCE_THRESHOLD
            and b.decoherence_debt < SIP_DECOHERENCE_THRESHOLD
        ):
            _spawn_sip_recomb_child(a, b, tick)
            a.sip_streak = 0
            b.sip_streak = 0


def _compute_metrics(node, tick_time):
    decoherence = node.compute_decoherence_field(tick_time)
    coherence = node.compute_coherence_level(tick_time)
    interference = len(node.pending_superpositions.get(tick_time, []))
    return (
        node.id,
        decoherence,
        coherence,
        interference,
        node.node_type.value,
        node.coherence_credit,
        node.decoherence_debt,
    )


def log_metrics_per_tick(global_tick):
    decoherence_log = {}
    coherence_log = {}
    classical_state = {}
    coherence_velocity = {}
    law_wave_log = {}
    interference_log = {}
    credit_log = {}
    debt_log = {}
    type_log = {}

    # Store last coherence to compute delta
    if not hasattr(log_metrics_per_tick, "_last_coherence"):
        log_metrics_per_tick._last_coherence = {}

    with ThreadPoolExecutor() as ex:
        results = list(
            ex.map(lambda n: _compute_metrics(n, global_tick), graph.nodes.values())
        )

    for node_id, decoherence, coherence, interference, ntype, credit, debt in results:
        node = graph.get_node(node_id)
        prev = log_metrics_per_tick._last_coherence.get(node_id, coherence)
        delta = coherence - prev
        log_metrics_per_tick._last_coherence[node_id] = coherence
        node.coherence_velocity = delta

        node.update_classical_state(decoherence, tick_time=global_tick, graph=graph)

        # track law-wave stability
        record = _law_wave_stability.setdefault(node_id, {"freqs": [], "stable": 0})
        record["freqs"].append(node.law_wave_frequency)
        if len(record["freqs"]) > 5:
            record["freqs"].pop(0)
        if len(record["freqs"]) == 5:
            if np.std(record["freqs"]) < 0.01:
                record["stable"] += 1
            else:
                record["stable"] = 0
        if record["stable"] >= 10:
            node.refractory_period = max(1.0, node.refractory_period - 0.1)
            log_json(
                Config.output_path("law_drift_log.json"),
                {
                    "tick": global_tick,
                    "node": node_id,
                    "new_refractory_period": node.refractory_period,
                },
            )
            record["stable"] = 0

        decoherence_log[node_id] = round(decoherence, 4)
        coherence_log[node_id] = round(coherence, 4)
        classical_state[node_id] = getattr(node, "is_classical", False)
        coherence_velocity[node_id] = round(delta, 5)
        law_wave_log[node_id] = round(node.law_wave_frequency, 4)
        interference_log[node_id] = interference
        credit_log[node_id] = round(credit, 3)
        debt_log[node_id] = round(debt, 3)
        type_log[node_id] = ntype

    clusters = graph.detect_clusters()
    graph.create_meta_nodes(clusters)

    log_json(Config.output_path("cluster_log.json"), {str(global_tick): clusters})

    log_json(Config.output_path("law_wave_log.json"), {str(global_tick): law_wave_log})

    log_json(
        Config.output_path("decoherence_log.json"), {str(global_tick): decoherence_log}
    )
    log_json(
        Config.output_path("coherence_log.json"), {str(global_tick): coherence_log}
    )
    log_json(
        Config.output_path("coherence_velocity_log.json"),
        {str(global_tick): coherence_velocity},
    )
    log_json(
        Config.output_path("classicalization_map.json"),
        {str(global_tick): classical_state},
    )
    log_json(
        Config.output_path("interference_log.json"),
        {str(global_tick): interference_log},
    )
    log_json(
        Config.output_path("tick_density_map.json"),
        {str(global_tick): interference_log},
    )
    log_json(
        Config.output_path("node_state_log.json"),
        {
            str(global_tick): {
                "type": type_log,
                "credit": credit_log,
                "debt": debt_log,
            }
        },
    )


def simulation_loop():
    """Start the main simulation thread."""

    def run():
        global_tick = 0
        _update_simulation_state(False, False, global_tick, None)
        while Config.is_running:
            Config.current_tick = global_tick
            print(f"== Tick {global_tick} ==")

            apply_global_forcing(global_tick)
            update_coherence_constraints()

            emit_ticks(global_tick)
            propagate_phases(global_tick)
            evaluate_nodes(global_tick)
            check_propagation(global_tick)

            log_metrics_per_tick(global_tick)
            log_bridge_states(global_tick)
            log_meta_node_ticks(global_tick)
            log_curvature_per_tick(global_tick)
            dynamic_bridge_management(global_tick)
            snapshot_path = snapshot_graph(global_tick)
            _update_simulation_state(False, False, global_tick, snapshot_path)

            for obs in observers:
                obs.observe(graph, global_tick)
                inferred = obs.infer_field_state()
                with open(
                    Config.output_path("observer_perceived_field.json"), "a"
                ) as f:
                    f.write(
                        json.dumps(
                            {"tick": global_tick, "observer": obs.id, "state": inferred}
                        )
                        + "\n"
                    )

                actual = {n.id: len(n.tick_history) for n in graph.nodes.values()}
                diff = {
                    nid: {
                        "actual": actual.get(nid, 0),
                        "inferred": inferred.get(nid, 0),
                    }
                    for nid in set(actual) | set(inferred)
                    if actual.get(nid, 0) != inferred.get(nid, 0)
                }
                if diff:
                    with open(
                        Config.output_path("observer_disagreement_log.json"), "a"
                    ) as f:
                        f.write(
                            json.dumps(
                                {"tick": global_tick, "observer": obs.id, "diff": diff}
                            )
                            + "\n"
                        )

            for bridge in graph.bridges:
                bridge.apply(global_tick, graph)

            limit = (
                Config.tick_limit if Config.allow_tick_override else Config.max_ticks
            )
            if limit and limit != -1 and global_tick >= limit:
                Config.is_running = False
                _update_simulation_state(False, True, global_tick, snapshot_path)
                write_output()

            global_tick += 1
            time.sleep(Config.tick_rate)

    threading.Thread(target=run, daemon=True).start()


def write_output():
    logger.stop()
    with open(Config.output_path("tick_trace.json"), "w") as f:
        json.dump(graph.to_dict(), f, indent=2)
    print(f"✅ Tick trace saved to {Config.output_path('tick_trace.json')}")

    inspection = graph.inspect_superpositions()
    with open(Config.output_path("inspection_log.json"), "w") as f:
        json.dump(inspection, f, indent=2)
    print(
        f"✅ Superposition inspection saved to {Config.output_path('inspection_log.json')}"
    )

    export_curvature_map()
    export_regional_maps()
    export_global_diagnostics()
    run_interpreter()


def export_curvature_map():
    """Aggregate curvature logs into a D3-friendly dataset."""
    grid = []
    try:
        with open(Config.output_path("curvature_log.json")) as f:
            for line in f:
                data = json.loads(line.strip())
                tick, edges = next(iter(data.items()))
                records = [
                    {
                        "source": k.split("->")[0],
                        "target": k.split("->")[1],
                        "delay": v["curved_delay"],
                    }
                    for k, v in edges.items()
                ]
                grid.append({"tick": int(tick), "edges": records})
    except FileNotFoundError:
        return

    with open(Config.output_path("curvature_map.json"), "w") as f:
        json.dump(grid, f, indent=2)
    print(f"✅ Curvature map exported to {Config.output_path('curvature_map.json')}")


def export_global_diagnostics():
    """Simple run-level metrics for diagnostics."""
    deco_lines = []
    try:
        with open(Config.output_path("decoherence_log.json")) as f:
            for line in f:
                deco_lines.append(json.loads(line))
    except FileNotFoundError:
        pass
    if not deco_lines:
        return
    first = next(iter(deco_lines[0].values()))
    last = next(iter(deco_lines[-1].values()))
    entropy_first = sum(first.values())
    entropy_last = sum(last.values())
    entropy_delta = entropy_last - entropy_first

    stability = []
    for entry in deco_lines:
        vals = list(entry.values())[0]
        stability.append(sum(1 for v in vals.values() if v < 0.4) / len(vals))
    coherence_stability_score = round(sum(stability) / len(stability), 3)

    collapse_events = 0
    try:
        with open(Config.output_path("classicalization_map.json")) as f:
            for line in f:
                states = json.loads(line)
                collapse_events += sum(
                    1 for v in next(iter(states.values())).values() if v
                )
    except FileNotFoundError:
        pass
    resilience = 0
    try:
        with open(Config.output_path("law_wave_log.json")) as f:
            resilience = sum(1 for _ in f)
    except FileNotFoundError:
        pass
    collapse_resilience_index = round(resilience / (collapse_events or 1), 3)

    adaptivity = 0
    if graph.bridges:
        active = sum(1 for b in graph.bridges if b.active)
        adaptivity = active / len(graph.bridges)

    diagnostics = {
        "coherence_stability_score": coherence_stability_score,
        "entropy_delta": round(entropy_delta, 3),
        "collapse_resilience_index": collapse_resilience_index,
        "network_adaptivity_index": round(adaptivity, 3),
    }
    with open(Config.output_path("global_diagnostics.json"), "w") as f:
        json.dump(diagnostics, f, indent=2)
    print("✅ Global diagnostics exported")


def export_regional_maps():
    regions = {}
    for nid, node in graph.nodes.items():
        region = nid[0]
        regions.setdefault(region, []).append(node)

    regional_pressure = {}
    for reg, nodes in regions.items():
        pressure = sum(n.collapse_pressure for n in nodes) / len(nodes)
        regional_pressure[reg] = round(pressure, 3)

    matrix = {}
    for edge in graph.edges:
        src_r = edge.source[0]
        tgt_r = edge.target[0]
        key = f"{src_r}->{tgt_r}"
        matrix[key] = matrix.get(key, 0) + 1

    with open(Config.output_path("regional_pressure_map.json"), "w") as f:
        json.dump(regional_pressure, f, indent=2)
    with open(Config.output_path("cluster_influence_matrix.json"), "w") as f:
        json.dump(matrix, f, indent=2)
    print("✅ Regional influence maps exported")
