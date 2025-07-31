"""Modular tick engine package.

This module exposes the public API of the tick engine and initialises a
number of diagnostic counters accessed by other services. These counters are
used during headless runs where the GUI is not active and therefore must
exist at import time.
"""

boundary_interactions_count = 0
"""Number of times ticks interacted with boundary nodes."""

void_absorption_events = 0
"""Ticks absorbed by void nodes."""

bridges_reformed_count = 0
"""Count of bridges reactivated after rupture."""

_law_wave_stability: dict[str, dict] = {}
"""Internal record of node law-wave frequency stability."""

_decay_durations: list[int] = []
"""Durations for which bridges decayed until deactivation."""

from .core import (
    SimulationRunner,
    add_observer,
    build_graph,
    clear_output_directory,
    emit_ticks,
    graph,
    kappa,
    pause_simulation,
    resume_simulation,
    stop_simulation,
    simulation_loop,
    _update_simulation_state,
)
from .orchestrators import EvaluationOrchestrator, MutationOrchestrator, IOOrchestrator
from .evaluator import (
    attach_graph as _attach_eval_graph,
    evaluate_nodes,
    mark_for_update,
    register_firing,
    reset_firing_limits,
    trigger_csp,
    check_propagation,
)
from . import bridge_manager
from ..logging import log_utils
from .bridge_manager import dynamic_bridge_management
from ..logging.log_utils import (
    log_bridge_states,
    log_curvature_per_tick,
    log_meta_node_ticks,
    log_metrics_per_tick,
    snapshot_graph,
    write_output,
    export_curvature_map,
    export_global_diagnostics,
    export_regional_maps,
)

# ensure submodules share the global graph
_attach_eval_graph(graph)
bridge_manager.attach_graph(graph)
log_utils.attach_graph(graph)

__all__ = [
    "SimulationRunner",
    "add_observer",
    "build_graph",
    "clear_output_directory",
    "emit_ticks",
    "graph",
    "kappa",
    "pause_simulation",
    "resume_simulation",
    "stop_simulation",
    "simulation_loop",
    "_update_simulation_state",
    "evaluate_nodes",
    "mark_for_update",
    "register_firing",
    "reset_firing_limits",
    "trigger_csp",
    "check_propagation",
    "dynamic_bridge_management",
    "log_bridge_states",
    "log_curvature_per_tick",
    "log_meta_node_ticks",
    "log_metrics_per_tick",
    "snapshot_graph",
    "write_output",
    "export_curvature_map",
    "export_global_diagnostics",
    "export_regional_maps",
    "EvaluationOrchestrator",
    "MutationOrchestrator",
    "IOOrchestrator",
]
