"""Modular tick engine package."""

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
)
from .evaluator import (
    attach_graph as _attach_eval_graph,
    evaluate_nodes,
    mark_for_update,
    register_firing,
    reset_firing_limits,
    trigger_csp,
    check_propagation,
)
from .bridge_manager import dynamic_bridge_management
from .log_utils import (
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
]
