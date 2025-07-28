"""Service objects related to :class:`Node` lifecycle."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Optional, List, Dict, Set, TYPE_CHECKING
import threading


from ..config import Config

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from .node import Node


class NodeInitializationService:
    """Initialize a :class:`~Causal_Web.engine.node.Node` instance."""

    def __init__(self, node: Node) -> None:
        self.node = node

    # ------------------------------------------------------------------
    def setup(
        self,
        node_id: str,
        x: float = 0.0,
        y: float = 0.0,
        frequency: float = 1.0,
        refractory_period: float | None = None,
        base_threshold: float = 0.5,
        phase: float = 0.0,
        *,
        origin_type: str = "seed",
        generation_tick: int = 0,
        parent_ids: Optional[List[str]] = None,
    ) -> None:
        self._basic(node_id, x, y, frequency, phase)
        self._runtime_state(refractory_period, base_threshold)
        self._cluster_metadata()
        self._propagation_metadata(origin_type, generation_tick, parent_ids)
        self._phase_four()
        self._threshold_params()
        self._spatial_index()

    # ------------------------------------------------------------------
    def _basic(
        self, node_id: str, x: float, y: float, frequency: float, phase: float
    ) -> None:
        n = self.node
        n.id = node_id
        n.x = x
        n.y = y
        n.frequency = frequency
        n.phase = phase
        n.coherence = 1.0
        n.decoherence = 0.0

    # ------------------------------------------------------------------
    def _runtime_state(
        self, refractory_period: float | None, base_threshold: float
    ) -> None:
        n = self.node
        n.tick_history = []
        n.emitted_tick_times: Set[float] = set()
        n.received_tick_times: Set[float] = set()
        n._tick_phase_lookup: Dict[int, float] = {}
        n.incoming_phase_queue = defaultdict(list)
        n.incoming_tick_counts = defaultdict(int)
        n.pending_superpositions = defaultdict(list)
        n._phase_cache: Dict[int, float] = {}
        n._coherence_cache: Dict[int, float] = {}
        n._decoherence_cache: Dict[int, float] = {}
        n.lock = threading.Lock()
        n.current_tick = 0
        n.subjective_ticks = 0
        n.last_emission_tick = None
        if refractory_period is None:
            refractory_period = getattr(Config, "refractory_period", 2.0)
        n.refractory_period = refractory_period
        n.last_tick_time = None
        n.base_threshold = base_threshold
        n.current_threshold = n.base_threshold
        n.collapse_origin = {}
        n._decoherence_streak = 0
        n.is_classical = False
        n.coherence_series: List[float] = []
        n.law_wave_frequency = 0.0
        n.entangled_with: Set[str] = set()
        n.coherence_velocity = 0.0
        from .node import NodeType

        n.node_type: NodeType = NodeType.NORMAL
        n.prev_node_type: NodeType = NodeType.NORMAL
        n.coherence_credit = 0.0
        n.decoherence_debt = 0.0
        n.phase_lock = False
        n.collapse_pressure = 0.0
        n.tick_drop_counts = defaultdict(int)

    # ------------------------------------------------------------------
    def _cluster_metadata(self) -> None:
        self.node.cluster_ids: Dict[int, int] = {}

    # ------------------------------------------------------------------
    def _propagation_metadata(
        self, origin_type: str, generation_tick: int, parent_ids: Optional[List[str]]
    ) -> None:
        n = self.node
        n.origin_type = origin_type
        n.generation_tick = generation_tick
        n.parent_ids = parent_ids or []
        n.sip_streak = 0

    # ------------------------------------------------------------------
    def _phase_four(self) -> None:
        n = self.node
        n.memory_window = getattr(Config, "memory_window", 20)
        n.memory: Dict[str, deque] = {
            "origins": deque(maxlen=n.memory_window),
            "coherence": deque(maxlen=n.memory_window),
            "decoherence": deque(maxlen=n.memory_window),
        }
        n.trust_profile: Dict[str, float] = {}
        n.phase_confidence_index = 1.0
        n.goals: Dict[str, float] = {}
        n.goal_error: Dict[str, float] = {}

    # ------------------------------------------------------------------
    def _threshold_params(self) -> None:
        n = self.node
        n.initial_coherence_threshold = getattr(
            Config, "initial_coherence_threshold", 0.6
        )
        n.steady_coherence_threshold = getattr(
            Config, "steady_coherence_threshold", 0.85
        )
        n.coherence_ramp_ticks = getattr(Config, "coherence_ramp_ticks", 10)
        n.dynamic_offset = 0.0

    # ------------------------------------------------------------------
    def _spatial_index(self) -> None:
        cell_size = getattr(Config, "SPATIAL_GRID_SIZE", 50)
        self.node.grid_x = int(self.node.x // cell_size)
        self.node.grid_y = int(self.node.y // cell_size)
