from typing import Any, Dict, List
import random
import math

from ..logging.logger import log_json


class Observer:
    """Epistemic observer with limited perceptual window."""

    def __init__(
        self,
        observer_id: str,
        window: int = 10,
        *,
        detector_mode: bool = False,
        measurement_settings: List[float] | None = None,
    ) -> None:
        """Create a new ``Observer`` instance.

        Parameters
        ----------
        observer_id:
            Unique identifier for the observer.
        window:
            Number of recent ticks retained in memory.
        detector_mode:
            When ``True`` the observer performs binary measurements on
            entangled ticks.
        measurement_settings:
            Optional list of angles used when sampling measurement settings.
            Defaults to ``[0.0, math.pi / 4]`` when not provided.
        """

        self.id = observer_id
        self.window = window
        self.detector_mode = detector_mode
        self.measurement_settings = (
            measurement_settings[:]
            if measurement_settings is not None
            else [0.0, math.pi / 4]
        )
        self.memory: List[Dict[str, Any]] = []
        self.belief_state: Dict[int, Dict[str, int]] = {}
        self.disagreement: List[Any] = []

    def observe(self, graph: Any, tick_time: int) -> None:
        """Record events at ``tick_time`` from ``graph``."""

        events: List[Dict[str, Any]] = []
        seen_nodes = set()
        for node in graph.nodes.values():
            if node.tick_history and node.tick_history[-1].time == tick_time:
                events.append(
                    {
                        "node": node.id,
                        "phase": node.tick_history[-1].phase,
                        "tick_id": node.tick_history[-1].trace_id,
                        "entangled_id": getattr(
                            node.tick_history[-1], "entangled_id", None
                        ),
                        "time": tick_time,
                        "inferred": False,
                    }
                )
                seen_nodes.add(node.id)

        # infer unseen upstream events
        for ev in list(events):
            upstream = graph.get_upstream_nodes(ev["node"])
            for up in upstream:
                if up not in seen_nodes:
                    events.append(
                        {"node": up, "phase": None, "time": tick_time, "inferred": True}
                    )
                    seen_nodes.add(up)

        self.belief_state.setdefault(tick_time, {})
        for ev in events:
            self.belief_state[tick_time].setdefault(ev["node"], 0)
            self.belief_state[tick_time][ev["node"]] += 1
            ent_id = ev.get("entangled_id")
            if ent_id and self.detector_mode:
                seed = f"{ent_id}-{self.id}-{ev['tick_id']}"
                rng = random.Random(seed)
                setting = rng.choice(self.measurement_settings)
                adjusted = ev["phase"] - setting
                outcome = 1 if math.cos(adjusted) > 0 else -1
                log_json(
                    "event",
                    "entangled_measurement",
                    {
                        "tick_id": ev["tick_id"],
                        "observer_id": self.id,
                        "entangled_id": ent_id,
                        "measurement_setting": setting,
                        "binary_outcome": outcome,
                    },
                    tick=tick_time,
                )
        self.memory.append({"tick": tick_time, "events": events})
        if len(self.memory) > self.window:
            self.memory.pop(0)

    def infer_field_state(self) -> Dict[str, int]:
        """Return a naive belief state derived from recent events."""

        state: Dict[str, int] = {}
        for entry in self.memory:
            for ev in entry["events"]:
                state.setdefault(ev["node"], 0)
                state[ev["node"]] += 1
        return state
