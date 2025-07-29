from typing import Any, Dict, List


class Observer:
    """Epistemic observer with limited perceptual window."""

    def __init__(self, observer_id: str, window: int = 10) -> None:
        """Create a new ``Observer`` instance."""

        self.id = observer_id
        self.window = window
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
