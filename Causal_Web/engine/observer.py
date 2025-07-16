class Observer:
    """Epistemic observer with limited perceptual window."""

    def __init__(self, observer_id: str, window: int = 10):
        self.id = observer_id
        self.window = window
        self.memory = []  # list of {'tick': int, 'events': [...]}
        self.belief_state = {}
        self.disagreement = []  # list of diffs per tick

    def observe(self, graph, tick_time: int):
        events = []
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

    def infer_field_state(self):
        """Simple inference based on memory of recent events."""
        state = {}
        for entry in self.memory:
            for ev in entry["events"]:
                state.setdefault(ev["node"], 0)
                state[ev["node"]] += 1
        return state
