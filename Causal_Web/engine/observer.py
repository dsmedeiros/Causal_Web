class Observer:
    """Epistemic observer with limited perceptual window."""
    def __init__(self, observer_id: str, window: int = 10):
        self.id = observer_id
        self.window = window
        self.memory = []  # list of {'tick': int, 'events': [...]}

    def observe(self, graph, tick_time: int):
        events = []
        for node in graph.nodes.values():
            if node.tick_history and node.tick_history[-1].time == tick_time:
                events.append({
                    'node': node.id,
                    'phase': node.tick_history[-1].phase,
                    'time': tick_time
                })
        self.memory.append({'tick': tick_time, 'events': events})
        if len(self.memory) > self.window:
            self.memory.pop(0)
