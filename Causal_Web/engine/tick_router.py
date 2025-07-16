class TickRouter:
    """Route ticks across LCCM layers"""

    LAYERS = [
        "tick",
        "phase",
        "collapse",
        "bridge",
        "coherence",
        "decoherence",
        "law-wave",
    ]

    @classmethod
    def next_layer(cls, current: str) -> str:
        try:
            idx = cls.LAYERS.index(current)
            if idx < len(cls.LAYERS) - 1:
                return cls.LAYERS[idx + 1]
        except ValueError:
            pass
        return cls.LAYERS[-1]

    @classmethod
    def route_tick(cls, node, tick):
        from ..config import Config
        import json
        from .logger import log_json

        new_layer = cls.next_layer(tick.layer)
        if new_layer != tick.layer:
            record = {
                "tick": tick.time,
                "node": node.id,
                "from": tick.layer,
                "to": new_layer,
                "trace_id": tick.trace_id,
            }
            log_json(Config.output_path("layer_transition_log.json"), record)
            tick.layer = new_layer
