from dataclasses import dataclass
import math
from random import random
from typing import Optional
import json

@dataclass
class BridgeEvent:
    tick: int
    event_type: str
    bridge_id: str
    source: str
    target: str
    coherence_at_event: Optional[float] = None

class Bridge:
    def __init__(self, 
                 node_a_id, 
                 node_b_id, 
                 bridge_type="braided", 
                 phase_offset=0.0,
                 drift_tolerance=None,
                 decoherence_limit=None):
        self.node_a_id = node_a_id
        self.node_b_id = node_b_id
        self.bridge_type = bridge_type  # "braided", "mirror", "unidirectional", etc.
        self.phase_offset = phase_offset  # For inverted or phase-shifted mirroring
        self.drift_tolerance = drift_tolerance
        self.decoherence_limit = decoherence_limit
        self.last_activation = None
        self.active = True
        self.last_rupture_tick = None
        self.last_reform_tick = None
        self.coherence_at_reform = None
        self.bridge_id = f"{self.node_a_id}->{self.node_b_id}"

    def _log_event(self, tick, event_type, value):
        event = BridgeEvent(
            tick=tick,
            event_type=event_type,
            bridge_id=self.bridge_id,
            source=self.node_a_id,
            target=self.node_b_id,
            coherence_at_event=value
        )
        with open("output/event_log.json", "a") as f:
            f.write(json.dumps(event.__dict__) + "\n")

    def probabilistic_bridge_failure(self, decoherence_strength, rupture_threshold=0.3, rupture_prob=0.9):
        if decoherence_strength > rupture_threshold and random() < rupture_prob:
            print(f"[BRIDGE] Probabilistic rupture at tick due to decoherence={decoherence_strength:.2f}")
            self.active = False
            return True
        return False

    def try_reactivate(self, tick_time, node_a, node_b, coherence_threshold=0.9):
        coherence = (node_a.compute_coherence_level(tick_time) + node_b.compute_coherence_level(tick_time)) / 2
        if not self.active and coherence > coherence_threshold:
            self.active = True
            self.last_reform_tick = tick_time
            self.coherence_at_reform = coherence
            print(f"[BRIDGE] Reactivated at tick {tick_time} with coherence={coherence:.2f}")
            self._log_event(tick_time, "bridge_reformed", coherence)

    def apply(self, tick_time, graph):
        node_a = graph.get_node(self.node_a_id)
        node_b = graph.get_node(self.node_b_id)

        self.try_reactivate(tick_time, node_a, node_b)

        a_collapsed = node_a.collapse_origin.get(tick_time) == "self"
        b_collapsed = node_b.collapse_origin.get(tick_time) == "self"

        if a_collapsed == b_collapsed:
            return

        if self.drift_tolerance is not None:
            phase_a = node_a.get_phase_at(tick_time)
            phase_b = node_b.get_phase_at(tick_time)
            if phase_a is not None and phase_b is not None:
                drift = abs((phase_a - phase_b + math.pi) % (2 * math.pi) - math.pi)
                if drift > self.drift_tolerance:
                    print(f"[BRIDGE] Drift too high at tick {tick_time}: {drift:.2f} > {self.drift_tolerance}")
                    self._log_event(tick_time, "bridge_drift", drift)
                    return

        decoherence_a = node_a.compute_decoherence_field(tick_time)
        decoherence_b = node_b.compute_decoherence_field(tick_time)
        avg_decoherence = (decoherence_a + decoherence_b) / 2

        if self.probabilistic_bridge_failure(avg_decoherence):
            self.last_rupture_tick = tick_time
            self._log_event(tick_time, "bridge_ruptured", avg_decoherence)
            return

        if self.decoherence_limit and self.last_activation is not None:
            if tick_time - self.last_activation > self.decoherence_limit:
                print(f"[BRIDGE] Decohered at tick {tick_time}, disabling bridge.")
                self.active = False
                self.last_rupture_tick = tick_time
                self._log_event(tick_time, "bridge_ruptured", avg_decoherence)
                return

        if not self.active:
            return

        if self.bridge_type == "braided":
            if a_collapsed:
                phase = node_a.get_phase_at(tick_time)
                node_b.apply_tick(tick_time, phase + self.phase_offset, graph, origin="bridge")
            elif b_collapsed:
                phase = node_b.get_phase_at(tick_time)
                node_a.apply_tick(tick_time, phase + self.phase_offset, graph, origin="bridge")
        elif self.bridge_type in {"unidirectional", "mirror"}:
            if a_collapsed:
                phase = node_a.get_phase_at(tick_time)
                node_b.apply_tick(tick_time, phase + self.phase_offset, graph, origin="bridge")

        self.last_activation = tick_time

    def to_dict(self):
        return {
            "source": self.node_a_id,
            "target": self.node_b_id,
            "type": self.bridge_type,
            "phase_offset": self.phase_offset,
            "drift_tolerance": self.drift_tolerance,
            "decoherence_limit": self.decoherence_limit,
            "active": self.active,
            "last_activation": self.last_activation,
            "last_rupture_tick": self.last_rupture_tick,
            "last_reform_tick": self.last_reform_tick,
            "coherence_at_reform": self.coherence_at_reform,
        }
