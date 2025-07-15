from dataclasses import dataclass
import math
from random import random
from typing import Optional
import json

from enum import Enum


class BridgeState(Enum):
    FORMING = "forming"
    STABLE = "stable"
    STRAINED = "strained"
    RUPTURING = "rupturing"
    RUPTURED = "ruptured"
    DORMANT = "dormant"

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
                 decoherence_limit=None,
                 initial_strength=1.0,
                 medium_type="standard",
                 mutable=True,
                 seeded=True,
                 formed_at_tick=0):
        self.node_a_id = node_a_id
        self.node_b_id = node_b_id
        self.bridge_type = bridge_type  # "braided", "mirror", "unidirectional", etc.
        self.phase_offset = phase_offset  # For inverted or phase-shifted mirroring
        self.drift_tolerance = drift_tolerance
        self.decoherence_limit = decoherence_limit
        self.initial_strength = initial_strength
        self.current_strength = initial_strength
        self.medium_type = medium_type
        self.mutable = mutable
        self.seeded = seeded
        self.formed_at_tick = formed_at_tick
        self.tick_load = 0
        self.phase_drift = 0.0
        self.coherence_flux = 0.0
        self.last_activation = None
        self.active = True
        self.state = BridgeState.FORMING
        self.fatigue = 0.0
        self.last_rupture_tick = None
        self.last_reform_tick = None
        self.coherence_at_reform = None
        self.bridge_id = f"{self.node_a_id}->{self.node_b_id}"

        # ---- Phase 4 additions ----
        self.rupture_history = []  # list of (tick, decoherence)
        self.reinforcement_streak = 0
        self.decoherence_exposure = []  # recent avg decoherence values
        self.trust_score = 0.5

        # ---- Phase 6 additions ----
        self.last_active_tick = formed_at_tick
        self.reformable = True
        self.memory_weight = 0.5

    def _log_dynamics(self, tick, event, conditions=None):
        record = {
            "bridge_id": self.bridge_id,
            "source": self.node_a_id,
            "target": self.node_b_id,
            "event": event,
            "tick": tick,
            "seeded": self.seeded
        }
        if conditions is not None:
            record["conditions"] = conditions
        with open("output/bridge_dynamics_log.json", "a") as f:
            f.write(json.dumps(record) + "\n")

    def update_state(self, tick: int) -> None:
        old = self.state
        if not self.active:
            if self.current_strength > 0 and self.fatigue <= 3.0:
                self.state = BridgeState.DORMANT
            else:
                self.state = BridgeState.RUPTURED
        else:
            avg = sum(self.decoherence_exposure[-5:]) / len(self.decoherence_exposure[-5:]) if self.decoherence_exposure else 0.0
            self.fatigue += avg
            if self.fatigue > 3.0:
                self.state = BridgeState.RUPTURING
                self.active = False
            elif self.fatigue > 1.5:
                self.state = BridgeState.STRAINED
            else:
                self.state = BridgeState.STABLE
        if old != self.state:
            self._log_dynamics(tick, self.state.value, {"from": old.value})

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

    # ---- Phase 6: plasticity ----
    def decay(self, tick_time, inactive_threshold=5):
        if self.last_active_tick is None:
            self.last_active_tick = tick_time
        if tick_time - self.last_active_tick >= inactive_threshold and self.current_strength > 0:
            self.current_strength = max(0.0, self.current_strength - 0.1)
            duration = tick_time - self.last_active_tick
            with open("output/bridge_decay_log.json", "a") as f:
                f.write(json.dumps({"tick": tick_time, "bridge": self.bridge_id, "strength": self.current_strength, "duration": duration}) + "\n")
            if self.current_strength == 0.0:
                self.active = False
                import engine.tick_engine as te
                te._decay_durations.append(duration)
                self._log_dynamics(tick_time, "decayed")

    def try_reform(self, tick_time, node_a, node_b, coherence_threshold=0.9):
        if not self.reformable or self.active:
            return
        coherence = (node_a.compute_coherence_level(tick_time) + node_b.compute_coherence_level(tick_time)) / 2
        if coherence > coherence_threshold and random() < self.memory_weight:
            self.active = True
            self.last_reform_tick = tick_time
            self.coherence_at_reform = coherence
            with open("output/bridge_reformation_log.json", "a") as f:
                f.write(json.dumps({"tick": tick_time, "bridge": self.bridge_id, "coherence": coherence}) + "\n")
            import engine.tick_engine as te
            te.bridges_reformed_count += 1
            self._log_event(tick_time, "bridge_reformed", coherence)
            self._log_dynamics(tick_time, "recovered", {"coherence": coherence})

    def try_reactivate(self, tick_time, node_a, node_b, coherence_threshold=0.9):
        coherence = (node_a.compute_coherence_level(tick_time) + node_b.compute_coherence_level(tick_time)) / 2
        if not self.active and coherence > coherence_threshold:
            self.active = True
            self.last_reform_tick = tick_time
            self.coherence_at_reform = coherence
            print(f"[BRIDGE] Reactivated at tick {tick_time} with coherence={coherence:.2f}")
            self._log_event(tick_time, "bridge_reformed", coherence)
            self._log_dynamics(tick_time, "recovered", {"coherence": coherence})

    def apply(self, tick_time, graph):
        node_a = graph.get_node(self.node_a_id)
        node_b = graph.get_node(self.node_b_id)

        self.decay(tick_time)
        self.try_reform(tick_time, node_a, node_b)

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
                    self.trust_score = max(0.0, self.trust_score - 0.05)
                    self.reinforcement_streak = 0
                    return

        decoherence_a = node_a.compute_decoherence_field(tick_time)
        decoherence_b = node_b.compute_decoherence_field(tick_time)
        avg_decoherence = (decoherence_a + decoherence_b) / 2
        debt_influence = (node_a.decoherence_debt + node_b.decoherence_debt) / 6.0
        rupture_chance = avg_decoherence + debt_influence
        self.decoherence_exposure.append(avg_decoherence)
        if len(self.decoherence_exposure) > 20:
            self.decoherence_exposure.pop(0)

        if self.probabilistic_bridge_failure(rupture_chance):
            self.last_rupture_tick = tick_time
            self._log_event(tick_time, "bridge_ruptured", avg_decoherence)
            self._log_dynamics(tick_time, "ruptured", {"decoherence": avg_decoherence})
            self.rupture_history.append((tick_time, avg_decoherence))
            self.trust_score = max(0.0, self.trust_score - 0.1)
            self.reinforcement_streak = 0
            self.update_state(tick_time)
            return

        if self.decoherence_limit and self.last_activation is not None:
            if tick_time - self.last_activation > self.decoherence_limit:
                print(f"[BRIDGE] Decohered at tick {tick_time}, disabling bridge.")
                self.active = False
                self.last_rupture_tick = tick_time
                self._log_event(tick_time, "bridge_ruptured", avg_decoherence)
                self._log_dynamics(tick_time, "ruptured", {"decoherence": avg_decoherence})
                self.rupture_history.append((tick_time, avg_decoherence))
                self.trust_score = max(0.0, self.trust_score - 0.1)
                self.reinforcement_streak = 0
                self.update_state(tick_time)
                return

        if not self.active:
            self.update_state(tick_time)
            return

        if self.bridge_type == "braided":
            if a_collapsed:
                phase = node_a.get_phase_at(tick_time)
                node_b.apply_tick(tick_time, phase + self.phase_offset, graph, origin="bridge")
                node_a.entangled_with.add(node_b.id)
                node_b.entangled_with.add(node_a.id)
            elif b_collapsed:
                phase = node_b.get_phase_at(tick_time)
                node_a.apply_tick(tick_time, phase + self.phase_offset, graph, origin="bridge")
                node_a.entangled_with.add(node_b.id)
                node_b.entangled_with.add(node_a.id)
        elif self.bridge_type in {"unidirectional", "mirror"}:
            if a_collapsed:
                phase = node_a.get_phase_at(tick_time)
                node_b.apply_tick(tick_time, phase + self.phase_offset, graph, origin="bridge")
                node_a.entangled_with.add(node_b.id)
                node_b.entangled_with.add(node_a.id)
        if self.active:
            self.tick_load += 1
            if phase_a is not None and phase_b is not None:
                self.phase_drift += abs((phase_a - phase_b + math.pi) % (2 * math.pi) - math.pi)
            self.coherence_flux += (node_a.coherence + node_b.coherence) / 2
        self.last_activation = tick_time
        if self.active:
            self.last_active_tick = tick_time
        self.reinforcement_streak += 1
        self.trust_score = min(1.0, self.trust_score + 0.01)
        self.update_state(tick_time)

    def to_dict(self):
        return {
            "source": self.node_a_id,
            "target": self.node_b_id,
            "type": self.bridge_type,
            "phase_offset": self.phase_offset,
            "drift_tolerance": self.drift_tolerance,
            "decoherence_limit": self.decoherence_limit,
            "initial_strength": self.initial_strength,
            "current_strength": self.current_strength,
            "medium_type": self.medium_type,
            "mutable": self.mutable,
            "seeded": self.seeded,
            "formed_at_tick": self.formed_at_tick,
            "active": self.active,
            "last_activation": self.last_activation,
            "last_rupture_tick": self.last_rupture_tick,
            "last_reform_tick": self.last_reform_tick,
            "coherence_at_reform": self.coherence_at_reform,
            "trust_score": self.trust_score,
            "reinforcement_streak": self.reinforcement_streak,
            "last_active_tick": self.last_active_tick,
            "reformable": self.reformable,
            "memory_weight": self.memory_weight,
            "avg_decoherence": sum(self.decoherence_exposure)/len(self.decoherence_exposure) if self.decoherence_exposure else 0.0,
            "state": self.state.value,
            "fatigue": round(self.fatigue, 3),
            "tick_load": self.tick_load,
            "phase_drift": round(self.phase_drift, 4),
            "coherence_flux": round(self.coherence_flux, 4),
        }
