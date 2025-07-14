import math
import cmath
from collections import defaultdict

class Node:
    def __init__(self, node_id, x=0.0, y=0.0, frequency=1.0, refractory_period=2, base_threshold=0.5):
        self.id = node_id
        self.x = x
        self.y = y
        self.frequency = frequency
        self.tick_history = []  # [(tick_time, phase)]
        self.incoming_phase_queue = defaultdict(list)  # tick_time -> [phase_i]
        self.pending_superpositions = defaultdict(list) # for logging and anaysis
        self.current_tick = 0
        self.subjective_ticks = 0  # For relativistic tracking
        self.last_emission_tick = None
        self.refractory_period = refractory_period
        self.last_tick_time = -math.inf
        self.base_threshold = base_threshold
        self.current_threshold = self.base_threshold
        self.collapse_origin = {} # tick_time -> "self" or "bridge"
        self._decoherence_streak = 0
        self.is_classical = False

    def compute_phase(self, tick_time):
        return 2 * math.pi * self.frequency * tick_time

    def compute_coherence_level(self, tick_time):
        phases = self.pending_superpositions.get(tick_time, [])
        if len(phases) < 2:
            return 1.0  # fully coherent by default
        complex_vecs = [cmath.rect(1.0, p % (2 * math.pi)) for p in phases]
        vector_sum = sum(complex_vecs)
        return abs(vector_sum) / len(phases)

    def compute_decoherence_field(self, tick_time):
        phases = self.pending_superpositions.get(tick_time, [])
        if len(phases) < 2:
            return 0.0
        normalized = [(p % (2 * math.pi)) for p in phases]
        mean_phase = sum(normalized) / len(normalized)
        variance = sum((p - mean_phase) ** 2 for p in normalized) / len(normalized)
        return variance

    def update_classical_state(self, decoherence_strength, threshold=0.4, streak_required=2):
        if decoherence_strength > threshold:
            self._decoherence_streak += 1
        else:
            self._decoherence_streak = 0

        if self._decoherence_streak >= streak_required:
            self.is_classical = True

    def schedule_tick(self, tick_time, incoming_phase):
        self.incoming_phase_queue[tick_time].append(incoming_phase)
        self.pending_superpositions[tick_time].append(incoming_phase) # track unresolved states
        print(f"[{self.id}] Received tick at {tick_time} with phase {incoming_phase:.2f}")

    def should_tick(self, tick_time):
        if tick_time - self.last_tick_time < self.refractory_period:
            print(f"[{self.id}] Suppressed by refractory period at {tick_time}")
            return False, None

        raw_phases = self.incoming_phase_queue[tick_time]
        complex_phases = [cmath.rect(1.0, p % (2 * math.pi)) for p in raw_phases]
        vector_sum = sum(complex_phases)
        magnitude = abs(vector_sum)

        if magnitude >= self.current_threshold:
            resultant_phase = cmath.phase(vector_sum)
            return True, resultant_phase

        return False, None

    def get_phase_at(self, tick_time):
        for t, p in self.tick_history:
            if t == tick_time:
                return p
        return None

    def apply_tick(self, tick_time, phase, graph, origin="self"):
        if any(t == tick_time for t, _ in self.tick_history):
            return

        self.current_tick += 1
        self.subjective_ticks += 1
        self.last_tick_time = tick_time
        self.current_threshold = min(self.current_threshold + 0.05, 1.0) # Slight adaptation
        self.tick_history.append((tick_time, phase))
        self.collapse_origin[tick_time] = origin
        print(f"[{self.id}] Tick at {tick_time} via {origin.upper()} | Phase: {phase:.2f}")
        
        # Recursive phase propagation
        for edge in graph.get_edges_from(self.id):
            delay = edge.adjusted_delay()
            attenuated = phase * edge.attenuation
            shifted = attenuated + edge.phase_shift
            target = graph.get_node(edge.target)
            target.schedule_tick(tick_time + delay, shifted)

    def maybe_tick(self, global_tick, graph):
        if global_tick in self.incoming_phase_queue:
            should_fire, phase = self.should_tick(global_tick)
            if should_fire:
                self.apply_tick(global_tick, phase, graph)
            else:
                print(f"[{self.id}] Interference at {global_tick} cancelled tick")
            del self.incoming_phase_queue[global_tick]
            # del self.pending_superpositions[global_tick]  # Clear resolved state
        else:
            self.current_threshold = max(self.base_threshold, self.current_threshold - 0.01)

    def emit_tick_if_ready(self, global_tick, graph):
        """
        Emits a self-tick only if the node is self-connected and not suppressed.
        Triggers real collapse via apply_tick().
        """
        if self.id not in graph.get_upstream_nodes(self.id):
            return

        if global_tick - self.last_tick_time < self.refractory_period:
            return

        phase = self.compute_phase(global_tick)
        self.apply_tick(global_tick, phase, graph, origin="self")


    def _emit(self, tick_time):
        phase = self.compute_phase(tick_time)
        self.tick_history.append((tick_time, phase))
        print(f"[{self.id}] Emitted tick at {tick_time} | Phase: {phase:.2f}")


class Edge:
    def __init__(self, source, target, attenuation, density, delay=1, phase_shift=0.0):
        self.source = source
        self.target = target
        self.attenuation = attenuation  # Multiplier for phase amplitude
        self.density = density          # Can affect delay dynamically
        self.delay = delay
        self.phase_shift = phase_shift

    def adjusted_delay(self):
        # Optionally adjust delay based on density (example logic)
        return self.delay + int(self.density)

    def propagate_phase(self, phase, global_tick, graph):
        target_node = graph.get_node(self.target)

        drift_phase = 0.0
        if isinstance(self.phase_shift, dict):
            mode = self.phase_shift.get("mode")
            if mode == "linear":
                rate = self.phase_shift.get("rate", 0.1)
                drift_phase = rate * global_tick
            elif mode == "nonlinear":
                amp = self.phase_shift.get("amplitude", math.pi)
                period = self.phase_shift.get("period", 60)
                drift_phase = amp * math.sin((2 * math.pi / period) * global_tick)
            else:
                raise ValueError(f"Unknown phase shift mode: {mode}")
        else:
            drift_phase = self.phase_shift # static phase shift

        shifted_phase = phase + drift_phase
        attenuated_phase = shifted_phase * self.attenuation
        scheduled_tick = global_tick + self.adjusted_delay()
        target_node.schedule_tick(scheduled_tick, attenuated_phase)
