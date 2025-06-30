import math
import cmath
from collections import defaultdict

class Node:
    def __init__(self, node_id, x=0.0, y=0.0, frequency=1.0):
        self.id = node_id
        self.x = x
        self.y = y
        self.frequency = frequency
        self.tick_history = []  # [(tick_time, phase)]
        self.incoming_phase_queue = defaultdict(list)  # tick_time -> [phase_i]
        self.current_tick = 0
        self.subjective_ticks = 0  # For relativistic tracking
        self.last_emission_tick = None

    def compute_phase(self, tick_time):
        return 2 * math.pi * self.frequency * tick_time

    def schedule_tick(self, tick_time, incoming_phase):
        self.incoming_phase_queue[tick_time].append(incoming_phase)
        print(f"[{self.id}] Received tick at {tick_time} with phase {incoming_phase:.2f}")

    def should_tick(self, tick_time):
        phases = self.incoming_phase_queue[tick_time]
        vector_sum = sum(cmath.rect(abs(p), cmath.phase(p)) for p in phases)
        magnitude = abs(vector_sum)
        threshold = 0.5
        if magnitude >= threshold:
            avg_phase = sum(phases) / len(phases)
            return True, avg_phase
        return False, None

    def apply_tick(self, tick_time, phase):
        self.current_tick += 1
        self.subjective_ticks += 1
        self.tick_history.append((tick_time, phase))
        print(f"[{self.id}] Tick at {tick_time} | Phase: {phase:.2f}")

    def maybe_tick(self, global_tick):
        if global_tick in self.incoming_phase_queue:
            should_fire, phase = self.should_tick(global_tick)
            if should_fire:
                self.apply_tick(global_tick, phase)
            else:
                print(f"[{self.id}] Interference at {global_tick} cancelled tick")
            del self.incoming_phase_queue[global_tick]
            return should_fire
        return False

    def emit_tick_if_ready(self, global_tick):
        if self.last_emission_tick is None:
            self.last_emission_tick = global_tick
            self._emit(global_tick)
        elif (global_tick - self.last_emission_tick) >= (1.0 / self.frequency):
            self.last_emission_tick = global_tick
            self._emit(global_tick)

    def _emit(self, tick_time):
        phase = self.compute_phase(tick_time)
        self.tick_history.append((tick_time, phase))
        print(f"[{self.id}] Emitted tick at {tick_time} | Phase: {phase:.2f}")


class Edge:
    def __init__(self, source, target, attenuation, density, delay=1):
        self.source = source
        self.target = target
        self.attenuation = attenuation  # Multiplier for phase amplitude
        self.density = density          # Can affect delay dynamically
        self.delay = delay

    def adjusted_delay(self):
        # Optionally adjust delay based on density (example logic)
        return self.delay + int(self.density)

    def propagate_phase(self, phase, global_tick, graph):
        target_node = graph.get_node(self.target)
        attenuated_phase = phase * self.attenuation
        scheduled_tick = global_tick
        target_node.schedule_tick(scheduled_tick, attenuated_phase)
