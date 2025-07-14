import math
import cmath
from collections import defaultdict
from dataclasses import dataclass
from typing import Set, List
import numpy as np
import json


@dataclass
class Tick:
    """Discrete causal pulse."""
    origin: str
    time: float
    amplitude: float
    phase: float

class Node:
    def __init__(self, node_id, x=0.0, y=0.0, frequency=1.0, refractory_period=2, base_threshold=0.5, phase=0.0):
        self.id = node_id
        self.x = x
        self.y = y
        self.frequency = frequency
        self.phase = phase
        self.coherence = 1.0
        self.decoherence = 0.0
        self.tick_history = []  # List[Tick]
        self.incoming_phase_queue = defaultdict(list)  # tick_time -> [phase_i]
        self.pending_superpositions = defaultdict(list) # for logging and analysis
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
        self.coherence_series: List[float] = []
        self.law_wave_frequency: float = 0.0
        self.entangled_with: Set[str] = set()
        self.coherence_velocity: float = 0.0

    def compute_phase(self, tick_time):
        return 2 * math.pi * self.frequency * tick_time

    def compute_coherence_level(self, tick_time):
        phases = self.pending_superpositions.get(tick_time, [])
        if len(phases) < 2:
            self.coherence = 1.0
            self._update_law_wave()
            return self.coherence
        complex_vecs = [cmath.rect(1.0, p % (2 * math.pi)) for p in phases]
        vector_sum = sum(complex_vecs)
        self.coherence = abs(vector_sum) / len(phases)
        self._update_law_wave()
        return self.coherence

    def compute_decoherence_field(self, tick_time):
        phases = self.pending_superpositions.get(tick_time, [])
        if len(phases) < 2:
            self.decoherence = 0.0
            return self.decoherence
        normalized = [(p % (2 * math.pi)) for p in phases]
        mean_phase = sum(normalized) / len(normalized)
        variance = sum((p - mean_phase) ** 2 for p in normalized) / len(normalized)
        self.decoherence = variance
        return self.decoherence

    def _update_law_wave(self, window: int = 20):
        """Compute dominant coherence frequency over a sliding window."""
        self.coherence_series.append(self.coherence)
        if len(self.coherence_series) > window:
            self.coherence_series.pop(0)
        if len(self.coherence_series) > 1:
            arr = np.array(self.coherence_series[-window:])
            spectrum = np.fft.rfft(arr)
            if spectrum.size > 1:
                idx = int(np.argmax(np.abs(spectrum[1:])) + 1)
                self.law_wave_frequency = idx / window

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
        for tick in self.tick_history:
            if tick.time == tick_time:
                return tick.phase
        return None

    def apply_tick(self, tick_time, phase, graph, origin="self"):
        if self.is_classical:
            print(f"[{self.id}] Classical node cannot emit ticks")
            return

        if any(tick.time == tick_time for tick in self.tick_history):
            return

        self.current_tick += 1
        self.subjective_ticks += 1
        self.last_tick_time = tick_time
        self.current_threshold = min(self.current_threshold + 0.05, 1.0) # Slight adaptation
        self.phase = phase
        self.tick_history.append(Tick(origin=origin, time=tick_time, amplitude=1.0, phase=phase))
        self.collapse_origin[tick_time] = origin
        print(f"[{self.id}] Tick at {tick_time} via {origin.upper()} | Phase: {phase:.2f}")
        
        # Recursive phase propagation with refraction
        for edge in graph.get_edges_from(self.id):
            target = graph.get_node(edge.target)
            delay = edge.adjusted_delay(target.law_wave_frequency)
            attenuated = phase * edge.attenuation
            shifted = attenuated + edge.phase_shift
            target.schedule_tick(tick_time + delay, shifted)

        if origin == "self":
            collapsed = self.propagate_collapse(tick_time, graph)
            if collapsed:
                self._log_collapse_chain(tick_time, collapsed)

    def propagate_collapse(self, tick_time, graph, threshold: float = 0.5):
        """Propagate collapse to entangled nodes if their decoherence is high.

        Returns a list of node IDs that collapsed due to this propagation.
        """
        collapsed = []
        for nid in self.entangled_with:
            other = graph.get_node(nid)
            if not other or other.collapse_origin.get(tick_time):
                continue
            deco = other.compute_decoherence_field(tick_time)
            if deco > threshold:
                other.apply_tick(tick_time, self.phase, graph, origin="entanglement")
                collapsed.append(nid)
        return collapsed

    def _log_collapse_chain(self, tick_time, collapsed):
        """Log collapse propagation events to file."""
        record = {"tick": tick_time, "source": self.id, "collapsed": collapsed}
        with open("output/collapse_chain_log.json", "a") as f:
            f.write(json.dumps(record) + "\n")

    def maybe_tick(self, global_tick, graph):
        if global_tick in self.incoming_phase_queue:
            should_fire, phase = self.should_tick(global_tick)
            if should_fire and not self.is_classical:
                self.apply_tick(global_tick, phase, graph)
            else:
                print(f"[{self.id}] Interference at {global_tick} cancelled tick")
            del self.incoming_phase_queue[global_tick]
            if global_tick in self.pending_superpositions:
                del self.pending_superpositions[global_tick]
        else:
            self.current_threshold = max(self.base_threshold, self.current_threshold - 0.01)

    def emit_tick_if_ready(self, global_tick, graph):
        """
        Emits a self-tick only if the node is self-connected and not suppressed.
        Triggers real collapse via apply_tick().
        """
        if self.is_classical:
            return
        if self.id not in graph.get_upstream_nodes(self.id):
            return

        if global_tick - self.last_tick_time < self.refractory_period:
            return

        phase = self.compute_phase(global_tick)
        self.apply_tick(global_tick, phase, graph, origin="self")


    def _emit(self, tick_time):
        phase = self.compute_phase(tick_time)
        self.tick_history.append(Tick(origin="self", time=tick_time, amplitude=1.0, phase=phase))
        print(f"[{self.id}] Emitted tick at {tick_time} | Phase: {phase:.2f}")


class Edge:
    def __init__(self, source, target, attenuation, density, delay=1, phase_shift=0.0):
        self.source = source
        self.target = target
        self.attenuation = attenuation  # Multiplier for phase amplitude
        self.density = density          # Can affect delay dynamically
        self.delay = delay
        self.phase_shift = phase_shift

    def adjusted_delay(self, law_wave_freq: float = None):
        base = self.delay + int(self.density)
        if law_wave_freq is not None:
            modifier = math.sin(2 * math.pi * law_wave_freq) * self.density
            return base + modifier
        return base

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
        scheduled_tick = global_tick + self.adjusted_delay(target_node.law_wave_frequency)
        target_node.schedule_tick(scheduled_tick, attenuated_phase)
