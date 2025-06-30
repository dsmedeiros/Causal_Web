import math
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

    def compute_phase(self, tick_time):
        return 2 * math.pi * self.frequency * tick_time

    def schedule_tick(self, tick_time, incoming_phase):
        self.incoming_phase_queue[tick_time].append(incoming_phase)
        print(f"[{self.id}] Received tick at {tick_time} with phase {incoming_phase:.2f}")

    def maybe_tick(self, global_tick):
        if global_tick in self.incoming_phase_queue:
            phases = self.incoming_phase_queue[global_tick]
            interference = sum(math.sin(p) for p in phases)
            threshold = 0.3
            if abs(interference) >= threshold:
                avg_phase = sum(phases) / len(phases)
                self.current_tick += 1
                self.tick_history.append((global_tick, avg_phase))
                print(f"[{self.id}] Tick at {global_tick} | Interference sum: {interference:.2f} | Phase: {avg_phase:.2f}")
                del self.incoming_phase_queue[global_tick]
                return True
            else:
                print(f"[{self.id}] Interference at {global_tick} cancelled tick (sum = {interference:.2f})")
                del self.incoming_phase_queue[global_tick]
        return False


class Edge:
    def __init__(self, source, target, delay=1):
        self.source = source
        self.target = target
        self.delay = delay
