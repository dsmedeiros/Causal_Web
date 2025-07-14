"""Causal explanation layer for interpreting CWT logs."""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


# ------------------------------------------------------------
# Helper to load newline-delimited JSON where each line maps a tick to values

def _load_json_lines(path: str) -> Dict[int, Dict]:
    records: Dict[int, Dict] = {}
    if not os.path.exists(path):
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            for k, v in obj.items():
                records[int(k)] = v
    return records


@dataclass
class ExplanationEvent:
    tick_range: Tuple[int, int]
    affected_nodes: List[str]
    rule_source: str
    explanation_text: str


class CausalAnalyst:
    """Analyze logs to generate causal explanations."""

    def __init__(self, output_dir: str = None, input_dir: str = None):
        base = os.path.join(os.path.dirname(__file__), "..")
        self.output_dir = output_dir or os.path.join(base, "output")
        self.input_dir = input_dir or os.path.join(base, "input")
        self.logs: Dict[str, Dict[int, Dict]] = {}
        self.graph: Dict = {}
        self.explanations: List[ExplanationEvent] = []

    # ------------------------------------------------------------
    def _path(self, name: str) -> str:
        return os.path.join(self.output_dir, name)

    # ------------------------------------------------------------
    def load_logs(self) -> None:
        """Load all required logs into memory."""
        paths = {
            "tick_trace": self._path("tick_trace.json"),
            "coherence": self._path("coherence_log.json"),
            "decoherence": self._path("decoherence_log.json"),
            "law_wave": self._path("law_wave_log.json"),
            "bridge_state": self._path("bridge_state_log.json"),
            "classicalization": self._path("classicalization_map.json"),
            "observer": self._path("observer_perceived_field.json"),
            "cluster": self._path("cluster_log.json"),
            "law_drift": self._path("law_drift_log.json"),
        }

        for key, path in paths.items():
            if path.endswith(".json") and not path.endswith("tick_trace.json"):
                self.logs[key] = _load_json_lines(path)
            elif key == "tick_trace" and os.path.exists(path):
                with open(path) as f:
                    try:
                        self.logs[key] = json.load(f)
                    except json.JSONDecodeError:
                        self.logs[key] = {}

        graph_path = os.path.join(self.input_dir, "graph.json")
        if os.path.exists(graph_path):
            with open(graph_path) as f:
                try:
                    self.graph = json.load(f)
                except json.JSONDecodeError:
                    self.graph = {}

    # ------------------------------------------------------------
    def detect_transitions(self) -> None:
        """Detect notable state transitions for later rule matching."""
        deco = self.logs.get("decoherence", {})
        spikes: Dict[str, List[Tuple[int, int, float]]] = {}
        streaks: Dict[str, List[float]] = {}
        for tick in sorted(deco.keys()):
            for node, val in deco[tick].items():
                lst = streaks.setdefault(node, [])
                if val > 0.5:
                    lst.append(val)
                else:
                    if len(lst) >= 3:
                        start = tick - len(lst)
                        spikes.setdefault(node, []).append((start, tick - 1, max(lst)))
                    streaks[node] = []
        # flush
        for node, lst in streaks.items():
            if len(lst) >= 3:
                start = max(deco.keys()) - len(lst) + 1
                spikes.setdefault(node, []).append((start, max(deco.keys()), max(lst)))
        self.transitions = {"decoherence_spikes": spikes}

    # ------------------------------------------------------------
    def _collapse_events(self) -> Dict[str, int]:
        collapse_log = self.logs.get("classicalization", {})
        events: Dict[str, int] = {}
        prev: Dict[str, bool] = {}
        for tick in sorted(collapse_log.keys()):
            states = collapse_log[tick]
            for node, state in states.items():
                if prev.get(node, False) != state and state:
                    events[node] = tick
                prev[node] = state
        return events

    # ------------------------------------------------------------
    def match_explanatory_rules(self) -> None:
        """Apply rule patterns to logs to create explanation events."""
        collapse_events = self._collapse_events()
        spikes = self.transitions.get("decoherence_spikes", {})

        for node, tick in collapse_events.items():
            # Rule: Decoherence-Induced Collapse
            candidate = [s for s in spikes.get(node, []) if s[1] >= tick - 1 and s[0] <= tick - 3]
            if candidate:
                start, end, mx = candidate[-1]
                text = (
                    f"Node {node} collapsed after sustained decoherence (max {mx:.2f}) "
                    f"between ticks {start}-{end}."
                )
                self.explanations.append(
                    ExplanationEvent((start, tick), [node], "rule:decoherence_induced_collapse", text)
                )

        # Rule: Law-Wave Stabilization
        law_wave = self.logs.get("law_wave", {})
        for node in {n for tick in law_wave for n in law_wave[tick]}:
            values = [law_wave[t][node] for t in sorted(law_wave) if node in law_wave[t]]
            window = []
            for i, v in enumerate(values):
                window.append(v)
                if len(window) > 5:
                    window.pop(0)
                if len(window) == 5:
                    avg = sum(window) / 5
                    var = sum((x - avg) ** 2 for x in window) / 5
                    if var < 0.01:
                        start_tick = sorted(law_wave.keys())[max(0, i - 4)]
                        end_tick = sorted(law_wave.keys())[i]
                        text = (
                            f"Node {node} maintained stable law-wave variance {var:.4f} "
                            f"from ticks {start_tick}-{end_tick}."
                        )
                        self.explanations.append(
                            ExplanationEvent((start_tick, end_tick), [node], "rule:law_wave_stabilization", text)
                        )
                        break

        # Rule: Meta-Node Onset
        cluster_log = self.logs.get("cluster", {})
        coherence = self.logs.get("coherence", {})
        for tick in sorted(cluster_log):
            clusters = cluster_log[tick]
            if not clusters:
                continue
            for cluster in clusters:
                if len(cluster) < 3:
                    continue
                if all(coherence.get(tick, {}).get(n, 0) > 0.85 for n in cluster):
                    text = (
                        f"Cluster {', '.join(cluster)} exhibited high coherence at tick {tick}, "
                        "indicating possible meta-node formation."
                    )
                    self.explanations.append(
                        ExplanationEvent((tick, tick), list(cluster), "rule:meta_node_onset", text)
                    )

    # ------------------------------------------------------------
    def generate_explanation_log(self) -> List[Dict]:
        data = [
            {
                "tick_range": list(e.tick_range),
                "affected_nodes": e.affected_nodes,
                "origin": e.rule_source,
                "explanation": e.explanation_text,
            }
            for e in self.explanations
        ]
        path = os.path.join(self.output_dir, "causal_explanations.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return data

    # ------------------------------------------------------------
    def generate_explanation_narrative(self) -> str:
        lines = []
        for e in self.explanations:
            rng = f"{e.tick_range[0]}-{e.tick_range[1]}" if e.tick_range[0] != e.tick_range[1] else str(e.tick_range[0])
            nodes = ", ".join(e.affected_nodes)
            lines.append(f"[{rng}] {nodes}: {e.explanation_text}")
        text = "\n".join(lines)
        path = os.path.join(self.output_dir, "causal_summary.txt")
        with open(path, "w") as f:
            f.write(text)
        return text

    # ------------------------------------------------------------
    def run(self) -> None:
        self.load_logs()
        self.detect_transitions()
        self.match_explanatory_rules()
        self.generate_explanation_log()
        self.generate_explanation_narrative()
        print("✅ Causal explanations generated")

