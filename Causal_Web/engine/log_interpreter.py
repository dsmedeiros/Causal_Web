import json
import os
from typing import Dict, List


def _load_json_lines(path: str) -> Dict[str, Dict]:
    data = {}
    if not os.path.exists(path):
        return data
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                data.update(obj)
            except json.JSONDecodeError:
                continue
    return data


class CWTLogInterpreter:
    """Simple multilayer interpreter for CWT logs."""

    def __init__(self, output_dir: str = None, graph_path: str = None):
        base = os.path.join(os.path.dirname(__file__), "..")
        self.output_dir = output_dir or os.path.join(base, "output")
        self.graph_path = graph_path or os.path.join(base, "input", "graph.json")
        self.graph = {}
        self.summary: Dict[str, Dict] = {}

    # ------------------------------------------------------------
    def load_graph(self) -> None:
        if os.path.exists(self.graph_path):
            with open(self.graph_path) as f:
                self.graph = json.load(f)
        else:
            self.graph = {}

    # ------------------------------------------------------------
    def interpret_curvature(self) -> None:
        path = os.path.join(self.output_dir, "curvature_map.json")
        if not os.path.exists(path):
            return
        with open(path) as f:
            records = json.load(f)
        stats: Dict[str, Dict[str, float]] = {}
        for entry in records:
            for edge in entry.get("edges", []):
                key = f"{edge['source']}->{edge['target']}"
                stats.setdefault(key, {"min": float("inf"), "max": float("-inf")})
                d = edge.get("delay", 0)
                if d < stats[key]["min"]:
                    stats[key]["min"] = d
                if d > stats[key]["max"]:
                    stats[key]["max"] = d
        self.summary["curvature"] = {
            k: {"delay_range": [v["min"], v["max"]]} for k, v in stats.items()
        }

    # ------------------------------------------------------------
    def interpret_collapse(self) -> None:
        path = os.path.join(self.output_dir, "classicalization_map.json")
        lines = _load_json_lines(path)
        if not lines:
            return
        collapse: Dict[str, int] = {}
        prev: Dict[str, bool] = {}
        for tick_str, states in lines.items():
            tick = int(tick_str)
            for node, state in states.items():
                if prev.get(node, False) != state and state:
                    collapse[node] = tick
                prev[node] = state
        if collapse:
            self.summary["collapse"] = collapse

    # ------------------------------------------------------------
    def interpret_coherence(self) -> None:
        path = os.path.join(self.output_dir, "coherence_log.json")
        lines = _load_json_lines(path)
        if not lines:
            return
        summary: Dict[str, Dict[str, float]] = {}
        for states in lines.values():
            for node, coh in states.items():
                entry = summary.setdefault(node, {"max": -1.0, "min": 2.0})
                if coh > entry["max"]:
                    entry["max"] = coh
                if coh < entry["min"]:
                    entry["min"] = coh
        self.summary["coherence"] = summary

    # ------------------------------------------------------------
    def interpret_law_wave(self) -> None:
        path = os.path.join(self.output_dir, "law_wave_log.json")
        lines = _load_json_lines(path)
        if not lines:
            return
        freqs: Dict[str, List[float]] = {}
        for states in lines.values():
            for node, freq in states.items():
                freqs.setdefault(node, []).append(freq)
        summary = {}
        for node, arr in freqs.items():
            if not arr:
                continue
            avg = sum(arr) / len(arr)
            var = sum((x - avg) ** 2 for x in arr) / len(arr)
            summary[node] = {"mean": round(avg, 4), "variance": round(var, 6)}
        self.summary["law_wave"] = summary

    # ------------------------------------------------------------
    def run(self) -> None:
        self.load_graph()
        self.interpret_curvature()
        self.interpret_collapse()
        self.interpret_coherence()
        self.interpret_law_wave()
        out_path = os.path.join(self.output_dir, "interpretation_log.json")
        with open(out_path, "w") as f:
            json.dump(self.summary, f, indent=2)
        print(f"✅ Interpretation saved to {out_path}")


def run_interpreter() -> None:
    interpreter = CWTLogInterpreter()
    interpreter.run()


if __name__ == "__main__":
    run_interpreter()
