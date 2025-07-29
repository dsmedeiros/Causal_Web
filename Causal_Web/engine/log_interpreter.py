import json
import os
from typing import Dict, List

from .causal_analyst import CausalAnalyst
from .base import OutputDirMixin, JsonLinesMixin
from ..config import Config


class CWTLogInterpreter(OutputDirMixin, JsonLinesMixin):
    """Simple multilayer interpreter for CWT logs."""

    def __init__(
        self, output_dir: str | None = None, graph_path: str | None = None
    ) -> None:
        super().__init__(output_dir=output_dir)
        base = os.path.join(os.path.dirname(__file__), "..")
        self.graph_path = graph_path or Config.graph_file
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
        """Aggregate curvature statistics from the log file."""
        path = self._path("curvature_map.json")
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return
        try:
            with open(path) as f:
                records = json.load(f)
        except json.JSONDecodeError:
            return
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
        path = self._path("classicalization_map.json")
        lines = self.load_json_lines(path)
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
        path = self._path("coherence_log.json")
        lines = self.load_json_lines(path)
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
        path = self._path("law_wave_log.json")
        lines = self.load_json_lines(path)
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
    def interpret_collapse_origins(self) -> None:
        """Summarize where collapses were first triggered."""
        path = self._path("collapse_front_log.json")
        if not os.path.exists(path):
            return
        origins: Dict[str, int] = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tick = rec.get("tick")
                if rec.get("event") == "collapse_start":
                    node = rec.get("node")
                    if node and node not in origins:
                        origins[node] = tick
        if origins:
            self.summary["collapse_origins"] = origins

    # ------------------------------------------------------------
    def interpret_collapse_chains(self) -> None:
        """Summarize lengths of collapse propagation chains."""
        path = self._path("collapse_chain_log.json")
        if not os.path.exists(path):
            return
        chains: Dict[str, int] = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                src = rec.get("source")
                length = len(rec.get("collapsed", []))
                chains[src] = max(chains.get(src, 0), length)
        if chains:
            self.summary["collapse_chains"] = chains

    # ------------------------------------------------------------
    def interpret_layer_transitions(self) -> None:
        """Aggregate layer transition counts per node."""
        path = self._path("layer_transition_log.json")
        if not os.path.exists(path):
            return
        totals: Dict[str, int] = {}
        by_node: Dict[str, Dict[str, int]] = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                frm = rec.get("from")
                to = rec.get("to")
                node = rec.get("node")
                if not frm or not to or not node:
                    continue
                key = f"{frm}->{to}"
                totals[key] = totals.get(key, 0) + 1
                by_node.setdefault(node, {}).setdefault(to, 0)
                by_node[node][to] += 1
        if totals:
            self.summary["layer_transitions"] = {
                "totals": totals,
                "by_node": by_node,
            }

    # ------------------------------------------------------------
    def interpret_rerouting(self) -> None:
        """Count tick rerouting events due to refraction."""
        path = self._path("refraction_log.json")
        if not os.path.exists(path):
            return
        counts = {"recursive": 0, "alt_path": 0}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("recursion_from"):
                    counts["recursive"] += 1
                if rec.get("via"):
                    counts["alt_path"] += 1
        if counts["recursive"] or counts["alt_path"]:
            self.summary["rerouting"] = counts

    # ------------------------------------------------------------
    def interpret_node_state_map(self) -> None:
        """Summarize node state transitions."""
        path = self._path("node_state_map.json")
        if not os.path.exists(path):
            return
        transitions: Dict[str, Dict[str, int]] = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                node = rec.get("node")
                frm = rec.get("from")
                to = rec.get("to")
                if not node or frm is None or to is None:
                    continue
                key = f"{frm}->{to}"
                transitions.setdefault(node, {}).setdefault(key, 0)
                transitions[node][key] += 1
        if transitions:
            self.summary["node_state_transitions"] = transitions

    # ------------------------------------------------------------
    def interpret_decoherence(self) -> None:
        path = self._path("decoherence_log.json")
        lines = self.load_json_lines(path)
        if not lines:
            return
        stats: Dict[str, Dict[str, float]] = {}
        for states in lines.values():
            for node, val in states.items():
                entry = stats.setdefault(node, {"max": -1.0, "min": 1e9})
                if val > entry["max"]:
                    entry["max"] = val
                if val < entry["min"]:
                    entry["min"] = val
        self.summary["decoherence"] = stats

    # ------------------------------------------------------------
    def interpret_clusters(self) -> None:
        path = self._path("cluster_log.json")
        lines = self.load_json_lines(path)
        if not lines:
            return
        first = None
        max_clusters = 0
        for t_str, clusters in lines.items():
            if clusters:
                if first is None:
                    first = int(t_str)
                max_clusters = max(max_clusters, len(clusters))
        self.summary["clusters"] = {
            "first_detected": first,
            "max_clusters": max_clusters,
        }

    # ------------------------------------------------------------
    def interpret_bridge_state(self) -> None:
        path = self._path("bridge_state_log.json")
        lines = self.load_json_lines(path)
        if not lines:
            return
        last_tick = max(int(t) for t in lines)
        self.summary["bridges"] = lines[str(last_tick)]

    # ------------------------------------------------------------
    def interpret_law_drift(self) -> None:
        path = self._path("law_drift_log.json")
        if not os.path.exists(path):
            return
        counts: Dict[str, int] = {}
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                node = rec.get("node")
                counts[node] = counts.get(node, 0) + 1
        if counts:
            self.summary["law_drift"] = counts

    # ------------------------------------------------------------
    def interpret_meta_nodes(self) -> None:
        path = self._path("meta_node_tick_log.json")
        lines = self.load_json_lines(path)
        if not lines:
            return
        counts: Dict[str, int] = {}
        for metas in lines.values():
            for meta_id in metas:
                counts[meta_id] = counts.get(meta_id, 0) + 1
        self.summary["meta_nodes"] = counts

    # ------------------------------------------------------------
    def interpret_tick_trace(self) -> None:
        path = self._path("tick_trace.json")
        if not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)
        counts = {
            nid: len(n.get("ticks", [])) for nid, n in data.get("nodes", {}).items()
        }
        layer_summary: Dict[str, Dict[str, int]] = {}
        for nid, n in data.get("nodes", {}).items():
            for tick in n.get("ticks", []):
                layer = tick.get("layer", "tick")
                layer_summary.setdefault(nid, {}).setdefault(layer, 0)
                layer_summary[nid][layer] += 1
        if counts:
            self.summary["tick_counts"] = counts
        if layer_summary:
            self.summary["layer_summary"] = layer_summary

    # ------------------------------------------------------------
    def interpret_inspection(self) -> None:
        path = self._path("inspection_log.json")
        if not os.path.exists(path):
            return
        with open(path) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                return
        self.summary["inspection_events"] = len(data)

    # ------------------------------------------------------------
    def interpret_console_log(self) -> None:
        path = self._path("cwt_console_output.txt")
        if not os.path.exists(path):
            return
        with open(path) as f:
            lines = f.readlines()
        ticks_logged = sum(1 for l in lines if l.startswith("== Tick"))
        self.summary["console"] = {
            "lines": len(lines),
            "ticks_logged": ticks_logged,
        }

    # ------------------------------------------------------------
    def generate_narrative(self) -> str:
        """Return a textual summary of the collected metrics."""

        from .services.serialization_service import NarrativeGeneratorService

        return NarrativeGeneratorService(self.summary).generate()

    # ------------------------------------------------------------
    def run(self) -> None:
        self.load_graph()
        self.interpret_curvature()
        self.interpret_collapse()
        self.interpret_collapse_origins()
        self.interpret_collapse_chains()
        self.interpret_coherence()
        self.interpret_law_wave()
        self.interpret_decoherence()
        self.interpret_layer_transitions()
        self.interpret_clusters()
        self.interpret_bridge_state()
        self.interpret_law_drift()
        self.interpret_meta_nodes()
        self.interpret_rerouting()
        self.interpret_node_state_map()
        self.interpret_tick_trace()
        self.interpret_inspection()
        self.interpret_console_log()
        out_path = self._path("interpretation_log.json")
        with open(out_path, "w") as f:
            json.dump(self.summary, f, indent=2)
        text_summary = self.generate_narrative()
        text_path = self._path("interpretation_summary.txt")
        with open(text_path, "w") as f:
            f.write(text_summary)
        print(f"✅ Interpretation saved to {out_path}")
        print(f"✅ Narrative saved to {text_path}")

        # Run causal analyst layer to produce explanations
        try:
            analyst = CausalAnalyst(output_dir=self.output_dir)
            analyst.run()
        except Exception as e:
            print(f"⚠️ Causal analyst failed: {e}")


def run_interpreter(
    output_dir: str | None = None, graph_path: str | None = None
) -> None:
    """Execute the multilayer log interpreter.

    Parameters
    ----------
    output_dir:
        Optional path to the directory containing log files. If ``None`` the
        default package output directory is used.
    graph_path:
        Optional path to the input graph JSON used for the run.
    """

    interpreter = CWTLogInterpreter(output_dir=output_dir, graph_path=graph_path)
    interpreter.run()


if __name__ == "__main__":
    run_interpreter()
