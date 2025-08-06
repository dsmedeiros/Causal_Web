from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class GraphSerializationService:
    """Serialize a ``CausalGraph`` into a dictionary."""

    graph: Any

    # ------------------------------------------------------------------
    def as_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation of the graph."""

        return {
            "nodes": self._nodes(),
            "superpositions": self._superpositions(),
            "edges": self._edges(),
            "bridges": self._bridges(),
            "tick_sources": self.graph.tick_sources,
            "meta_nodes": self._meta_nodes(),
        }

    # ------------------------------------------------------------------
    def _nodes(self) -> List[Dict[str, Any]]:
        nodes = []
        for nid, n in self.graph.nodes.items():
            nodes.append(
                {
                    "id": nid,
                    "x": n.x,
                    "y": n.y,
                    "ticks": [
                        {
                            "time": t.time,
                            "amplitude": t.amplitude,
                            "phase": t.phase,
                            "generation_tick": getattr(t, "generation_tick", 0),
                            "origin": t.origin,
                            "layer": getattr(t, "layer", "tick"),
                            "trace_id": getattr(t, "trace_id", ""),
                        }
                        for t in n.tick_history
                    ],
                    "phase": n.phase,
                    "coherence": n.coherence,
                    "decoherence": n.decoherence,
                    "frequency": n.frequency,
                    "refractory_period": n.refractory_period,
                    "base_threshold": n.base_threshold,
                    "collapse_origin": n.collapse_origin,
                    "is_classical": getattr(n, "is_classical", False),
                    "decoherence_streak": getattr(n, "_decoherence_streak", 0),
                    "last_tick_time": n.last_tick_time,
                    "subjective_ticks": n.subjective_ticks,
                    "law_wave_frequency": n.law_wave_frequency,
                    "trust_profile": n.trust_profile,
                    "phase_confidence": n.phase_confidence_index,
                    "goals": n.goals,
                    "origin_type": n.origin_type,
                    "generation_tick": n.generation_tick,
                    "parent_ids": n.parent_ids,
                    "node_type": n.node_type.value,
                    "coherence_credit": n.coherence_credit,
                    "decoherence_debt": n.decoherence_debt,
                    "phase_lock": n.phase_lock,
                }
            )
        return nodes

    # ------------------------------------------------------------------
    def _superpositions(self) -> Dict[str, Dict[str, List[float]]]:
        result = {}
        for nid, node in self.graph.nodes.items():
            if not node.pending_superpositions:
                continue
            result[nid] = {
                str(t): [
                    round(float(p[0] if isinstance(p, (tuple, list)) else p), 4)
                    for p in node.pending_superpositions[t]
                ]
                for t in node.pending_superpositions
            }
        return result

    # ------------------------------------------------------------------
    def _edges(self) -> List[Dict[str, Any]]:
        return [
            {
                "from": e.source,
                "to": e.target,
                "delay": e.delay,
                "attenuation": e.attenuation,
                "density": e.density,
                "phase_shift": e.phase_shift,
                "epsilon": getattr(e, "epsilon", False),
                "partner_id": getattr(e, "partner_id", None),
            }
            for e in self.graph.edges
        ]

    # ------------------------------------------------------------------
    def _bridges(self) -> List[Dict[str, Any]]:
        return [b.to_dict() for b in self.graph.bridges]

    # ------------------------------------------------------------------
    def _meta_nodes(self) -> Dict[str, Dict[str, Any]]:
        return {
            mid: {
                "members": m.member_ids,
                "constraints": m.constraints,
                "type": m.meta_type,
                "origin": m.origin,
                "collapsed": m.collapsed,
                "x": m.x,
                "y": m.y,
            }
            for mid, m in self.graph.meta_nodes.items()
        }


@dataclass
class NarrativeGeneratorService:
    """Build a human readable summary from interpreter metrics."""

    summary: Dict[str, Any]

    # ------------------------------------------------------------------
    def generate(self) -> str:
        """Return a formatted narrative summarizing the simulation."""

        lines: List[str] = []
        lines.extend(self._tick_summary())
        lines.extend(self._collapse_summary())
        lines.extend(self._coherence_summary())
        lines.extend(self._decoherence_summary())
        lines.extend(self._cluster_summary())
        lines.extend(self._law_drift_summary())
        lines.extend(self._bridge_summary())
        lines.extend(self._origin_summary())
        lines.extend(self._chain_summary())
        lines.extend(self._layer_summary())
        lines.extend(self._reroute_summary())
        lines.extend(self._inspection_summary())
        lines.extend(self._console_summary())
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def _tick_summary(self) -> List[str]:
        result = []
        ticks = self.summary.get("tick_counts", {})
        if ticks:
            total = sum(ticks.values())
            result.append(
                f"The simulation recorded {total} ticks across {len(ticks)} nodes."
            )
            for nid, cnt in ticks.items():
                result.append(f"- Node {nid} emitted {cnt} ticks.")
        return result

    # ------------------------------------------------------------------
    def _collapse_summary(self) -> List[str]:
        result = []
        collapse = self.summary.get("collapse")
        if collapse:
            parts = [f"{n} at tick {t}" for n, t in collapse.items()]
            result.append("Nodes collapsed to classical states: " + ", ".join(parts))
        return result

    # ------------------------------------------------------------------
    def _coherence_summary(self) -> List[str]:
        result = []
        coh = self.summary.get("coherence")
        if coh:
            for nid, data in coh.items():
                result.append(
                    f"- {nid} coherence ranged from {data['min']:.3f} to {data['max']:.3f}."
                )
        return result

    # ------------------------------------------------------------------
    def _decoherence_summary(self) -> List[str]:
        result = []
        deco = self.summary.get("decoherence")
        if deco:
            for nid, data in deco.items():
                result.append(
                    f"- {nid} decoherence ranged from {data['min']:.3f} to {data['max']:.3f}."
                )
        return result

    # ------------------------------------------------------------------
    def _cluster_summary(self) -> List[str]:
        result = []
        if "clusters" in self.summary:
            c = self.summary["clusters"]
            if c["first_detected"] is not None:
                result.append(
                    f"Clusters first appeared at tick {c['first_detected']} with up to {c['max_clusters']} cluster(s)."
                )
        return result

    # ------------------------------------------------------------------
    def _law_drift_summary(self) -> List[str]:
        result = []
        if "law_drift" in self.summary:
            events = sum(self.summary["law_drift"].values())
            result.append(f"Law drift events recorded: {events} total.")
        return result

    # ------------------------------------------------------------------
    def _bridge_summary(self) -> List[str]:
        result = []
        bridges = self.summary.get("bridges")
        if bridges:
            for b, data in bridges.items():
                state = "active" if data.get("active") else "inactive"
                result.append(
                    f"- Bridge {b} ended {state}; last rupture at {data.get('last_rupture_tick')}"
                )
        return result

    # ------------------------------------------------------------------
    def _origin_summary(self) -> List[str]:
        result = []
        if "collapse_origins" in self.summary:
            parts = [f"{n} at {t}" for n, t in self.summary["collapse_origins"].items()]
            result.append("Collapse origins: " + ", ".join(parts))
        return result

    # ------------------------------------------------------------------
    def _chain_summary(self) -> List[str]:
        result = []
        if "collapse_chains" in self.summary:
            for src, length in self.summary["collapse_chains"].items():
                result.append(f"- Collapse from {src} affected {length} nodes")
        return result

    # ------------------------------------------------------------------
    def _layer_summary(self) -> List[str]:
        result = []
        if "layer_transitions" in self.summary:
            total = sum(self.summary["layer_transitions"]["totals"].values())
            result.append(f"Layer transitions recorded: {total}")
        return result

    # ------------------------------------------------------------------
    def _reroute_summary(self) -> List[str]:
        result = []
        if "rerouting" in self.summary:
            r = self.summary["rerouting"]
            result.append(
                f"Rerouting events - recursive: {r['recursive']}, alt paths: {r['alt_path']}"
            )
        return result

    # ------------------------------------------------------------------
    def _inspection_summary(self) -> List[str]:
        result = []
        if "inspection_events" in self.summary:
            result.append(
                f"Recorded {self.summary['inspection_events']} superposition inspections."
            )
        return result

    # ------------------------------------------------------------------
    def _console_summary(self) -> List[str]:
        result = []
        if "console" in self.summary:
            c = self.summary["console"]
            result.append(
                f"Console log contains {c['lines']} lines covering {c['ticks_logged']} ticks."
            )
        return result
