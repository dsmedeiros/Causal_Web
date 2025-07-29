from __future__ import annotations

import json
import os
from typing import Dict, List

from .causal_analyst import ExplanationEvent, CausalAnalyst
from ..models.base import OutputDirMixin


class ExplanationRuleMatcher(OutputDirMixin):
    """Service object to evaluate explanation rules."""

    def __init__(self, analyst: CausalAnalyst) -> None:
        super().__init__(output_dir=analyst.output_dir)
        self.analyst = analyst
        self.logs = analyst.logs
        self.explanations: List[ExplanationEvent] = analyst.explanations

    # lifecycle stages
    def run(self) -> None:
        """Execute all explanation-matching stages."""
        self._match_decoherence_induced_collapse()
        self._match_law_wave_stabilization()
        self._match_meta_node_onset()
        self._match_layer_transitions()
        self._match_collapse_origin()
        self._match_collapse_chain()
        self._match_emergence_events()
        self._match_propagation_failures()

    # rule implementations
    def _match_decoherence_induced_collapse(self) -> None:
        collapse_events = self.analyst._collapse_events()
        spikes = self.analyst.transitions.get("decoherence_spikes", {})
        for node, tick in collapse_events.items():
            candidate = [
                s for s in spikes.get(node, []) if s[1] >= tick - 1 and s[0] <= tick - 3
            ]
            if candidate:
                start, end, mx = candidate[-1]
                text = (
                    f"Node {node} collapsed after sustained decoherence (max {mx:.2f}) "
                    f"between ticks {start}-{end}."
                )
                self.explanations.append(
                    ExplanationEvent(
                        (start, tick), [node], "rule:decoherence_induced_collapse", text
                    )
                )

    def _match_law_wave_stabilization(self) -> None:
        law_wave: Dict[int, Dict] = self.logs.get("law_wave", {})
        for node in {n for tick in law_wave for n in law_wave[tick]}:
            values = [
                law_wave[t][node] for t in sorted(law_wave) if node in law_wave[t]
            ]
            window: List[float] = []
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
                            ExplanationEvent(
                                (start_tick, end_tick),
                                [node],
                                "rule:law_wave_stabilization",
                                text,
                            )
                        )
                        break

    def _match_meta_node_onset(self) -> None:
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
                        ExplanationEvent(
                            (tick, tick), list(cluster), "rule:meta_node_onset", text
                        )
                    )

    def _match_layer_transitions(self) -> None:
        transitions = self.logs.get("layer_transitions", {})
        for t in sorted(transitions):
            for ev in transitions[t]:
                if ev.get("from") == "decoherence" and ev.get("to") == "collapse":
                    text = f"Node {ev['node']} transitioned from decoherence to collapse at tick {t}."
                    self.explanations.append(
                        ExplanationEvent(
                            (t, t), [ev["node"]], "rule:layer_transition", text
                        )
                    )

    def _match_collapse_origin(self) -> None:
        front = self.logs.get("collapse_front", {})
        for tick, info in front.items():
            if info.get("event") == "collapse_start":
                node = info.get("node")
                text = f"Collapse initiated at node {node}"
                self.explanations.append(
                    ExplanationEvent((tick, tick), [node], "rule:collapse_origin", text)
                )

    def _match_collapse_chain(self) -> None:
        chains = self.logs.get("collapse_chain", {})
        for tick, info in chains.items():
            src = info.get("source")
            collapsed = [c.get("node") for c in info.get("collapsed", [])]
            if src and collapsed:
                text = f"Collapse from {src} propagated to {', '.join(collapsed)} at tick {tick}."
                self.explanations.append(
                    ExplanationEvent(
                        (tick, tick), [src] + collapsed, "rule:collapse_chain", text
                    )
                )
            children = info.get("children_spawned")
            if src and children:
                text = f"Node {children[0]} condensed from chaos following collapse of MetaNode {src}."
                self.explanations.append(
                    ExplanationEvent(
                        (tick, tick), children, "rule:csp_generation", text
                    )
                )

    def _match_emergence_events(self) -> None:
        emergence_path = self._path("node_emergence_log.json")
        if not os.path.exists(emergence_path):
            return
        with open(emergence_path) as f:
            for line in f:
                rec = json.loads(line)
                if (
                    rec.get("origin_type") == "SIP_RECOMB"
                    and len(rec.get("parents", [])) == 2
                ):
                    c = rec.get("id")
                    parents = rec.get("parents", [])
                    tick = rec.get("tick", 0)
                    bridge_state = self.logs.get("bridge_state", {}).get(tick, {})
                    bridge = bridge_state.get(
                        f"{parents[0]}->{parents[1]}"
                    ) or bridge_state.get(f"{parents[1]}->{parents[0]}")
                    trust = bridge.get("trust_score") if bridge else None
                    freq = rec.get("sigma_phi")
                    text = (
                        f"Node {c} was generated by recombination between {parents[0]} and {parents[1]} "
                        f"across a stable high-trust bridge. The resulting law-wave frequency is an adaptive blend "
                        f"of its parents ({freq:.3f})."
                    )
                    self.explanations.append(
                        ExplanationEvent((tick, tick), [c], "rule:sip_recomb", text)
                    )
                elif rec.get("origin_type") == "CSP":
                    c = rec.get("id")
                    tick = rec.get("tick", 0)
                    collapse_id = None
                    collapse_log_path = self._path("collapse_chain_log.json")
                    if os.path.exists(collapse_log_path):
                        with open(collapse_log_path) as cf:
                            for ln in cf:
                                evt = json.loads(ln)
                                if c in evt.get("children_spawned", []):
                                    collapse_id = evt.get(
                                        "collapsed_entity"
                                    ) or evt.get("source")
                                    break
                    text = (
                        f"Node {c} condensed from chaotic causal potential released by the collapse of {collapse_id}. "
                        "It was unstable at birth with high phase variance and low confidence."
                    )
                    self.explanations.append(
                        ExplanationEvent((tick, tick), [c], "rule:csp_generation", text)
                    )

    def _match_propagation_failures(self) -> None:
        fail_path = self._path("propagation_failure_log.json")
        if not os.path.exists(fail_path):
            return
        with open(fail_path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("type") == "SIP_FAILURE":
                    node = rec.get("child")
                    parent = rec.get("parent")
                    text = (
                        f"Node {node} failed to stabilize after SIP attempt by {parent}. "
                        "Its collapse increased local decoherence, weakening the surrounding structure."
                    )
                    self.explanations.append(
                        ExplanationEvent(
                            (rec.get("tick", 0), rec.get("tick", 0)),
                            [node],
                            "rule:sip_failure",
                            text,
                        )
                    )
                elif rec.get("type") == "CSP_FAILURE":
                    collapsed = rec.get("parent")
                    text = (
                        f"A condensation seed near collapse site {collapsed} failed to accumulate coherence and dissolved. "
                        "The tick energy dissipated into ambient decoherence."
                    )
                    self.explanations.append(
                        ExplanationEvent(
                            (rec.get("tick", 0), rec.get("tick", 0)),
                            [collapsed],
                            "rule:csp_seed_dissolution",
                            text,
                        )
                    )
