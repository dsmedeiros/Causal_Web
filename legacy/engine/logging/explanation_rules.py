from __future__ import annotations

import json
import os
from typing import Dict, List

from .causal_analyst import ExplanationEvent, CausalAnalyst
from ..models.base import OutputDirMixin, JsonLinesMixin


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
        events = JsonLinesMixin.filter_event_log(
            self._path("events_log.jsonl"), "node_emergence_log"
        )
        if not events:
            return

        collapse_events = JsonLinesMixin.filter_event_log(
            self._path("events_log.jsonl"), "collapse_chain_log"
        )

        for tick, lst in events.items():
            for rec in lst:
                payload = rec.get("payload", {})
                value = payload.get("value", {})
                origin = value.get("origin_type")
                parents = value.get("parents", [])
                child = value.get("node_id") or value.get("id")
                freq = value.get("sigma_phi")

                if origin == "SIP_RECOMB" and len(parents) == 2:
                    bridge_state = self.logs.get("bridge_state", {}).get(tick, {})
                    bridge = bridge_state.get(
                        f"{parents[0]}->{parents[1]}"
                    ) or bridge_state.get(f"{parents[1]}->{parents[0]}")
                    trust = bridge.get("trust_score") if bridge else None
                    text = (
                        f"Node {child} was generated by recombination between {parents[0]} and {parents[1]} "
                        f"across a stable high-trust bridge. The resulting law-wave frequency is an adaptive blend "
                        f"of its parents ({freq:.3f})."
                    )
                    self.explanations.append(
                        ExplanationEvent((tick, tick), [child], "rule:sip_recomb", text)
                    )
                    continue

                if origin == "CSP":
                    collapse_id = None
                    for cl_tick, cl_list in collapse_events.items():
                        for evt in cl_list:
                            cpayload = evt.get("payload", {})
                            if child in cpayload.get("children_spawned", []):
                                collapse_id = cpayload.get(
                                    "collapsed_entity"
                                ) or cpayload.get("source")
                                break
                        if collapse_id:
                            break
                    text = (
                        f"Node {child} condensed from chaotic causal potential released by the collapse of {collapse_id}. "
                        "It was unstable at birth with high phase variance and low confidence."
                    )
                    self.explanations.append(
                        ExplanationEvent(
                            (tick, tick), [child], "rule:csp_generation", text
                        )
                    )

    def _match_propagation_failures(self) -> None:
        events = JsonLinesMixin.filter_event_log(
            self._path("events_log.jsonl"), "propagation_failure_log"
        )
        if not events:
            return
        for tick, lst in events.items():
            for rec in lst:
                payload = rec.get("payload", {})
                if payload.get("type") == "SIP_FAILURE":
                    node = payload.get("child")
                    parent = payload.get("parent")
                    text = (
                        f"Node {node} failed to stabilize after SIP attempt by {parent}. "
                        "Its collapse increased local decoherence, weakening the surrounding structure."
                    )
                    self.explanations.append(
                        ExplanationEvent((tick, tick), [node], "rule:sip_failure", text)
                    )
                elif payload.get("type") == "CSP_FAILURE":
                    collapsed = payload.get("parent")
                    text = (
                        f"A condensation seed near collapse site {collapsed} failed to accumulate coherence and dissolved. "
                        "The tick energy dissipated into ambient decoherence."
                    )
                    self.explanations.append(
                        ExplanationEvent(
                            (tick, tick), [collapsed], "rule:csp_seed_dissolution", text
                        )
                    )
