from __future__ import annotations

from dataclasses import dataclass
import math
from random import random
from typing import Optional, TYPE_CHECKING
import json
from enum import Enum
import uuid

from ...config import Config
from .base import LoggingMixin

if TYPE_CHECKING:
    from .node import Node
    from .graph import CausalGraph

from enum import Enum


class BridgeState(Enum):
    FORMING = "forming"
    STABLE = "stable"
    STRAINED = "strained"
    RUPTURING = "rupturing"
    RUPTURED = "ruptured"
    DORMANT = "dormant"


class BridgeType(Enum):
    """Available bridging mechanisms between nodes."""

    BRAIDED = "braided"
    MIRROR = "mirror"
    UNIDIRECTIONAL = "unidirectional"


class MediumType(Enum):
    """Transport mediums for bridge propagation."""

    STANDARD = "standard"


@dataclass
class BridgeEvent:
    tick: int
    event_type: str
    bridge_id: str
    source: str
    target: str
    coherence_at_event: Optional[float] = None


class Bridge(LoggingMixin):
    def __init__(
        self,
        node_a_id: str,
        node_b_id: str,
        bridge_type: BridgeType | str = BridgeType.BRAIDED,
        phase_offset: float = 0.0,
        drift_tolerance: float | None = None,
        decoherence_limit: float | None = None,
        initial_strength: float = 1.0,
        medium_type: MediumType | str = MediumType.STANDARD,
        mutable: bool = True,
        seeded: bool = True,
        formed_at_tick: int = 0,
        *,
        is_entangled: bool = False,
        entangled_id: str | None = None,
    ) -> None:
        """Create a new bridge between two node identifiers."""
        self.node_a_id = node_a_id
        self.node_b_id = node_b_id
        if isinstance(bridge_type, str):
            bridge_type = BridgeType(bridge_type)
        if isinstance(medium_type, str):
            medium_type = MediumType(medium_type)
        self.bridge_type = bridge_type
        self.phase_offset = phase_offset  # For inverted or phase-shifted mirroring
        self.drift_tolerance = drift_tolerance
        self.decoherence_limit = decoherence_limit
        self.initial_strength = initial_strength
        self.current_strength = initial_strength
        self.medium_type = medium_type
        self.mutable = mutable
        self.seeded = seeded
        self.formed_at_tick = formed_at_tick
        self.tick_load = 0
        self.phase_drift = 0.0
        self.coherence_flux = 0.0
        self.last_activation = None
        self.active = True
        self.state = BridgeState.FORMING
        self.fatigue = 0.0
        self.last_rupture_tick = None
        self.last_reform_tick = None
        self.coherence_at_reform = None
        self.bridge_id = f"{self.node_a_id}->{self.node_b_id}"

        # ---- Phase 4 additions ----
        self.rupture_history = []  # list of (tick, decoherence)
        self.reinforcement_streak = 0
        self.decoherence_exposure = []  # recent avg decoherence values
        self.trust_score = 0.5

        # ---- Phase 6 additions ----
        self.last_active_tick = formed_at_tick
        self.reformable = True
        self.memory_weight = 0.5

        self.is_entangled = is_entangled
        if is_entangled:
            self.entangled_id = entangled_id or str(uuid.uuid4())
        else:
            self.entangled_id = None

    def _fatigue_multiplier(self, tick: int) -> float:
        """Return dynamic fatigue scaling factor during early stabilization."""
        ramp = getattr(Config, "BRIDGE_STABILIZATION_TICKS", 50)
        progress = min(max(tick - self.formed_at_tick, 0), ramp) / ramp
        return 0.5 + 0.5 * progress

    def _reform_threshold(self, tick: int) -> float:
        """Return dynamic coherence threshold for bridge reformation."""
        ramp = getattr(Config, "BRIDGE_STABILIZATION_TICKS", 50)
        progress = min(max(tick - self.formed_at_tick, 0), ramp) / ramp
        return 0.75 + 0.15 * progress

    def _log_dynamics(
        self, tick: int, event: str, conditions: dict | None = None
    ) -> None:
        record = {
            "bridge_id": self.bridge_id,
            "source": self.node_a_id,
            "target": self.node_b_id,
            "event": event,
            "tick": tick,
            "seeded": self.seeded,
        }
        if conditions is not None:
            record["conditions"] = conditions
        self._log("bridge_dynamics_log.json", record)

    def update_state(self, tick: int) -> None:
        """Update ``self.state`` based on fatigue and activation."""
        old = self.state
        if not self.active:
            if self.current_strength > 0 and self.fatigue <= 3.0:
                self.state = BridgeState.DORMANT
            else:
                self.state = BridgeState.RUPTURED
        else:
            avg = (
                sum(self.decoherence_exposure[-5:])
                / len(self.decoherence_exposure[-5:])
                if self.decoherence_exposure
                else 0.0
            )
            self.fatigue += avg * self._fatigue_multiplier(tick)
            if self.fatigue > 3.0:
                self.state = BridgeState.RUPTURING
                self.active = False
            elif self.fatigue > 1.5:
                self.state = BridgeState.STRAINED
            else:
                self.state = BridgeState.STABLE
        if old != self.state:
            self._log_dynamics(tick, self.state.value, {"from": old.value})
            if self.state in {BridgeState.RUPTURING, BridgeState.RUPTURED}:
                avg_decoh = (
                    sum(self.decoherence_exposure[-5:])
                    / len(self.decoherence_exposure[-5:])
                    if self.decoherence_exposure
                    else 0.0
                )
                avg_coh = 1 - avg_decoh
                self._log_rupture(tick, "fatigue", avg_coh)

    def _log_event(self, tick: int, event_type: str, value: float | None) -> None:
        event = BridgeEvent(
            tick=tick,
            event_type=event_type,
            bridge_id=self.bridge_id,
            source=self.node_a_id,
            target=self.node_b_id,
            coherence_at_event=value,
        )
        self._log("event_log.json", event.__dict__)

    def _log_rupture(self, tick: int, reason: str, coherence: float | None) -> None:
        record = {
            "tick": tick,
            "bridge": self.bridge_id,
            "source": self.node_a_id,
            "target": self.node_b_id,
            "reason": reason,
            "coherence": round(coherence, 4) if coherence is not None else None,
            "fatigue": round(self.fatigue, 3),
        }
        self._log("bridge_rupture_log.json", record)

    def probabilistic_bridge_failure(
        self,
        decoherence_strength: float,
        rupture_threshold: float = 0.3,
        rupture_prob: float = 0.9,
    ) -> bool:
        """Return ``True`` if decoherence causes the bridge to rupture."""
        if decoherence_strength > rupture_threshold and random() < rupture_prob:
            print(
                f"[BRIDGE] Probabilistic rupture at tick due to decoherence={decoherence_strength:.2f}"
            )
            self.active = False
            return True
        return False

    # ---- Phase 6: plasticity ----
    def decay(self, tick_time: int, inactive_threshold: int = 5) -> None:
        """Reduce strength when inactive for ``inactive_threshold`` ticks."""
        if self.last_active_tick is None:
            self.last_active_tick = tick_time
        if (
            tick_time - self.last_active_tick >= inactive_threshold
            and self.current_strength > 0
        ):
            self.current_strength = max(0.0, self.current_strength - 0.1)
            duration = tick_time - self.last_active_tick
            self._log(
                "bridge_decay_log.json",
                {
                    "tick": tick_time,
                    "bridge": self.bridge_id,
                    "strength": self.current_strength,
                    "duration": duration,
                },
            )
            if self.current_strength == 0.0:
                self.active = False
                from . import tick_engine as te

                te._decay_durations.append(duration)
                self._log_dynamics(tick_time, "decayed")

    def try_reform(
        self,
        tick_time: int,
        node_a: "Node",
        node_b: "Node",
        coherence_threshold: float = 0.9,
    ) -> None:
        """Reactivate a ruptured bridge when coherence is high enough."""
        if not self.reformable or self.active:
            return
        coherence = (
            node_a.compute_coherence_level(tick_time)
            + node_b.compute_coherence_level(tick_time)
        ) / 2
        threshold = self._reform_threshold(tick_time)
        if coherence > threshold and random() < self.memory_weight:
            self.active = True
            self.last_reform_tick = tick_time
            self.coherence_at_reform = coherence
            self._log(
                "bridge_reformation_log.json",
                {
                    "tick": tick_time,
                    "bridge": self.bridge_id,
                    "coherence": coherence,
                },
            )
            from . import tick_engine as te

            te.bridges_reformed_count += 1
            self._log_event(tick_time, "bridge_reformed", coherence)
            self._log_dynamics(tick_time, "recovered", {"coherence": coherence})

    def try_reactivate(
        self,
        tick_time: int,
        node_a: "Node",
        node_b: "Node",
        coherence_threshold: float = 0.9,
    ) -> None:
        """Re-enable an inactive bridge when coherence improves."""
        coherence = (
            node_a.compute_coherence_level(tick_time)
            + node_b.compute_coherence_level(tick_time)
        ) / 2
        threshold = self._reform_threshold(tick_time)
        if not self.active and coherence > threshold:
            self.active = True
            self.last_reform_tick = tick_time
            self.coherence_at_reform = coherence
            print(
                f"[BRIDGE] Reactivated at tick {tick_time} with coherence={coherence:.2f}"
            )
            self._log_event(tick_time, "bridge_reformed", coherence)
            self._log_dynamics(tick_time, "recovered", {"coherence": coherence})

    def apply(self, tick_time: int, graph: "CausalGraph") -> None:
        """Apply the bridge logic for ``tick_time``."""
        from .services.sim_services import BridgeApplyService

        BridgeApplyService(self, tick_time, graph).process()

    def to_dict(self) -> dict:
        """Return a serialization-friendly representation of the bridge."""
        return {
            "source": self.node_a_id,
            "target": self.node_b_id,
            "type": self.bridge_type.value,
            "phase_offset": self.phase_offset,
            "drift_tolerance": self.drift_tolerance,
            "decoherence_limit": self.decoherence_limit,
            "initial_strength": self.initial_strength,
            "current_strength": self.current_strength,
            "medium_type": self.medium_type.value,
            "mutable": self.mutable,
            "seeded": self.seeded,
            "formed_at_tick": self.formed_at_tick,
            "active": self.active,
            "last_activation": self.last_activation,
            "last_rupture_tick": self.last_rupture_tick,
            "last_reform_tick": self.last_reform_tick,
            "coherence_at_reform": self.coherence_at_reform,
            "trust_score": self.trust_score,
            "reinforcement_streak": self.reinforcement_streak,
            "last_active_tick": self.last_active_tick,
            "reformable": self.reformable,
            "memory_weight": self.memory_weight,
            "is_entangled": self.is_entangled,
            "entangled_id": self.entangled_id,
            "avg_decoherence": (
                sum(self.decoherence_exposure) / len(self.decoherence_exposure)
                if self.decoherence_exposure
                else 0.0
            ),
            "state": self.state.value,
            "fatigue": round(self.fatigue, 3),
            "tick_load": self.tick_load,
            "phase_drift": round(self.phase_drift, 4),
            "coherence_flux": round(self.coherence_flux, 4),
        }
