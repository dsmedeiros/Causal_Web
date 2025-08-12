"""Parameter normalisation utilities."""

from __future__ import annotations

from typing import Dict


class Normalizer:
    """Map raw parameters to dimensionless groups and back."""

    def to_groups(self, raw: Dict[str, float]) -> Dict[str, float]:
        """Convert raw parameters to dimensionless groups."""

        return {
            "Delta_over_W0": raw["Delta"] / raw["W0"],
            "alpha_d_over_leak": raw["alpha_d"] / raw["alpha_leak"],
        }

    def to_raw(
        self, base: Dict[str, float], groups: Dict[str, float]
    ) -> Dict[str, float]:
        """Reconstruct raw parameters from groups using a baseline config."""

        raw = dict(base)
        if "Delta_over_W0" in groups:
            raw["Delta"] = groups["Delta_over_W0"] * raw["W0"]
        if "alpha_d_over_leak" in groups:
            raw["alpha_d"] = groups["alpha_d_over_leak"] * raw["alpha_leak"]
        return raw
