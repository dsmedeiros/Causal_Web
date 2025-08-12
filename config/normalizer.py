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
            "sigma_reinforce_over_decay": raw["sigma_reinforce"] / raw["lambda_decay"],
            "a_over_b": raw["a"] / raw["b"],
            "eta_times_W0": raw["eta"] * raw["W0"],
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
        if "sigma_reinforce_over_decay" in groups:
            raw["sigma_reinforce"] = (
                groups["sigma_reinforce_over_decay"] * raw["lambda_decay"]
            )
        if "a_over_b" in groups:
            raw["a"] = groups["a_over_b"] * raw["b"]
        if "eta_times_W0" in groups:
            raw["eta"] = groups["eta_times_W0"] / raw["W0"]
        return raw
