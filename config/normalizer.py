"""Parameter normalisation utilities."""

from __future__ import annotations

from typing import Dict


class Normalizer:
    """Map raw parameters to dimensionless groups and back."""

    def to_groups(self, raw: Dict[str, float]) -> Dict[str, float]:
        """Convert raw parameters to dimensionless groups."""

        delta = raw["Delta"]
        w0 = raw["W0"]
        alpha_d = raw["alpha_d"]
        leak = raw["leak"]
        return {
            "Delta_over_W0": delta / w0,
            "alpha_d_over_leak": alpha_d / leak,
        }

    def to_raw(self, groups: Dict[str, float]) -> Dict[str, float]:
        """Reconstruct raw parameters from groups using nominal scales."""

        # The nominal scales are arbitrary for demonstration purposes.
        return {
            "Delta": groups.get("Delta_over_W0", 0.0) * 1.0,
            "W0": 1.0,
            "alpha_d": groups.get("alpha_d_over_leak", 0.0) * 1.0,
            "leak": 1.0,
        }
