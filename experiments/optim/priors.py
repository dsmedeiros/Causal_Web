from __future__ import annotations

"""Prior helpers for optimizers."""

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np


class Prior:
    """Base class for parameter priors."""

    def sample(self, rng: np.random.Generator) -> float:
        raise NotImplementedError


@dataclass
class GaussianPrior(Prior):
    """Gaussian prior for continuous parameters."""

    mu: float
    sigma: float

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.normal(self.mu, self.sigma))


@dataclass
class DiscretePrior(Prior):
    """Discrete prior with explicit probabilities."""

    values: Sequence[float]
    probs: Sequence[float]

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.choice(self.values, p=self.probs))


def build_priors(
    topk_rows: Sequence[Dict[str, float]],
    bins: int | None = None,
) -> Dict[str, Prior]:
    """Build simple priors from top-k configurations.

    Parameters
    ----------
    topk_rows:
        Sequence of configuration dictionaries taken from Top-K results.
    bins:
        Optional number of quantile bins for continuous parameters. When
        provided, continuous values are converted to a :class:`DiscretePrior`
        with ``bins`` equally spaced quantiles.  When ``None`` the continuous
        values are modelled with a :class:`GaussianPrior`.
    """

    priors: Dict[str, Prior] = {}
    if not topk_rows:
        return priors
    keys = topk_rows[0].keys()
    for k in keys:
        vals = [row[k] for row in topk_rows]
        uniq_vals = set(vals)
        all_int = all(isinstance(v, (int, np.integer)) for v in vals)
        if all_int and len(uniq_vals) <= len(vals):
            uniq, counts = np.unique(vals, return_counts=True)
            probs = counts / counts.sum()
            priors[k] = DiscretePrior(list(uniq), list(probs))
        else:
            arr = np.array(vals, dtype=float)
            if bins and bins > 0:
                qs = np.linspace(0, 1, bins + 2)[1:-1]
                centres = np.quantile(arr, qs)
                probs = np.full(len(centres), 1.0 / len(centres))
                priors[k] = DiscretePrior(list(map(float, centres)), list(probs))
            else:
                mu = float(arr.mean())
                sigma = float(arr.std() or 1.0)
                priors[k] = GaussianPrior(mu, sigma)
    return priors
