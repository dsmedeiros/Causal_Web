import random

from Causal_Web.analysis.twin import TwinNode
from Causal_Web.engine.horizon import HorizonThermodynamics


def test_page_curve_behavior():
    rng = random.Random(0)
    horizon = HorizonThermodynamics(temperature=2.0, rng=rng)
    nodes = [TwinNode(str(i)) for i in range(3)]
    for n in nodes:
        horizon.register(n, energy=2.0)

    for _ in range(1000):
        horizon.step(nodes)
        if horizon.emitted_quanta >= horizon.total_quanta:
            break

    entropy = horizon.outside_entropy
    peak = max(entropy)
    peak_idx = entropy.index(peak)
    assert peak_idx < len(entropy) - 1
    # Final entropy should return to zero but allow for floating point tolerance
    assert abs(entropy[-1]) < 1e-10
    assert all(entropy[i] <= entropy[i + 1] for i in range(peak_idx))
    assert all(entropy[i] >= entropy[i + 1] for i in range(peak_idx, len(entropy) - 1))
