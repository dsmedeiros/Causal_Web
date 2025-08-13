from Causal_Web.analysis.twin import TwinNode
from Causal_Web.engine import scheduler


def test_proper_time_dilation():
    stationary = TwinNode("A")
    traveller = TwinNode("B")
    dt = 1.0
    steps = 5
    v = 0.6
    for _ in range(steps):
        traveller.x += v * dt
        scheduler.step([stationary, traveller], dt)
    for _ in range(steps):
        traveller.x -= v * dt
        scheduler.step([stationary, traveller], dt)
    assert stationary.tau > traveller.tau
    asym = (stationary.tau - traveller.tau) / stationary.tau
    assert asym > 0.05
