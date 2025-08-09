from Causal_Web.engine.engine_v2.scheduler import DepthScheduler
from Causal_Web.engine.engine_v2.state import Packet


def test_scheduler_deterministic_order_and_peek():
    sched = DepthScheduler()
    # push out-of-order depths
    sched.push(3, 0, 0, Packet(0, 0))
    sched.push(1, 1, 0, Packet(0, 1))
    sched.push(2, 2, 0, Packet(0, 2))

    assert sched.peek_depth() == 1
    depths = []
    while sched:
        depth, dst, edge, pkt = sched.pop()
        depths.append(depth)
    assert depths == [1, 2, 3]


def test_scheduler_tie_breaking():
    sched = DepthScheduler()
    sched.push(5, 2, 1, Packet(0, 2))
    sched.push(5, 1, 2, Packet(0, 1))
    sched.push(5, 1, 1, Packet(0, 1))

    results = []
    while sched:
        depth, dst, edge, pkt = sched.pop()
        results.append((depth, dst, edge))

    assert results == [(5, 1, 1), (5, 1, 2), (5, 2, 1)]
