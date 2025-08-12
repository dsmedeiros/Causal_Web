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
        depth, dst, edge, _seq, pkt = sched.pop()
        depths.append(depth)
    assert depths == [1, 2, 3]


def test_scheduler_tie_breaking():
    sched = DepthScheduler()
    sched.push(5, 2, 1, Packet(0, 2))
    sched.push(5, 1, 2, Packet(0, 1))
    sched.push(5, 1, 1, Packet(0, 1))

    results = []
    while sched:
        depth, dst, edge, _seq, pkt = sched.pop()
        results.append((depth, dst, edge))

    assert results == [(5, 1, 1), (5, 1, 2), (5, 2, 1)]


def test_scheduler_same_dst_edge_seq_order():
    """Lock tie-breaking order for simultaneous arrivals.

    When multiple packets share the same depth and destination, the scheduler
    must fall back to `(dst, edge_id, seq)` ordering so processing is
    deterministic.  A previous bug left this ordering undefined, producing
    flaky behaviours depending on insertion order.  This test fixes the rule
    by asserting edge identifiers break the tie before packet sequence.
    """

    sched = DepthScheduler()
    sched.push(5, 1, 2, Packet(0, 0, "a"))
    sched.push(5, 1, 1, Packet(0, 0, "b"))
    sched.push(5, 1, 1, Packet(0, 0, "c"))

    popped = []
    while sched:
        depth, dst, edge, _seq, pkt = sched.pop()
        popped.append((dst, edge, pkt.payload))

    assert popped == [(1, 1, "b"), (1, 1, "c"), (1, 2, "a")]


def test_scheduler_two_packet_edge_order():
    """Ensure deterministic ordering for identical depths and destinations."""

    sched = DepthScheduler()
    sched.push(5, 1, 2, Packet(0, 0))
    sched.push(5, 1, 1, Packet(0, 0))

    popped = []
    while sched:
        depth, dst, edge, _seq, pkt = sched.pop()
        popped.append((dst, edge))

    assert popped == [(1, 1), (1, 2)]
